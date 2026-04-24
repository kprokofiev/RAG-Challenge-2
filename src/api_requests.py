import os
import json
from dataclasses import dataclass
from dotenv import load_dotenv
from typing import Any, Union, List, Dict, Type, Optional, Literal
from openai import OpenAI
import asyncio
from src.api_request_parallel_processor import process_api_requests_from_file
from openai.lib._parsing import type_to_response_format_param 
import tiktoken
import src.prompts as prompts
import requests
from json_repair import repair_json
from pydantic import BaseModel
import google.generativeai as genai
from copy import deepcopy
from tenacity import retry, stop_after_attempt, wait_fixed
from src.openai_model_router import (
    build_budget_trace,
    commit_routed_usage,
    choose_routed_model,
    extract_usage_metrics,
    is_quota_exhausted_error,
    mark_tier_exhausted,
    next_tier_index,
    record_usage,
    release_routed_reservation,
    reserve_routed_model,
)
from src.exec_llm_env import require_exec_openai_api_key



def _get_llm_timeout_seconds() -> float:
    raw = os.getenv("DDKIT_LLM_TIMEOUT_SECONDS", "120")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 120.0


def _get_llm_max_retries() -> int:
    raw = os.getenv("DDKIT_LLM_MAX_RETRIES", "0")
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def _estimate_text_tokens(value: str) -> int:
    try:
        encoding = tiktoken.get_encoding("o200k_base")
        return len(encoding.encode(value or ""))
    except Exception:
        return max(1, len(value or "") // 4)


def _map_thinking_mode_to_effort(thinking_mode: Optional[str]) -> Optional[str]:
    mode = str(thinking_mode or "").strip().lower()
    if mode in {"", "off"}:
        return "none"
    if mode in {"low", "medium", "high", "xhigh", "minimal", "none"}:
        return mode
    return "medium"


def _extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)
    parts: List[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(str(text))
            elif isinstance(content, dict) and content.get("text"):
                parts.append(str(content.get("text")))
    return "\n".join(parts)


def _extract_response_parsed(response: Any, response_format: Type[BaseModel]) -> Optional[BaseModel]:
    parsed_output = getattr(response, "output_parsed", None)
    if parsed_output is not None:
        return parsed_output
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            parsed = getattr(content, "parsed", None)
            if parsed is not None:
                return response_format.model_validate(parsed)
            if isinstance(content, dict) and content.get("parsed") is not None:
                return response_format.model_validate(content.get("parsed"))
    return None


def _response_incomplete_reason(response: Any) -> str:
    incomplete = getattr(response, "incomplete_details", None)
    if incomplete is None and isinstance(response, dict):
        incomplete = response.get("incomplete_details")
    if not incomplete:
        return ""
    reason = getattr(incomplete, "reason", None)
    if reason is None and isinstance(incomplete, dict):
        reason = incomplete.get("reason")
    return str(reason or "").strip().lower()


def _extract_reasoning_summary(response: Any) -> str:
    reasoning = getattr(response, "reasoning", None)
    if reasoning is None and isinstance(response, dict):
        reasoning = response.get("reasoning")
    if not reasoning:
        return ""
    summary = getattr(reasoning, "summary", None)
    if summary is None and isinstance(reasoning, dict):
        summary = reasoning.get("summary")
    if not summary:
        return ""
    parts: List[str] = []
    if isinstance(summary, list):
        for item in summary:
            text = getattr(item, "text", None)
            if text:
                parts.append(str(text))
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item.get("text")))
    elif isinstance(summary, str):
        parts.append(summary)
    return "\n".join(parts)


def _reasoning_summary_mode() -> str:
    value = str(os.getenv("DDKIT_EXEC_REASONING_SUMMARY") or "concise").strip().lower()
    if value in {"auto", "concise", "detailed"}:
        return value
    return "concise"


def _estimated_exec_completion_tokens(max_output_tokens: Optional[int]) -> int:
    if max_output_tokens is not None:
        return max(256, int(max_output_tokens))
    raw = (os.getenv("DDKIT_EXEC_REASONING_ESTIMATED_OUTPUT_TOKENS") or "8000").strip()
    try:
        return max(256, int(raw))
    except ValueError:
        return 8000


def _is_truncated_structured_output_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in text
        for marker in (
            "max_output_tokens",
            "incomplete",
            "eof while parsing",
            "json_invalid",
            "invalid json",
            "unterminated string",
            "truncated",
            "could not parse response",
        )
    )


@dataclass
class ExecReasoningCallResult:
    parsed_output: Any
    raw_response: Any
    model_selected: str
    budget_trace: Dict[str, Any]
    reasoning_summary: str = ""


def call_exec_reasoning_model(
    system_content: str,
    human_content: str,
    response_format: Type[BaseModel],
    requested_model: Optional[str] = None,
    thinking_mode: Optional[str] = None,
    max_output_tokens: int = 2400,
    metadata: Optional[Dict[str, Any]] = None,
    block_class: Optional[str] = None,
) -> ExecReasoningCallResult:
    api_key = require_exec_openai_api_key()
    client = OpenAI(
        api_key=api_key,
        timeout=_get_llm_timeout_seconds(),
        max_retries=_get_llm_max_retries(),
    )
    prompt_tokens = _estimate_text_tokens(system_content) + _estimate_text_tokens(human_content)
    reasoning_effort = _map_thinking_mode_to_effort(thinking_mode)
    minimum_tier_index = 0
    current_max_output_tokens = (
        max(256, int(max_output_tokens))
        if max_output_tokens is not None
        else None
    )
    max_attempts = max(1, int(os.getenv("DDKIT_EXEC_REASONING_MAX_ATTEMPTS", "3") or 3))
    retry_token_increment = max(400, int(os.getenv("DDKIT_EXEC_REASONING_RETRY_TOKEN_INCREMENT", "1600") or 1600))
    retry_token_cap = max(
        _estimated_exec_completion_tokens(current_max_output_tokens),
        int(os.getenv("DDKIT_EXEC_REASONING_MAX_OUTPUT_TOKEN_CAP", "7200") or 7200),
    )
    reasoning_summary_mode = _reasoning_summary_mode()
    attempt = 0

    while True:
        attempt += 1
        estimated_total_tokens = prompt_tokens + _estimated_exec_completion_tokens(current_max_output_tokens)
        routed = reserve_routed_model(
            requested_model=requested_model,
            estimated_total_tokens=estimated_total_tokens,
            minimum_tier_index=minimum_tier_index,
            block_class=block_class,
            thinking_mode=thinking_mode,
            allow_nano_final=block_class != "critical",
        )
        params = {
            "model": routed.model,
            "instructions": system_content,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": human_content}]}],
            "metadata": metadata or {},
            "reasoning": {"effort": reasoning_effort, "summary": reasoning_summary_mode},
        }
        if current_max_output_tokens is not None:
            params["max_output_tokens"] = current_max_output_tokens
        try:
            if hasattr(client.responses, "parse"):
                response = client.responses.parse(
                    text_format=response_format,
                    **params,
                )
                parsed_output = _extract_response_parsed(response, response_format)
                if parsed_output is None:
                    incomplete_reason = _response_incomplete_reason(response)
                    usage = extract_usage_metrics(response)
                    budget_after = commit_routed_usage(routed, usage)
                    if (
                        incomplete_reason == "max_output_tokens"
                        and current_max_output_tokens is not None
                        and attempt < max_attempts
                        and current_max_output_tokens < retry_token_cap
                    ):
                        current_max_output_tokens = min(
                            retry_token_cap,
                            current_max_output_tokens + retry_token_increment,
                        )
                        minimum_tier_index = 0
                        continue
                    parsed_output = response_format.model_validate_json(repair_json(_extract_response_text(response)))
                    budget_trace = build_budget_trace(
                        routed,
                        usage_actual=usage,
                        budget_snapshot_after=budget_after,
                        reasoning_effort_actual=reasoning_effort,
                    )
                    return ExecReasoningCallResult(
                        parsed_output=parsed_output,
                        raw_response=response,
                        model_selected=routed.model,
                        budget_trace=budget_trace,
                        reasoning_summary=_extract_reasoning_summary(response),
                    )
            else:  # pragma: no cover
                response = client.responses.create(
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": response_format.__name__,
                            "schema": response_format.model_json_schema(),
                            "strict": True,
                        }
                    },
                    **params,
                )
                parsed_output = response_format.model_validate_json(repair_json(_extract_response_text(response)))
            usage = extract_usage_metrics(response)
            budget_after = commit_routed_usage(routed, usage)
            budget_trace = build_budget_trace(
                routed,
                usage_actual=usage,
                budget_snapshot_after=budget_after,
                reasoning_effort_actual=reasoning_effort,
            )
            return ExecReasoningCallResult(
                parsed_output=parsed_output,
                raw_response=response,
                model_selected=routed.model,
                budget_trace=budget_trace,
                reasoning_summary=_extract_reasoning_summary(response),
            )
        except Exception as exc:
            release_routed_reservation(routed)
            if is_quota_exhausted_error(exc) and routed.tier in {"elite", "mini"}:
                mark_tier_exhausted(routed, str(exc))
                minimum_tier_index = next_tier_index(routed.tier)
                continue
            if (
                _is_truncated_structured_output_error(exc)
                and current_max_output_tokens is not None
                and attempt < max_attempts
                and current_max_output_tokens < retry_token_cap
            ):
                current_max_output_tokens = min(
                    retry_token_cap,
                    current_max_output_tokens + retry_token_increment,
                )
                minimum_tier_index = 0
                continue
            raise


class BaseOpenaiProcessor:
    def __init__(self):
        self.llm = self.set_up_llm()
        self.default_model = os.getenv("DDKIT_DEFAULT_MODEL", "gpt-5.4")

    def set_up_llm(self):
        load_dotenv()
        llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=_get_llm_timeout_seconds(),
            max_retries=2
            )
        return llm

    def send_message(
        self,
        model=None,
        temperature=0.5,
        seed=None,  # For deterministic outputs
        system_content='You are a helpful assistant.',
        human_content='Hello!',
        is_structured=False,
        response_format=None,
        max_completion_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,  # "none"|"low"|"medium"|"high"
        ):
        if model is None:
            model = self.default_model

        # Resolve max_completion_tokens from env if not passed
        if max_completion_tokens is None:
            _env_mct = os.getenv("DDKIT_MAX_COMPLETION_TOKENS", "").strip()
            if _env_mct:
                try:
                    max_completion_tokens = int(_env_mct)
                except ValueError:
                    pass

        minimum_tier_index = 0
        while True:
            routed = choose_routed_model(model, minimum_tier_index=minimum_tier_index)
            active_model = routed.model

            # Resolve reasoning_effort: env override, then auto-set "none" for gpt-5.x
            active_reasoning_effort = reasoning_effort
            if active_reasoning_effort is None:
                active_reasoning_effort = os.getenv("DDKIT_REASONING_EFFORT") or None
            if active_reasoning_effort is None and active_model.startswith("gpt-5"):
                active_reasoning_effort = "none"

            params = {
                "model": active_model,
                "seed": seed,
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": human_content}
                ]
            }

            # temperature: skip for reasoning models and gpt-5.x with reasoning_effort != "none"
            _is_reasoning = active_model.startswith("o1") or active_model.startswith("o3") or active_model.startswith("o4")
            _gpt5_thinking = active_model.startswith("gpt-5") and active_reasoning_effort and active_reasoning_effort != "none"
            if not _is_reasoning and not _gpt5_thinking:
                params["temperature"] = temperature

            # reasoning_effort only applies to unstructured calls (parse() doesn't accept it)
            if active_reasoning_effort is not None and not is_structured:
                params["reasoning_effort"] = active_reasoning_effort

            if max_completion_tokens is not None:
                params["max_completion_tokens"] = max_completion_tokens

            try:
                if not is_structured:
                    completion = self.llm.chat.completions.create(**params)
                    content = completion.choices[0].message.content
                else:
                    params["response_format"] = response_format
                    completion = self.llm.beta.chat.completions.parse(**params)
                    response = completion.choices[0].message.parsed
                    content = response.dict()

                usage = extract_usage_metrics(completion)
                record_usage(routed, usage)
                self.response_data = {
                    "requested_model": model,
                    "model": completion.model,
                    "router_tier": routed.tier,
                    "input_tokens": usage.get("input_tokens"),
                    "output_tokens": usage.get("output_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "cached_tokens": usage.get("cached_tokens", 0),
                    "reasoning_tokens": usage.get("reasoning_tokens", 0),
                }
                print(self.response_data)
                return content
            except Exception as exc:
                if is_quota_exhausted_error(exc) and routed.tier in {"elite", "mini"}:
                    mark_tier_exhausted(routed, str(exc))
                    minimum_tier_index = next_tier_index(routed.tier)
                    continue
                raise

    @staticmethod
    def count_tokens(string, encoding_name="o200k_base"):
        encoding = tiktoken.get_encoding(encoding_name)

        # Encode the string and count the tokens
        tokens = encoding.encode(string)
        token_count = len(tokens)

        return token_count


class BaseIBMAPIProcessor:
    def __init__(self):
        load_dotenv()
        self.api_token = os.getenv("IBM_API_KEY")
        self.base_url = "https://rag.timetoact.at/ibm"
        self.default_model = 'meta-llama/llama-3-3-70b-instruct'
    def check_balance(self):
        """Check the current balance for the provided token."""
        balance_url = f"{self.base_url}/balance"
        headers = {"Authorization": f"Bearer {self.api_token}"}
        
        try:
            response = requests.get(balance_url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as err:
            print(f"Error checking balance: {err}")
            return None
    
    def get_available_models(self):
        """Get a list of available foundation models."""
        models_url = f"{self.base_url}/foundation_model_specs"
        
        try:
            response = requests.get(models_url)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as err:
            print(f"Error getting available models: {err}")
            return None
    
    def get_embeddings(self, texts, model_id="ibm/granite-embedding-278m-multilingual"):
        """Get vector embeddings for the provided text inputs."""
        embeddings_url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": texts,
            "model_id": model_id
        }
        
        try:
            response = requests.post(embeddings_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as err:
            print(f"Error getting embeddings: {err}")
            return None
    
    def send_message(
        self,
        # model='meta-llama/llama-3-1-8b-instruct',
        model=None,
        temperature=0.5,
        seed=None,  # For deterministic outputs
        system_content='You are a helpful assistant.',
        human_content='Hello!',
        is_structured=False,
        response_format=None,
        max_new_tokens=5000,
        min_new_tokens=1,
        **kwargs
    ):
        if model is None:
            model = self.default_model
        text_generation_url = f"{self.base_url}/text_generation"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        # Prepare the input messages
        input_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": human_content}
        ]
        
        # Prepare parameters with defaults and any additional parameters
        parameters = {
            "temperature": temperature,
            "random_seed": seed,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            **kwargs
        }
        
        payload = {
            "input": input_messages,
            "model_id": model,
            "parameters": parameters
        }
        
        try:
            response = requests.post(text_generation_url, headers=headers, json=payload)
            response.raise_for_status()
            completion = response.json()

            content = completion.get("results")[0].get("generated_text")
            self.response_data = {"model": completion.get("model_id"), "input_tokens": completion.get("results")[0].get("input_token_count"), "output_tokens": completion.get("results")[0].get("generated_token_count")}
            print(self.response_data)
            if is_structured and response_format is not None:
                try:
                    repaired_json = repair_json(content)
                    parsed_dict = json.loads(repaired_json)
                    validated_data = response_format.model_validate(parsed_dict)
                    content = validated_data.model_dump()
                    return content
                
                except Exception as err:
                    print("Error processing structured response, attempting to reparse the response...")
                    reparsed = self._reparse_response(content, system_content)
                    try:
                        repaired_json = repair_json(reparsed)
                        reparsed_dict = json.loads(repaired_json)
                        try:
                            validated_data = response_format.model_validate(reparsed_dict)
                            print("Reparsing successful!")
                            content = validated_data.model_dump()
                            return content
                        
                        except Exception:
                            return reparsed_dict
                        
                    except Exception as reparse_err:
                        print(f"Reparse failed with error: {reparse_err}")
                        print(f"Reparsed response: {reparsed}")
                        return content
            
            return content

        except requests.HTTPError as err:
            print(f"Error generating text: {err}")
            return None

    def _reparse_response(self, response, system_content):

        user_prompt = prompts.AnswerSchemaFixPrompt.user_prompt.format(
            system_prompt=system_content,
            response=response
        )
        
        reparsed_response = self.send_message(
            system_content=prompts.AnswerSchemaFixPrompt.system_prompt,
            human_content=user_prompt,
            is_structured=False
        )
        
        return reparsed_response

     
class BaseGeminiProcessor:
    def __init__(self):
        self.llm = self._set_up_llm()
        self.default_model = 'gemini-2.0-flash-001'
        # self.default_model = "gemini-2.0-flash-thinking-exp-01-21",
        
    def _set_up_llm(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        return genai

    def list_available_models(self) -> None:
        """
        Prints available Gemini models that support text generation.
        """
        print("Available models for text generation:")
        for model in self.llm.list_models():
            if "generateContent" in model.supported_generation_methods:
                print(f"- {model.name}")
                print(f"  Input token limit: {model.input_token_limit}")
                print(f"  Output token limit: {model.output_token_limit}")
                print()

    def _log_retry_attempt(retry_state):
        """Print information about the retry attempt"""
        exception = retry_state.outcome.exception()
        print(f"\nAPI Error encountered: {str(exception)}")
        print("Waiting 20 seconds before retry...\n")

    @retry(
        wait=wait_fixed(20),
        stop=stop_after_attempt(3),
        before_sleep=_log_retry_attempt,
    )
    def _generate_with_retry(self, model, human_content, generation_config):
        """Wrapper for generate_content with retry logic"""
        try:
            return model.generate_content(
                human_content,
                generation_config=generation_config
            )
        except Exception as e:
            if getattr(e, '_attempt_number', 0) == 3:
                print(f"\nRetry failed. Error: {str(e)}\n")
            raise

    def _parse_structured_response(self, response_text, response_format):
        try:
            repaired_json = repair_json(response_text)
            parsed_dict = json.loads(repaired_json)
            validated_data = response_format.model_validate(parsed_dict)
            return validated_data.model_dump()
        except Exception as err:
            print(f"Error parsing structured response: {err}")
            print("Attempting to reparse the response...")
            reparsed = self._reparse_response(response_text, response_format)
            return reparsed

    def _reparse_response(self, response, response_format):
        """Reparse invalid JSON responses using the model itself."""
        user_prompt = prompts.AnswerSchemaFixPrompt.user_prompt.format(
            system_prompt=prompts.AnswerSchemaFixPrompt.system_prompt,
            response=response
        )
        
        try:
            reparsed_response = self.send_message(
                model="gemini-2.0-flash-001",
                system_content=prompts.AnswerSchemaFixPrompt.system_prompt,
                human_content=user_prompt,
                is_structured=False
            )
            
            try:
                repaired_json = repair_json(reparsed_response)
                reparsed_dict = json.loads(repaired_json)
                try:
                    validated_data = response_format.model_validate(reparsed_dict)
                    print("Reparsing successful!")
                    return validated_data.model_dump()
                except Exception:
                    return reparsed_dict
            except Exception as reparse_err:
                print(f"Reparse failed with error: {reparse_err}")
                print(f"Reparsed response: {reparsed_response}")
                return response
        except Exception as e:
            print(f"Reparse attempt failed: {e}")
            return response

    def send_message(
        self,
        model=None,
        temperature: float = 0.5,
        seed=12345,  # For back compatibility
        system_content: str = "You are a helpful assistant.",
        human_content: str = "Hello!",
        is_structured: bool = False,
        response_format: Optional[Type[BaseModel]] = None,
    ) -> Union[str, Dict, None]:
        if model is None:
            model = self.default_model

        generation_config = {"temperature": temperature}
        
        prompt = f"{system_content}\n\n---\n\n{human_content}"

        model_instance = self.llm.GenerativeModel(
            model_name=model,
            generation_config=generation_config
        )

        try:
            response = self._generate_with_retry(model_instance, prompt, generation_config)

            self.response_data = {
                "model": response.model_version,
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count
            }
            print(self.response_data)
            
            if is_structured and response_format is not None:
                return self._parse_structured_response(response.text, response_format)
            
            return response.text
        except Exception as e:
            raise Exception(f"API request failed after retries: {str(e)}")


class APIProcessor:
    def __init__(self, provider: Literal["openai", "ibm", "gemini"] ="openai"):
        self.provider = provider.lower()
        if self.provider == "openai":
            self.processor = BaseOpenaiProcessor()
        elif self.provider == "ibm":
            self.processor = BaseIBMAPIProcessor()
        elif self.provider == "gemini":
            self.processor = BaseGeminiProcessor()

    def send_message(
        self,
        model=None,
        temperature=0.5,
        seed=None,
        system_content="You are a helpful assistant.",
        human_content="Hello!",
        is_structured=False,
        response_format=None,
        **kwargs
    ):
        """
        Routes the send_message call to the appropriate processor.
        The underlying processor's send_message method is responsible for handling the parameters.
        """
        if model is None:
            model = self.processor.default_model
        return self.processor.send_message(
            model=model,
            temperature=temperature,
            seed=seed,
            system_content=system_content,
            human_content=human_content,
            is_structured=is_structured,
            response_format=response_format,
            **kwargs
        )

    def get_answer_from_rag_context(self, question, rag_context, schema, model):
        system_prompt, response_format, user_prompt = self._build_rag_context_prompts(schema)
        
        answer_dict = self.processor.send_message(
            model=model,
            system_content=system_prompt,
            human_content=user_prompt.format(context=rag_context, question=question),
            is_structured=True,
            response_format=response_format
        )
        self.response_data = self.processor.response_data
        return answer_dict


    def _build_rag_context_prompts(self, schema):
        """Return prompts tuple for the given schema."""
        use_schema_prompt = True if self.provider == "ibm" or self.provider == "gemini" else False
        
        if schema == "name":
            system_prompt = (prompts.AnswerWithRAGContextNamePrompt.system_prompt_with_schema 
                            if use_schema_prompt else prompts.AnswerWithRAGContextNamePrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextNamePrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextNamePrompt.user_prompt
        elif schema == "number":
            system_prompt = (prompts.AnswerWithRAGContextNumberPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.AnswerWithRAGContextNumberPrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextNumberPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextNumberPrompt.user_prompt
        elif schema == "boolean":
            system_prompt = (prompts.AnswerWithRAGContextBooleanPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.AnswerWithRAGContextBooleanPrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextBooleanPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextBooleanPrompt.user_prompt
        elif schema == "names":
            system_prompt = (prompts.AnswerWithRAGContextNamesPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.AnswerWithRAGContextNamesPrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextNamesPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextNamesPrompt.user_prompt
        elif schema == "comparative":
            system_prompt = (prompts.ComparativeAnswerPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.ComparativeAnswerPrompt.system_prompt)
            response_format = prompts.ComparativeAnswerPrompt.AnswerSchema
            user_prompt = prompts.ComparativeAnswerPrompt.user_prompt
        else:
            raise ValueError(f"Unsupported schema: {schema}")
        return system_prompt, response_format, user_prompt

    def get_rephrased_questions(self, original_question: str, companies: List[str]) -> Dict[str, str]:
        """Use LLM to break down a comparative question into individual questions."""
        answer_dict = self.processor.send_message(
            system_content=prompts.RephrasedQuestionsPrompt.system_prompt,
            human_content=prompts.RephrasedQuestionsPrompt.user_prompt.format(
                question=original_question,
                companies=", ".join([f'"{company}"' for company in companies])
            ),
            is_structured=True,
            response_format=prompts.RephrasedQuestionsPrompt.RephrasedQuestions
        )
        
        # Convert the answer_dict to the desired format
        questions_dict = {item["company_name"]: item["question"] for item in answer_dict["questions"]}
        
        return questions_dict


class AsyncOpenaiProcessor:
    
    def _get_unique_filepath(self, base_filepath):
        """Helper method to get unique filepath"""
        if not os.path.exists(base_filepath):
            return base_filepath
        
        base, ext = os.path.splitext(base_filepath)
        counter = 1
        while os.path.exists(f"{base}_{counter}{ext}"):
            counter += 1
        return f"{base}_{counter}{ext}"

    async def process_structured_ouputs_requests(
        self,
        model=None,
        temperature=0.5,
        seed=None,
        system_content="You are a helpful assistant.",
        queries=None,
        response_format=None,
        requests_filepath='./temp_async_llm_requests.jsonl',
        save_filepath='./temp_async_llm_results.jsonl',
        preserve_requests=False,
        preserve_results=True,
        request_url="https://api.openai.com/v1/chat/completions",
        max_requests_per_minute=3_500,
        max_tokens_per_minute=3_500_000,
        token_encoding_name="o200k_base",
        max_attempts=5,
        logging_level=20,
        progress_callback=None,
        max_completion_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
    ):
        if model is None:
            model = os.getenv("DDKIT_DEFAULT_MODEL", "gpt-5.4")

        routed = choose_routed_model(model)
        model = routed.model

        # Resolve reasoning_effort
        if reasoning_effort is None:
            reasoning_effort = os.getenv("DDKIT_REASONING_EFFORT") or None
        if reasoning_effort is None and model.startswith("gpt-5"):
            reasoning_effort = "none"

        # Resolve max_completion_tokens
        if max_completion_tokens is None:
            _env_mct = os.getenv("DDKIT_MAX_COMPLETION_TOKENS", "").strip()
            if _env_mct:
                try:
                    max_completion_tokens = int(_env_mct)
                except ValueError:
                    pass

        # Create requests for jsonl
        jsonl_requests = []
        _is_reasoning = model.startswith("o1") or model.startswith("o3") or model.startswith("o4")
        _gpt5_thinking = model.startswith("gpt-5") and reasoning_effort and reasoning_effort != "none"
        for idx, query in enumerate(queries):
            request = {
                "model": model,
                "seed": seed,
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": query},
                ],
                'response_format': type_to_response_format_param(response_format),
                'metadata': {'original_index': idx}
            }
            if not _is_reasoning and not _gpt5_thinking:
                request["temperature"] = temperature
            if reasoning_effort is not None:
                request["reasoning_effort"] = reasoning_effort
            if max_completion_tokens is not None:
                request["max_completion_tokens"] = max_completion_tokens
            jsonl_requests.append(request)
            
        # Get unique filepaths if files already exist
        requests_filepath = self._get_unique_filepath(requests_filepath)
        save_filepath = self._get_unique_filepath(save_filepath)

        # Write requests to JSONL file
        with open(requests_filepath, "w") as f:
            for request in jsonl_requests:
                json_string = json.dumps(request)
                f.write(json_string + "\n")

        # Process API requests
        total_requests = len(jsonl_requests)

        async def monitor_progress():
            last_count = 0
            while True:
                try:
                    with open(save_filepath, 'r') as f:
                        current_count = sum(1 for _ in f)
                        if current_count > last_count:
                            if progress_callback:
                                for _ in range(current_count - last_count):
                                    progress_callback()
                            last_count = current_count
                        if current_count >= total_requests:
                            break
                except FileNotFoundError:
                    pass
                await asyncio.sleep(0.1)

        async def process_with_progress():
            await asyncio.gather(
                process_api_requests_from_file(
                    requests_filepath=requests_filepath,
                    save_filepath=save_filepath,
                    request_url=request_url,
                    api_key=os.getenv("OPENAI_API_KEY"),
                    max_requests_per_minute=max_requests_per_minute,
                    max_tokens_per_minute=max_tokens_per_minute,
                    token_encoding_name=token_encoding_name,
                    max_attempts=max_attempts,
                    logging_level=logging_level
                ),
                monitor_progress()
            )

        await process_with_progress()

        with open(save_filepath, "r") as f:
            validated_data_list = []
            results = []
            for line_number, line in enumerate(f, start=1):
                raw_line = line.strip()
                try:
                    result = json.loads(raw_line)
                except json.JSONDecodeError as e:
                    print(f"[ERROR] Line {line_number}: Failed to load JSON from line: {raw_line}")
                    continue

                # Check finish_reason in the API response
                finish_reason = result[1]['choices'][0].get('finish_reason', '')
                if finish_reason != "stop":
                    print(f"[WARNING] Line {line_number}: finish_reason is '{finish_reason}' (expected 'stop').")

                usage = extract_usage_metrics(result[1])
                record_usage(routed, usage)

                # Safely parse answer; if it fails, leave answer empty and report the error.
                try:
                    answer_content = result[1]['choices'][0]['message']['content']
                    answer_parsed = json.loads(answer_content)
                    answer = response_format(**answer_parsed).model_dump()
                except Exception as e:
                    print(f"[ERROR] Line {line_number}: Failed to parse answer JSON. Error: {e}.")
                    answer = ""

                results.append({
                    'index': result[2],
                    'question': result[0]['messages'],
                    'answer': answer,
                    'usage': usage,
                })
            
            # Sort by original index and build final list
            validated_data_list = [
                {'question': r['question'], 'answer': r['answer'], 'usage': r['usage']} 
                for r in sorted(results, key=lambda x: x['index']['original_index'])
            ]

        if not preserve_requests:
            os.remove(requests_filepath)

        if not preserve_results:
            os.remove(save_filepath)
        else:  # Fix requests order
            with open(save_filepath, "r") as f:
                results = [json.loads(line) for line in f]
            
            sorted_results = sorted(results, key=lambda x: x[2]['original_index'])
            
            with open(save_filepath, "w") as f:
                for result in sorted_results:
                    json_string = json.dumps(result)
                    f.write(json_string + "\n")
            
        return validated_data_list
