from types import SimpleNamespace
from unittest import TestCase, mock

from pydantic import BaseModel

from src.api_requests import call_exec_reasoning_model


class _DemoSchema(BaseModel):
    foo: str


class _FakeUsage:
    def __init__(self, input_tokens: int, output_tokens: int, total_tokens: int, reasoning_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = total_tokens
        self.input_tokens_details = SimpleNamespace(cached_tokens=0)
        self.output_tokens_details = SimpleNamespace(reasoning_tokens=reasoning_tokens)


class _FakeResponse:
    def __init__(
        self,
        *,
        status: str,
        output_parsed=None,
        output_text: str = "",
        incomplete_reason: str | None = None,
        usage: _FakeUsage | None = None,
    ):
        self.status = status
        self.output_parsed = output_parsed
        self.output_text = output_text
        self.output = []
        self.reasoning = None
        self.usage = usage
        self.incomplete_details = (
            SimpleNamespace(reason=incomplete_reason) if incomplete_reason else None
        )


class _FakeResponsesClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.parse_calls = []
        self.create_calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        self.parse_calls.append(kwargs)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def create(self, **kwargs):
        self.calls.append(kwargs)
        self.create_calls.append(kwargs)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class _FakeOpenAIClient:
    def __init__(self, responses):
        self.responses = _FakeResponsesClient(responses)


class ExecApiRequestsTests(TestCase):
    def test_call_exec_reasoning_model_retries_on_max_output_tokens_incomplete(self):
        completed = _FakeResponse(
            status="completed",
            output_parsed=_DemoSchema(foo="ok"),
            output_text='{"foo":"ok"}',
            usage=_FakeUsage(input_tokens=10, output_tokens=20, total_tokens=30, reasoning_tokens=12),
        )
        incomplete = _FakeResponse(
            status="incomplete",
            output_parsed=None,
            output_text="",
            incomplete_reason="max_output_tokens",
            usage=_FakeUsage(input_tokens=10, output_tokens=2400, total_tokens=2410, reasoning_tokens=2400),
        )
        fake_client = _FakeOpenAIClient([incomplete, completed])
        routed = SimpleNamespace(
            model="gpt-5.4-mini",
            tier="mini",
            requested_model="gpt-5.4-mini",
            day_key="2026-04-14",
            redis_key="router-key",
            fallback_reason=None,
            budget_snapshot_before={},
            reset_at_utc="2026-04-15T00:00:00Z",
            reservation_id=None,
            reserved_tokens=0,
            block_class="critical",
            thinking_mode_requested="high",
        )

        with mock.patch("src.api_requests.require_exec_openai_api_key", return_value="test-key"), mock.patch(
            "src.api_requests.OpenAI", return_value=fake_client
        ) as openai_mock, mock.patch.dict(
            "os.environ", {"DDKIT_LLM_MAX_RETRIES": "0"}, clear=False
        ), mock.patch(
            "src.api_requests.reserve_routed_model", return_value=routed
        ), mock.patch(
            "src.api_requests.commit_routed_usage", return_value={}
        ), mock.patch(
            "src.api_requests.build_budget_trace",
            return_value={"model_selected": "gpt-5.4-mini", "thinking_mode_requested": "high"},
        ):
            result = call_exec_reasoning_model(
                system_content="Return the object only.",
                human_content="Return {'foo':'ok'}",
                response_format=_DemoSchema,
                requested_model="gpt-5.4-mini",
                thinking_mode="high",
                max_output_tokens=2400,
                metadata={"phase": "planner"},
                block_class="critical",
            )

        self.assertEqual(result.parsed_output.foo, "ok")
        self.assertEqual(openai_mock.call_args.kwargs["max_retries"], 0)
        self.assertEqual(len(fake_client.responses.calls), 2)
        self.assertEqual(fake_client.responses.calls[0]["max_output_tokens"], 2400)
        self.assertGreater(fake_client.responses.calls[1]["max_output_tokens"], 2400)

    def test_call_exec_reasoning_model_retries_on_truncated_structured_parse_error(self):
        completed = _FakeResponse(
            status="completed",
            output_parsed=_DemoSchema(foo="ok"),
            output_text='{"foo":"ok"}',
            usage=_FakeUsage(input_tokens=10, output_tokens=20, total_tokens=30, reasoning_tokens=12),
        )
        fake_client = _FakeOpenAIClient([ValueError("Invalid JSON: EOF while parsing object"), completed])
        routed = SimpleNamespace(
            model="gpt-5.4-mini",
            tier="mini",
            requested_model="gpt-5.4-mini",
            day_key="2026-04-14",
            redis_key="router-key",
            fallback_reason=None,
            budget_snapshot_before={},
            reset_at_utc="2026-04-15T00:00:00Z",
            reservation_id=None,
            reserved_tokens=0,
            block_class="critical",
            thinking_mode_requested="high",
        )

        with mock.patch("src.api_requests.require_exec_openai_api_key", return_value="test-key"), mock.patch(
            "src.api_requests.OpenAI", return_value=fake_client
        ) as openai_mock, mock.patch(
            "src.api_requests.reserve_routed_model", return_value=routed
        ), mock.patch(
            "src.api_requests.release_routed_reservation"
        ) as release_mock, mock.patch(
            "src.api_requests.commit_routed_usage", return_value={}
        ), mock.patch(
            "src.api_requests.build_budget_trace",
            return_value={"model_selected": "gpt-5.4-mini", "thinking_mode_requested": "high"},
        ):
            result = call_exec_reasoning_model(
                system_content="Return the object only.",
                human_content="Return {'foo':'ok'}",
                response_format=_DemoSchema,
                requested_model="gpt-5.4-mini",
                thinking_mode="high",
                max_output_tokens=2400,
                metadata={"phase": "answerer"},
                block_class="critical",
            )

        self.assertEqual(result.parsed_output.foo, "ok")
        self.assertEqual(openai_mock.call_args.kwargs["max_retries"], 0)
        self.assertEqual(len(fake_client.responses.calls), 2)
        self.assertEqual(fake_client.responses.calls[0]["max_output_tokens"], 2400)
        self.assertGreater(fake_client.responses.calls[1]["max_output_tokens"], 2400)
        release_mock.assert_called_once_with(routed)

    def test_call_exec_reasoning_model_falls_back_to_create_on_empty_parsed_output(self):
        empty_parsed = _FakeResponse(
            status="completed",
            output_parsed="",
            output_text="",
            usage=_FakeUsage(input_tokens=10, output_tokens=10, total_tokens=20, reasoning_tokens=8),
        )
        completed = _FakeResponse(
            status="completed",
            output_parsed=_DemoSchema(foo="ok"),
            output_text='{"foo":"ok"}',
            usage=_FakeUsage(input_tokens=10, output_tokens=20, total_tokens=30, reasoning_tokens=12),
        )
        fake_client = _FakeOpenAIClient([empty_parsed, completed])
        routed = SimpleNamespace(
            model="gpt-5.4-mini",
            tier="mini",
            requested_model="gpt-5.4-mini",
            day_key="2026-04-14",
            redis_key="router-key",
            fallback_reason=None,
            budget_snapshot_before={},
            reset_at_utc="2026-04-15T00:00:00Z",
            reservation_id=None,
            reserved_tokens=0,
            block_class="critical",
            thinking_mode_requested="high",
        )

        with mock.patch("src.api_requests.require_exec_openai_api_key", return_value="test-key"), mock.patch(
            "src.api_requests.OpenAI", return_value=fake_client
        ) as openai_mock, mock.patch(
            "src.api_requests.reserve_routed_model", return_value=routed
        ), mock.patch(
            "src.api_requests.release_routed_reservation"
        ) as release_mock, mock.patch(
            "src.api_requests.commit_routed_usage", return_value={}
        ), mock.patch(
            "src.api_requests.build_budget_trace",
            return_value={"model_selected": "gpt-5.4-mini", "thinking_mode_requested": "high"},
        ):
            result = call_exec_reasoning_model(
                system_content="Return the object only.",
                human_content="Return {'foo':'ok'}",
                response_format=_DemoSchema,
                requested_model="gpt-5.4-mini",
                thinking_mode="high",
                max_output_tokens=2400,
                metadata={"phase": "planner"},
                block_class="critical",
            )

        self.assertEqual(result.parsed_output.foo, "ok")
        self.assertEqual(openai_mock.call_args.kwargs["max_retries"], 0)
        self.assertEqual(len(fake_client.responses.calls), 2)
        self.assertEqual(len(fake_client.responses.parse_calls), 1)
        self.assertEqual(len(fake_client.responses.create_calls), 1)
        self.assertEqual(fake_client.responses.parse_calls[0]["max_output_tokens"], 2400)
        self.assertEqual(fake_client.responses.create_calls[0]["max_output_tokens"], 2400)
        self.assertEqual(fake_client.responses.create_calls[0]["text"]["format"]["type"], "json_schema")
        release_mock.assert_not_called()

