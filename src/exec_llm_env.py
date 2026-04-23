"""
Strict OpenAI key loading for the exec decision engine.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

from dotenv import load_dotenv


_DEFAULT_EXEC_ENV_PATH = Path(r"C:\GItHub\pharm_search\.env")


def _candidate_env_paths() -> Iterable[Path]:
    explicit_path = (os.getenv("DDKIT_EXEC_OPENAI_ENV_FILE") or "").strip()
    if explicit_path:
        yield Path(explicit_path)
        return
    yield _DEFAULT_EXEC_ENV_PATH


def load_exec_openai_env() -> Optional[Path]:
    if (os.getenv("OPENAI_API_KEY") or "").strip():
        return None
    for path in _candidate_env_paths():
        if not path.exists():
            continue
        load_dotenv(dotenv_path=path, override=False)
        if (os.getenv("OPENAI_API_KEY") or "").strip():
            return path
    return None


def require_exec_openai_api_key() -> str:
    loaded_from = load_exec_openai_env()
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if api_key:
        return api_key
    expected_paths = ", ".join(str(path) for path in _candidate_env_paths())
    source_hint = f" Last attempted env file: {loaded_from}." if loaded_from else ""
    raise RuntimeError(
        "Exec reasoning requires OPENAI_API_KEY and heuristic mode is disabled. "
        f"Expected key from env or one of: {expected_paths}.{source_hint}"
    )
