#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=""

if command -v pyenv >/dev/null 2>&1; then
  for minor in 3.11 3.12; do
    PYENV_MATCH="$(pyenv versions --bare | grep -E "^${minor}\." | sort -V | tail -n 1 || true)"
    if [ -n "${PYENV_MATCH}" ]; then
      export PYENV_VERSION="${PYENV_MATCH}"
      PYTHON_BIN=python
      break
    fi
  done
fi

if [ -z "${PYTHON_BIN}" ]; then
  for candidate in python3.11 python3.12 python; do
    if command -v "${candidate}" >/dev/null 2>&1 && "${candidate}" -c 'import sys; raise SystemExit(not ((3, 11) <= sys.version_info[:2] < (3, 13)))' >/dev/null 2>&1; then
      PYTHON_BIN="${candidate}"
      break
    fi
  done
fi

if [ -z "${PYTHON_BIN}" ]; then
  PYTHON_BIN=python
fi

"${PYTHON_BIN}" - <<'PY'
import sys

version = sys.version_info
if version < (3, 11) or version >= (3, 13):
    raise SystemExit(
        "RAG-Challenge-2 requires Python 3.11 or 3.12 for pinned binary "
        "dependencies such as faiss-cpu==1.9.0.post1. In Codex Cloud, open "
        "Set package versions and pin Python to 3.11."
    )
print(f"Using Python {sys.version.split()[0]}")
PY

"${PYTHON_BIN}" -m pip install -U pip setuptools wheel
"${PYTHON_BIN}" -m pip install -r requirements.txt
"${PYTHON_BIN}" -m pip install -e .
