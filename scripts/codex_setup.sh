#!/usr/bin/env bash
set -euo pipefail

if command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN=python3.11
elif command -v python3.12 >/dev/null 2>&1; then
  PYTHON_BIN=python3.12
else
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
