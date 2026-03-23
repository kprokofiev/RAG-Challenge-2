"""
Centralized embedding interface.

All embedding calls (indexing and retrieval) go through this module.
Provides:
  - Consistent model selection via settings.embeddings_model
  - L2-normalization for FAISS IndexFlatIP compatibility
  - Embedding manifest generation for index versioning
  - Future hook point for caching layer

Pipeline version is bumped when chunking/cleaning/normalization changes
in a way that makes old embeddings incompatible.
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os

from src.settings import settings

load_dotenv()

logger = logging.getLogger(__name__)

# Bump this when chunking, cleaning, or normalization logic changes
# in a way that invalidates cached embeddings / stored FAISS indices.
EMBEDDING_PIPELINE_VERSION = "1"

# Dimension expectations per model (for validation)
_MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def expected_dimension(model: Optional[str] = None) -> int:
    """Return expected vector dimension for the given model."""
    m = model or settings.embeddings_model
    return _MODEL_DIMENSIONS.get(m, 0)


def current_manifest() -> dict:
    """Return a manifest dict describing the current embedding config.

    Written alongside .faiss files so that merge/load can validate
    compatibility before mixing indices from different models.
    """
    model = settings.embeddings_model
    return {
        "embedding_model": model,
        "embedding_dim": expected_dimension(model),
        "pipeline_version": EMBEDDING_PIPELINE_VERSION,
    }


def validate_manifest_compat(stored: dict, label: str = "") -> bool:
    """Check whether a stored manifest is compatible with current config.

    Returns True if compatible, False otherwise (logs a warning).
    """
    cur = current_manifest()
    ok = True
    prefix = f"[{label}] " if label else ""

    if stored.get("embedding_model") != cur["embedding_model"]:
        logger.warning(
            "%sindex model mismatch: stored=%s current=%s",
            prefix,
            stored.get("embedding_model"),
            cur["embedding_model"],
        )
        ok = False

    stored_dim = stored.get("embedding_dim", 0)
    if stored_dim and stored_dim != cur["embedding_dim"]:
        logger.warning(
            "%sindex dim mismatch: stored=%d current=%d",
            prefix, stored_dim, cur["embedding_dim"],
        )
        ok = False

    return ok


def write_manifest(directory: Path, filename: str = "_embedding_manifest.json"):
    """Write current embedding manifest to a directory."""
    manifest = current_manifest()
    path = directory / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path


def read_manifest(directory: Path, filename: str = "_embedding_manifest.json") -> Optional[dict]:
    """Read embedding manifest from a directory, if it exists."""
    path = directory / filename
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
