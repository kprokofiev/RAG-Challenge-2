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
import struct
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
import os

from src.settings import settings

load_dotenv()

try:
    import redis as redis_lib
except ImportError:
    redis_lib = None  # type: ignore

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


# ── Embedding cache (Redis-backed) ──────────────────────────────────────────
# Key format: emb:{pipeline_version}:{model_short}:{sha256_24}
# Value: raw float32 bytes (dim * 4 bytes)
# TTL: 30 days (re-run within a month → full cache hit)

_CACHE_TTL = 86400 * 30  # 30 days


def _cache_key(text: str, model: str) -> str:
    """Build a Redis key for an embedding cache entry."""
    h = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:24]
    model_short = model.replace("text-embedding-3-", "te3-")
    return f"emb:{EMBEDDING_PIPELINE_VERSION}:{model_short}:{h}"


class EmbeddingCache:
    """Redis-backed embedding cache for doc/chunk ingestion.

    Stores vector bytes keyed by content hash + model + pipeline version.
    Cache is invalidated automatically when pipeline version bumps.
    """

    def __init__(self, redis_client):
        self._r = redis_client
        self._hits = 0
        self._misses = 0

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "cache_hits": self._hits,
            "cache_misses": self._misses,
            "cache_hit_rate": round(self._hits / total, 3) if total else 0.0,
        }

    def reset_stats(self):
        self._hits = 0
        self._misses = 0

    def get(self, text: str, model: str) -> Optional[np.ndarray]:
        """Retrieve a cached embedding vector, or None on miss."""
        try:
            raw = self._r.get(_cache_key(text, model))
        except Exception:
            return None
        if raw is None:
            self._misses += 1
            return None
        self._hits += 1
        return np.frombuffer(raw, dtype=np.float32).copy()

    def put(self, text: str, model: str, vec: List[float]):
        """Store an embedding vector in cache."""
        try:
            arr = np.array(vec, dtype=np.float32)
            self._r.setex(_cache_key(text, model), _CACHE_TTL, arr.tobytes())
        except Exception as e:
            logger.debug("embedding_cache put error: %s", e)

    def get_batch(self, texts: List[str], model: str) -> Tuple[List[Optional[List[float]]], List[int]]:
        """Batch lookup. Returns (results, miss_indices).

        results[i] is the cached vector (as list) or None.
        miss_indices lists positions that need fresh embedding.
        """
        keys = [_cache_key(t, model) for t in texts]
        try:
            raw_values = self._r.mget(keys)
        except Exception:
            self._misses += len(texts)
            return [None] * len(texts), list(range(len(texts)))

        results: List[Optional[List[float]]] = []
        miss_indices: List[int] = []
        for i, raw in enumerate(raw_values):
            if raw is not None:
                self._hits += 1
                results.append(np.frombuffer(raw, dtype=np.float32).tolist())
            else:
                self._misses += 1
                results.append(None)
                miss_indices.append(i)
        return results, miss_indices

    def put_batch(self, texts: List[str], model: str, vecs: List[List[float]]):
        """Store multiple embeddings in a Redis pipeline (atomic batch)."""
        try:
            pipe = self._r.pipeline(transaction=False)
            for text, vec in zip(texts, vecs):
                arr = np.array(vec, dtype=np.float32)
                pipe.setex(_cache_key(text, model), _CACHE_TTL, arr.tobytes())
            pipe.execute()
        except Exception as e:
            logger.debug("embedding_cache put_batch error: %s", e)


def get_embedding_cache() -> Optional[EmbeddingCache]:
    """Create an EmbeddingCache backed by the worker's Redis, or None."""
    if redis_lib is None:
        return None
    url = settings.redis_url
    if not url:
        return None
    try:
        client = redis_lib.from_url(url, socket_connect_timeout=3)
        client.ping()
        return EmbeddingCache(client)
    except Exception as e:
        logger.warning("embedding_cache unavailable: %s", e)
        return None
