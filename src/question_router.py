"""
Question Router — Sprint 22 WS2-D2
====================================
Config-based router that resolves a question_id from exec_question_library.yaml
into a RoutedQuestion with all metadata needed by downstream pipeline.

NOT an LLM classifier. Rules-first, config-driven.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_QUESTION_LIBRARY_PATH = _CONFIG_DIR / "exec_question_library.yaml"


class RoutedQuestion(BaseModel):
    """Output of the question router — everything downstream needs."""
    question_id: str
    question_text: str
    question_type: str
    business_lens: str
    required_jurisdictions: List[str] = Field(default_factory=list)
    required_confidence: str = "medium"
    required_sections: List[str] = Field(default_factory=list)
    must_have_fields: List[str] = Field(default_factory=list)
    fallback_policy: str = "allow_answer_with_unknowns"


def _load_question_library(path: Optional[Path] = None) -> Dict[str, Dict]:
    """Load question library and index by question_id."""
    p = path or _QUESTION_LIBRARY_PATH
    if not p.exists():
        logger.warning("exec_question_library.yaml not found at %s", p)
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    questions = data.get("questions", [])
    return {q["id"]: q for q in questions if "id" in q}


class QuestionRouter:
    """
    Config-based question router.
    Resolves question_id → RoutedQuestion from exec_question_library.yaml.
    """

    def __init__(self, library_path: Optional[Path] = None):
        self.library = _load_question_library(library_path)

    def route(
        self,
        question_id: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> RoutedQuestion:
        """
        Route a question by ID.

        Args:
            question_id: ID from exec_question_library.yaml (e.g. 'q_reg_4geo').
            overrides: Optional dict to override specific fields.

        Returns:
            RoutedQuestion with all metadata.

        Raises:
            KeyError: if question_id not found in library.
        """
        if question_id not in self.library:
            available = list(self.library.keys())
            raise KeyError(
                f"Question '{question_id}' not found. "
                f"Available: {available}"
            )

        q = self.library[question_id]
        routed = RoutedQuestion(
            question_id=q["id"],
            question_text=q.get("title", ""),
            question_type=q.get("question_type", "unknown"),
            business_lens=q.get("business_lens", "general"),
            required_jurisdictions=q.get("required_jurisdictions", []),
            required_confidence=q.get("required_confidence", "medium"),
            required_sections=q.get("required_sections", []),
            must_have_fields=q.get("must_have_fields", []),
            fallback_policy=q.get("fallback_policy", "allow_answer_with_unknowns"),
        )

        # Apply overrides if provided
        if overrides:
            for key, val in overrides.items():
                if hasattr(routed, key):
                    setattr(routed, key, val)

        return routed

    def list_questions(self) -> List[Dict[str, str]]:
        """List all available questions (id + title)."""
        return [
            {"id": q["id"], "title": q.get("title", ""), "type": q.get("question_type", "")}
            for q in self.library.values()
        ]
