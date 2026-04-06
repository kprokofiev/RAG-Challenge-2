"""
Executive Writer — Sprint 22 WS2-D7
======================================
Transforms AnswerFrame into human-readable executive answer.
Uses LLM only for final prose generation — NOT for fact finding.

Output format:
  1. Conclusion
  2. What supports it
  3. Where uncertainty remains
  4. Why this matters commercially
  5. What we should do next

Constraint: exec_writer MUST NOT invent facts or add facts absent from answer_frame.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ── Template-based writer (default, no LLM needed) ─────────────────────────

_EXEC_TEMPLATE = """## {question_text}

### Conclusion
{conclusion}

### Confidence: {confidence_label}
{confidence_detail}

### What Supports This
{supporting_facts}

### Where Uncertainty Remains
{unknowns_block}

### Why This Matters Commercially
{business_block}

### Recommended Next Steps
{next_actions_block}

---
*Scope: {scope_summary}*
*Claims: {claims_count} | Evidence refs: {evidence_count} | Unknowns: {unknowns_count}*
"""


class ExecWriterResult:
    """Result of exec writer."""
    def __init__(self, markdown: str, short_summary: str, confidence_label: str):
        self.markdown = markdown
        self.short_summary = short_summary
        self.confidence_label = confidence_label

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.markdown,
            "short_summary": self.short_summary,
            "confidence_label": self.confidence_label,
        }


class ExecWriter:
    """
    Generates executive answers from AnswerFrame.

    Two modes:
    - template (default): deterministic markdown from template
    - llm: uses APIProcessor for polished prose (optional, Sprint 22 P1)
    """

    def __init__(self, api_processor=None, mode: str = "template"):
        """
        Args:
            api_processor: Optional APIProcessor for LLM prose generation.
            mode: 'template' (default) or 'llm'.
        """
        self.api = api_processor
        self.mode = mode

    def write(
        self,
        answer_frame,  # AnswerFrame from claim_builder
        lens_profile: Optional[str] = None,
    ) -> ExecWriterResult:
        """
        Generate executive answer from AnswerFrame.
        """
        if self.mode == "llm" and self.api:
            return self._write_llm(answer_frame, lens_profile)
        return self._write_template(answer_frame, lens_profile)

    def _write_template(
        self, answer_frame, lens_profile: Optional[str]
    ) -> ExecWriterResult:
        """Template-based answer generation (default)."""

        # Build conclusion from claims
        strong_claims = [c for c in answer_frame.claims if c.support_level in ("strong", "moderate")]
        weak_claims = [c for c in answer_frame.claims if c.support_level in ("weak", "unsupported")]

        if strong_claims:
            conclusion_lines = []
            for c in strong_claims:
                conclusion_lines.append(f"- {c.text}")
            conclusion = "\n".join(conclusion_lines)
        else:
            conclusion = "Insufficient evidence to form a definitive conclusion."

        # Confidence
        conf = answer_frame.confidence
        confidence_label = conf.get("overall", "unknown")
        confidence_detail = (
            f"Based on {conf.get('total_claims', 0)} claims: "
            f"{conf.get('strong_claims', 0)} strong, "
            f"{conf.get('moderate_claims', 0)} moderate, "
            f"{conf.get('weak_claims', 0)} weak, "
            f"{conf.get('unsupported_claims', 0)} unsupported. "
            f"Answer mode: {conf.get('answer_mode', 'unknown')}."
        )

        # Supporting facts
        supporting_lines = []
        for c in answer_frame.claims:
            if c.support_level in ("strong", "moderate") and c.evidence_refs:
                refs_str = ", ".join(c.evidence_refs[:3])
                supporting_lines.append(f"- {c.text} [refs: {refs_str}]")
        supporting_facts = "\n".join(supporting_lines) if supporting_lines else "No strongly supported facts available."

        # Unknowns
        unknowns_lines = []
        for u in answer_frame.unknowns:
            fp = u.get("field_path", "?")
            msg = u.get("message", u.get("reason_code", "unknown"))
            unknowns_lines.append(f"- **{fp}**: {msg}")
        if weak_claims:
            for c in weak_claims:
                unknowns_lines.append(f"- {c.text} (weak/no evidence)")
        unknowns_block = "\n".join(unknowns_lines) if unknowns_lines else "No significant uncertainties identified."

        # Business implications
        business_block = "\n".join(
            f"- {imp}" for imp in answer_frame.business_implication
        ) if answer_frame.business_implication else "No specific commercial implications identified."

        # Next actions
        next_actions_block = "\n".join(
            f"- {act}" for act in answer_frame.recommended_next_actions
        ) if answer_frame.recommended_next_actions else "No immediate actions required."

        # Scope summary
        scope = answer_frame.scope
        scope_parts = [f"mode={scope.get('entity_mode', '?')}"]
        jur = scope.get("jurisdictions", {})
        if jur:
            scope_parts.append(f"jurisdictions={','.join(jur.keys())}")
        warnings = scope.get("warnings", [])
        if warnings:
            scope_parts.append(f"warnings={len(warnings)}")
        scope_summary = " | ".join(scope_parts)

        # All evidence refs
        all_refs = set()
        for c in answer_frame.claims:
            all_refs.update(c.evidence_refs)

        markdown = _EXEC_TEMPLATE.format(
            question_text=answer_frame.question_text,
            conclusion=conclusion,
            confidence_label=confidence_label.upper(),
            confidence_detail=confidence_detail,
            supporting_facts=supporting_facts,
            unknowns_block=unknowns_block,
            business_block=business_block,
            next_actions_block=next_actions_block,
            scope_summary=scope_summary,
            claims_count=len(answer_frame.claims),
            evidence_count=len(all_refs),
            unknowns_count=len(answer_frame.unknowns),
        )

        # Short summary
        short = self._build_short_summary(answer_frame, confidence_label)

        return ExecWriterResult(
            markdown=markdown.strip(),
            short_summary=short,
            confidence_label=confidence_label,
        )

    def _write_llm(
        self, answer_frame, lens_profile: Optional[str]
    ) -> ExecWriterResult:
        """LLM-based prose generation from AnswerFrame."""
        # Build the prompt — strictly constrained to answer_frame contents
        claims_text = "\n".join(
            f"- [{c.support_level}] {c.text}" for c in answer_frame.claims
        )
        unknowns_text = "\n".join(
            f"- {u.get('field_path', '?')}: {u.get('message', '')}"
            for u in answer_frame.unknowns
        )
        implications_text = "\n".join(
            f"- {imp}" for imp in answer_frame.business_implication
        )

        system_prompt = (
            "You are an executive pharma analyst. Write a concise executive answer "
            "based STRICTLY on the claims, unknowns, and implications provided. "
            "You MUST NOT invent or add any facts not present in the input. "
            "Structure: 1) Conclusion 2) Supporting evidence 3) Uncertainties "
            "4) Commercial relevance 5) Recommended actions."
        )
        if lens_profile:
            system_prompt += f"\nBusiness lens: {lens_profile}."

        human_prompt = (
            f"Question: {answer_frame.question_text}\n\n"
            f"Confidence: {answer_frame.confidence.get('overall', 'unknown')}\n\n"
            f"Claims:\n{claims_text}\n\n"
            f"Unknowns:\n{unknowns_text or 'None'}\n\n"
            f"Business Implications:\n{implications_text or 'None'}\n\n"
            f"Next Actions:\n"
            + "\n".join(f"- {a}" for a in answer_frame.recommended_next_actions)
        )

        try:
            response = self.api.send_message(
                system_content=system_prompt,
                human_content=human_prompt,
                temperature=0.3,
            )
            markdown = response if isinstance(response, str) else str(response)
        except Exception as e:
            logger.error("LLM exec write failed: %s; falling back to template", e)
            return self._write_template(answer_frame, lens_profile)

        confidence_label = answer_frame.confidence.get("overall", "unknown")
        short = self._build_short_summary(answer_frame, confidence_label)

        return ExecWriterResult(
            markdown=markdown,
            short_summary=short,
            confidence_label=confidence_label,
        )

    @staticmethod
    def _build_short_summary(answer_frame, confidence_label: str) -> str:
        """Build a 1-2 sentence summary."""
        strong = [c for c in answer_frame.claims if c.support_level in ("strong", "moderate")]
        if strong:
            first = strong[0].text[:120]
            return f"{first}. Confidence: {confidence_label}. {len(answer_frame.claims)} claims, {len(answer_frame.unknowns)} unknowns."
        return f"Insufficient data. Confidence: {confidence_label}. {len(answer_frame.unknowns)} unknowns remaining."
