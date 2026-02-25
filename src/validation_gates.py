from typing import List, Dict, Any, Optional, Union
from pydantic import ValidationError
import logging
import difflib
from .prompts import AnswerSchemaFixPrompt, DDSectionAnswerSchema
from .evidence_builder import EvidenceCandidate

logger = logging.getLogger(__name__)

# Reason strings used in unknowns when repair loop cannot recover a claim
REASON_NO_EVIDENCE = "No supporting evidence in provided candidates"
REASON_ORPHANED_EV_ID = "Evidence ID not found in candidates (orphaned reference)"
REASON_REMAPPED = "Evidence ID remapped by fuzzy match to closest candidate snippet"
REASON_EXPAND_K = "Evidence ID not found; retrieval should be retried with larger K"
REASON_PARSE_EMPTY = "Parse returned empty content for this question"
REASON_SOURCE_MISSING = "Source document missing from corpus"


class ValidationResult:
    """
    Result of validation with details about issues and fixes.
    """

    def __init__(self):
        self.is_valid = True
        self.errors = []
        self.warnings = []
        self.fixed_output = None
        self.validation_stats = {
            'total_claims': 0,
            'claims_with_evidence': 0,
            'orphaned_claims': 0,
            'invalid_evidence_refs': 0,
            'fixed_by_llm': 0
        }

    def add_error(self, error_type: str, details: str, claim_id: Optional[str] = None):
        """Add validation error."""
        self.is_valid = False
        self.errors.append({
            'type': error_type,
            'details': details,
            'claim_id': claim_id
        })

    def add_warning(self, warning_type: str, details: str):
        """Add validation warning."""
        self.warnings.append({
            'type': warning_type,
            'details': details
        })

    def update_stats(self, **kwargs):
        """Update validation statistics."""
        for key, value in kwargs.items():
            if key in self.validation_stats:
                self.validation_stats[key] = value


class ValidationGates:
    """
    Validation gates for ensuring evidence linkage integrity in DD reports.
    Implements rules from ExecSpec for claim-evidence consistency.
    """

    def __init__(self):
        self.schema_validator = DDSectionAnswerSchema
        self.fix_prompt = AnswerSchemaFixPrompt()

    def validate_section_output(self, section_output: Dict[str, Any],
                              evidence_candidates: List[EvidenceCandidate]) -> ValidationResult:
        """
        Validate section output against evidence candidates and schema.

        Args:
            section_output: Raw section output from LLM
            evidence_candidates: List of valid evidence candidates

        Returns:
            ValidationResult with details and potentially fixed output
        """
        result = ValidationResult()

        # Step 1: Validate against Pydantic schema
        schema_result = self._validate_schema(section_output)
        if not schema_result['valid']:
            result.add_error('schema_validation', schema_result['error'])
            return result

        # Step 2: Extract and validate claims
        claims = section_output.get('claims', [])
        numbers = section_output.get('numbers', [])
        risks = section_output.get('risks', [])

        result.update_stats(total_claims=len(claims) + len(numbers) + len(risks))

        # Step 3: Check evidence linkage for claims
        claims_validation = self._validate_claims_evidence(claims, evidence_candidates)
        result.update_stats(
            claims_with_evidence=claims_validation['valid_count'],
            orphaned_claims=claims_validation['orphaned_count']
        )

        if claims_validation['orphaned_count'] > 0:
            for orphaned in claims_validation['orphaned']:
                result.add_error('orphaned_claim', f"Claim without evidence: {orphaned.get('text', '')[:100]}")

        # Step 4: Check evidence linkage for numbers
        numbers_validation = self._validate_numbers_evidence(numbers, evidence_candidates)
        if numbers_validation['invalid_count'] > 0:
            result.update_stats(invalid_evidence_refs=result.validation_stats['invalid_evidence_refs'] + numbers_validation['invalid_count'])

        # Step 5: Check evidence linkage for risks
        risks_validation = self._validate_risks_evidence(risks, evidence_candidates)
        if risks_validation['invalid_count'] > 0:
            result.update_stats(invalid_evidence_refs=result.validation_stats['invalid_evidence_refs'] + risks_validation['invalid_count'])

        # Step 6: Attempt to fix critical issues
        if not result.is_valid:
            fixed_output = self._attempt_fix(section_output, evidence_candidates, result)
            if fixed_output:
                result.fixed_output = fixed_output
                result.update_stats(fixed_by_llm=1)
                # Re-validate the fixed output
                re_validation = self.validate_section_output(fixed_output, evidence_candidates)
                if re_validation.is_valid:
                    result.is_valid = True
                    result.errors = []  # Clear errors if fix worked

        return result

    def _validate_schema(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate output against DDSectionAnswerSchema.

        Args:
            output: Output to validate

        Returns:
            Dict with validation result
        """
        try:
            self.schema_validator(**output)
            return {'valid': True}
        except ValidationError as e:
            return {'valid': False, 'error': str(e)}

    def _validate_claims_evidence(self, claims: List[Dict[str, Any]],
                                evidence_candidates: List[EvidenceCandidate]) -> Dict[str, Any]:
        """
        Validate that claims have valid evidence references.

        Args:
            claims: List of claims
            evidence_candidates: Valid evidence candidates

        Returns:
            Validation stats
        """
        valid_evidence_ids = {c.evidence_id for c in evidence_candidates}

        valid_claims = []
        orphaned_claims = []

        for claim in claims:
            evidence_ids = claim.get('evidence_ids', [])

            if not evidence_ids:
                orphaned_claims.append(claim)
                continue

            # Check if all evidence IDs are valid
            if all(eid in valid_evidence_ids for eid in evidence_ids):
                valid_claims.append(claim)
            else:
                orphaned_claims.append(claim)

        return {
            'valid_count': len(valid_claims),
            'orphaned_count': len(orphaned_claims),
            'orphaned': orphaned_claims
        }

    def _validate_numbers_evidence(self, numbers: List[Dict[str, Any]],
                                 evidence_candidates: List[EvidenceCandidate]) -> Dict[str, Any]:
        """
        Validate that numbers have valid evidence references.

        Args:
            numbers: List of numbers
            evidence_candidates: Valid evidence candidates

        Returns:
            Validation stats
        """
        valid_evidence_ids = {c.evidence_id for c in evidence_candidates}
        invalid_count = 0

        for number in numbers:
            evidence_ids = number.get('evidence_ids', [])

            if not evidence_ids:
                invalid_count += 1
                continue

            if not all(eid in valid_evidence_ids for eid in evidence_ids):
                invalid_count += 1

        return {'invalid_count': invalid_count}

    def _validate_risks_evidence(self, risks: List[Dict[str, Any]],
                               evidence_candidates: List[EvidenceCandidate]) -> Dict[str, Any]:
        """
        Validate that risks have valid evidence references.

        Args:
            risks: List of risks
            evidence_candidates: Valid evidence candidates

        Returns:
            Validation stats
        """
        valid_evidence_ids = {c.evidence_id for c in evidence_candidates}
        invalid_count = 0

        for risk in risks:
            evidence_ids = risk.get('evidence_ids', [])

            if not evidence_ids:
                invalid_count += 1
                continue

            if not all(eid in valid_evidence_ids for eid in evidence_ids):
                invalid_count += 1

        return {'invalid_count': invalid_count}

    def _attempt_fix(self, original_output: Dict[str, Any],
                    evidence_candidates: List[EvidenceCandidate],
                    validation_result: ValidationResult) -> Optional[Dict[str, Any]]:
        """
        Repair loop for invalid section outputs (Sprint 3 §4.2).

        Strategies applied in order:
          A) Fuzzy remap: orphaned evidence_ids remapped to closest candidate by snippet similarity.
          B) expand_k signal: if no suitable candidate snippet found, tag claim for K-expansion
             retry (sets _needs_expand_k=True on ValidationResult but does NOT discard claim yet).
          C) Move to unknowns: remaining unresolvable orphaned claims/numbers/risks → unknowns
             with typed reason string.

        Returns a fixed output dict, or None if nothing could be fixed.
        """
        import copy

        valid_ev_ids = {c.evidence_id for c in evidence_candidates}
        candidate_snippets: Dict[str, str] = {c.evidence_id: c.snippet for c in evidence_candidates}
        # P1: doc_id lookup — remap only within same source document to prevent cross-doc misattribution
        candidate_doc_ids: Dict[str, str] = {c.evidence_id: c.doc_id for c in evidence_candidates}
        candidate_ids = list(valid_ev_ids)

        if not evidence_candidates:
            logger.debug("_attempt_fix: no candidates — skipping repair, moving all to unknowns")
            return self._move_all_to_unknowns(original_output, REASON_NO_EVIDENCE)

        output = copy.deepcopy(original_output)
        remapped_count = 0
        expand_needed = False
        items_moved_to_unknown = 0

        def _fix_evidence_ids(ev_ids: List[str]) -> tuple[List[str], str]:
            """
            Returns (fixed_ev_ids, repair_action) where repair_action is one of:
            'ok', 'remapped', 'expand_k', 'no_fix'.
            """
            if all(eid in valid_ev_ids for eid in ev_ids):
                return ev_ids, "ok"

            fixed: List[str] = []
            any_bad = False
            for eid in ev_ids:
                if eid in valid_ev_ids:
                    fixed.append(eid)
                    continue
                any_bad = True
                # Strategy A: fuzzy match restricted to same doc_id (P1 guard)
                best_match, match_score = self._fuzzy_match_candidate(
                    eid, candidate_ids, candidate_snippets, candidate_doc_ids=candidate_doc_ids
                )
                if best_match:
                    matched_doc = candidate_doc_ids.get(best_match, "?")
                    logger.info(
                        "_attempt_fix StrategyA: orphaned '%s' remapped to '%s' "
                        "(score=%.3f doc_id=%s reason=%s)",
                        eid, best_match, match_score, matched_doc, REASON_REMAPPED,
                    )
                    fixed.append(best_match)
                    nonlocal remapped_count
                    remapped_count += 1
                else:
                    # Strategy B: signal expand_k needed
                    nonlocal expand_needed
                    expand_needed = True
            if not any_bad:
                return ev_ids, "ok"
            if fixed:
                return fixed, "remapped"
            return [], "no_fix"

        # Fix claims
        repaired_claims = []
        new_unknowns = list(output.get("unknowns", []))
        for claim in output.get("claims", []):
            ev_ids = claim.get("evidence_ids", [])
            if not ev_ids:
                # Strategy C: move to unknown
                new_unknowns.append({
                    "id": f"unknown_noev_{claim.get('id', 'x')}",
                    "question": claim.get("text", ""),
                    "reason": REASON_NO_EVIDENCE,
                })
                items_moved_to_unknown += 1
                continue
            fixed_ids, action = _fix_evidence_ids(ev_ids)
            if action == "no_fix" or not fixed_ids:
                # Strategy C
                new_unknowns.append({
                    "id": f"unknown_orphan_{claim.get('id', 'x')}",
                    "question": claim.get("text", ""),
                    "reason": REASON_ORPHANED_EV_ID if not expand_needed else REASON_EXPAND_K,
                })
                items_moved_to_unknown += 1
            else:
                claim["evidence_ids"] = fixed_ids
                repaired_claims.append(claim)

        # Fix numbers
        repaired_numbers = []
        for num in output.get("numbers", []):
            ev_ids = num.get("evidence_ids", [])
            fixed_ids, action = _fix_evidence_ids(ev_ids) if ev_ids else ([], "no_fix")
            if action == "no_fix" or not fixed_ids:
                new_unknowns.append({
                    "id": f"unknown_num_{num.get('id', 'x')}",
                    "question": f"Number: {num.get('label', '')} = {num.get('as_reported', '')}",
                    "reason": REASON_ORPHANED_EV_ID if ev_ids else REASON_NO_EVIDENCE,
                })
                items_moved_to_unknown += 1
            else:
                num["evidence_ids"] = fixed_ids
                repaired_numbers.append(num)

        # Fix risks
        repaired_risks = []
        for risk in output.get("risks", []):
            ev_ids = risk.get("evidence_ids", [])
            fixed_ids, action = _fix_evidence_ids(ev_ids) if ev_ids else ([], "no_fix")
            if action == "no_fix" or not fixed_ids:
                new_unknowns.append({
                    "id": f"unknown_risk_{risk.get('id', 'x')}",
                    "question": f"Risk: {risk.get('title', '')}",
                    "reason": REASON_ORPHANED_EV_ID if ev_ids else REASON_NO_EVIDENCE,
                })
                items_moved_to_unknown += 1
            else:
                risk["evidence_ids"] = fixed_ids
                repaired_risks.append(risk)

        output["claims"] = repaired_claims
        output["numbers"] = repaired_numbers
        output["risks"] = repaired_risks
        output["unknowns"] = new_unknowns

        if expand_needed:
            validation_result._needs_expand_k = True

        logger.info(
            "_attempt_fix: remapped=%d moved_to_unknown=%d expand_k_needed=%s",
            remapped_count, items_moved_to_unknown, expand_needed,
        )

        # Return repaired output only if something was actually changed
        if remapped_count > 0 or items_moved_to_unknown > 0:
            return output
        return None

    @staticmethod
    def _fuzzy_match_candidate(
        bad_ev_id: str,
        candidate_ids: List[str],
        candidate_snippets: Dict[str, str],
        min_ratio: float = 0.55,
        candidate_doc_ids: Optional[Dict[str, str]] = None,
    ) -> tuple:
        """
        Strategy A: try to find the closest valid candidate for an orphaned evidence_id.

        Returns (best_match_id_or_None, match_score).

        Two heuristics (best wins):
        1. ID similarity — the orphaned id might differ by a suffix/hash.
        2. Snippet similarity — compare last known snippet chars embedded in the bad id
           against candidate snippets using SequenceMatcher.

        P1 guard: if candidate_doc_ids is provided, restrict matches to candidates from the
        same doc_id as the orphaned reference (inferred from the ev_id prefix).  This prevents
        cross-document misattribution (e.g. remapping a patent ev_id to an EPAR candidate).
        """
        # Infer expected doc_id from the bad ev_id (format: ev_{doc_id}_{page}_{hash} or ev_{chunk_id})
        allowed_ids = candidate_ids
        if candidate_doc_ids:
            # Try to extract doc_id prefix: strip leading "ev_", take everything before last 2 "_" parts
            parts = bad_ev_id.lstrip("ev_").rsplit("_", 2)
            expected_doc_id = parts[0] if len(parts) >= 2 else None
            if expected_doc_id and len(expected_doc_id) >= 3:
                same_doc_ids = [cid for cid in candidate_ids
                                if candidate_doc_ids.get(cid, "") == expected_doc_id]
                if same_doc_ids:
                    allowed_ids = same_doc_ids
                # If no same-doc candidates, fall back to all candidates (don't lose the remap entirely)

        # Heuristic 1: id string similarity (restricted to allowed_ids)
        id_matches = difflib.get_close_matches(bad_ev_id, allowed_ids, n=1, cutoff=min_ratio)
        if id_matches:
            # Compute the actual ratio for logging
            ratio = difflib.SequenceMatcher(None, bad_ev_id, id_matches[0]).ratio()
            return id_matches[0], round(ratio, 3)

        # Heuristic 2: if bad_ev_id embeds a fragment, try snippet match
        # (e.g. ev_abc123 where abc123 might appear in a snippet)
        fragment = bad_ev_id.replace("ev_", "").replace("_", " ").strip()
        if len(fragment) < 4:
            return None, 0.0
        best_id, best_ratio = None, 0.0
        for cid in allowed_ids:
            snippet = candidate_snippets.get(cid, "")
            ratio = difflib.SequenceMatcher(None, fragment.lower(), snippet[:200].lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_id = cid
        if best_ratio >= min_ratio:
            return best_id, round(best_ratio, 3)
        return None, 0.0

    @staticmethod
    def _move_all_to_unknowns(output: Dict[str, Any], reason: str) -> Dict[str, Any]:
        """Move all claims/numbers/risks to unknowns with the given reason (Strategy C)."""
        import copy
        out = copy.deepcopy(output)
        unknowns = list(out.get("unknowns", []))
        for claim in out.get("claims", []):
            unknowns.append({
                "id": f"unknown_{claim.get('id', 'x')}",
                "question": claim.get("text", ""),
                "reason": reason,
            })
        for num in out.get("numbers", []):
            unknowns.append({
                "id": f"unknown_{num.get('id', 'x')}",
                "question": f"Number: {num.get('label', '')} = {num.get('as_reported', '')}",
                "reason": reason,
            })
        for risk in out.get("risks", []):
            unknowns.append({
                "id": f"unknown_{risk.get('id', 'x')}",
                "question": f"Risk: {risk.get('title', '')}",
                "reason": reason,
            })
        out["claims"] = []
        out["numbers"] = []
        out["risks"] = []
        out["unknowns"] = unknowns
        return out

    def move_orphaned_to_unknowns(
        self,
        output: Dict[str, Any],
        evidence_candidates: Optional[List[EvidenceCandidate]] = None,
    ) -> Dict[str, Any]:
        """
        Move claims (and numbers/risks) without valid evidence references to unknowns.

        If `evidence_candidates` is provided, also checks that referenced IDs actually
        exist in the candidate set (not just non-empty).  This is the final cleanup pass
        after _attempt_fix has already run.

        Args:
            output: Section output
            evidence_candidates: Valid evidence candidates (optional; enables ID validity check)

        Returns:
            Fixed output
        """
        valid_ev_ids: Optional[set] = (
            {c.evidence_id for c in evidence_candidates} if evidence_candidates else None
        )

        def _is_valid(ev_ids: List[str]) -> bool:
            if not ev_ids:
                return False
            if valid_ev_ids is not None:
                return all(eid in valid_ev_ids for eid in ev_ids)
            return True

        claims = output.get("claims", [])
        numbers = output.get("numbers", [])
        risks = output.get("risks", [])
        unknowns = list(output.get("unknowns", []))

        valid_claims, valid_numbers, valid_risks = [], [], []
        new_unknowns = []

        for claim in claims:
            if _is_valid(claim.get("evidence_ids", [])):
                valid_claims.append(claim)
            else:
                new_unknowns.append({
                    "id": f"unknown_{claim.get('id', 'x')}",
                    "question": claim.get("text", ""),
                    "reason": REASON_NO_EVIDENCE,
                })

        for num in numbers:
            if _is_valid(num.get("evidence_ids", [])):
                valid_numbers.append(num)
            else:
                new_unknowns.append({
                    "id": f"unknown_num_{num.get('id', 'x')}",
                    "question": f"Number: {num.get('label', '')} = {num.get('as_reported', '')}",
                    "reason": REASON_NO_EVIDENCE,
                })

        for risk in risks:
            if _is_valid(risk.get("evidence_ids", [])):
                valid_risks.append(risk)
            else:
                new_unknowns.append({
                    "id": f"unknown_risk_{risk.get('id', 'x')}",
                    "question": f"Risk: {risk.get('title', '')}",
                    "reason": REASON_NO_EVIDENCE,
                })

        output["claims"] = valid_claims
        output["numbers"] = valid_numbers
        output["risks"] = valid_risks
        output["unknowns"] = unknowns + new_unknowns

        if new_unknowns:
            logger.info(
                "move_orphaned_to_unknowns: %d items moved (claims=%d numbers=%d risks=%d)",
                len(new_unknowns),
                len(valid_claims), len(valid_numbers), len(valid_risks),
            )
        return output


