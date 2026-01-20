from typing import List, Dict, Any, Optional, Union
from pydantic import ValidationError
import logging
from .prompts import AnswerSchemaFixPrompt, DDSectionAnswerSchema
from .evidence_builder import EvidenceCandidate

logger = logging.getLogger(__name__)


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
        Attempt to fix validation issues using LLM.

        Args:
            original_output: Original invalid output
            evidence_candidates: Valid evidence candidates
            validation_result: Current validation result

        Returns:
            Fixed output or None if fix failed
        """
        try:
            # Prepare schema definition
            schema_definition = f"""
            class Claim(BaseModel):
                id: str
                text: str
                evidence_ids: List[str]
                confidence: Optional[float]

            class NumberWithEvidence(BaseModel):
                id: str
                label: str
                value: Union[float, int]
                as_reported: str
                unit: Optional[str]
                currency: Optional[str]
                scale: Optional[str]
                period: Optional[str]
                as_of_date: Optional[str]
                evidence_ids: List[str]

            class Risk(BaseModel):
                id: str
                title: str
                severity: str
                description: str
                evidence_ids: List[str]

            class Unknown(BaseModel):
                id: str
                question: str
                reason: str

            class Evidence(BaseModel):
                id: str
                doc_id: str
                doc_title: Optional[str]
                page: int
                snippet: str

            class DDSectionAnswerSchema(BaseModel):
                section_id: str
                claims: List[Claim]
                numbers: List[NumberWithEvidence]
                risks: List[Risk]
                unknowns: List[Unknown]
                evidence: List[Evidence]
                notes: Optional[str]
            """

            # Prepare evidence list for context
            evidence_list = "\n".join([
                f"- {c.evidence_id}: {c.snippet[:100]}..."
                for c in evidence_candidates
            ])

            # Create fix prompt
            user_prompt = f"""
Schema definition:
{schema_definition}

Available evidence IDs:
{evidence_list}

Raw invalid output:
{original_output}

Fix the output to be valid according to DDSectionAnswerSchema. Move invalid claims to unknowns if needed.
"""

            # This would call LLM to fix the output
            # For now, return None as placeholder
            logger.info("LLM fix attempted but not implemented yet")
            return None

        except Exception as e:
            logger.error(f"Failed to attempt LLM fix: {e}")
            return None

    def move_orphaned_to_unknowns(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Move claims without evidence to unknowns section.

        Args:
            output: Section output with orphaned claims

        Returns:
            Fixed output
        """
        claims = output.get('claims', [])
        unknowns = output.get('unknowns', [])

        valid_claims = []
        new_unknowns = []

        for claim in claims:
            if claim.get('evidence_ids'):
                valid_claims.append(claim)
            else:
                # Move to unknowns
                unknown = {
                    'id': f"unknown_{claim['id']}",
                    'question': claim.get('text', ''),
                    'reason': 'No supporting evidence found'
                }
                new_unknowns.append(unknown)

        output['claims'] = valid_claims
        output['unknowns'] = unknowns + new_unknowns

        return output


