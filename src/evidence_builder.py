import hashlib
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class EvidenceCandidate:
    """
    A candidate piece of evidence from retrieved chunks.
    Used to build the evidence candidates list for LLM prompting.
    """

    def __init__(self, chunk: Dict[str, Any], doc_id: str, doc_title: Optional[str] = None):
        self.evidence_id = self._generate_evidence_id(chunk, doc_id)
        self.doc_id = doc_id
        self.doc_title = doc_title
        self.page = chunk.get('page_from', chunk.get('page', 0))
        self.snippet = self._extract_snippet(chunk)
        self.chunk_type = chunk.get('type', 'content')
        self.confidence = self._calculate_confidence(chunk)

    def _generate_evidence_id(self, chunk: Dict[str, Any], doc_id: str) -> str:
        """
        Generate deterministic evidence ID from chunk data.

        Args:
            chunk: Chunk data
            doc_id: Document identifier

        Returns:
            Deterministic evidence ID
        """
        # Use chunk ID or create hash-based ID
        chunk_id = chunk.get('id', chunk.get('chunk_id', ''))
        if chunk_id:
            return f"ev_{chunk_id}"

        # Fallback: hash-based ID
        content_hash = hashlib.md5(f"{doc_id}_{self.page}_{chunk.get('text', '')[:100]}".encode()).hexdigest()[:8]
        return f"ev_{doc_id}_{self.page}_{content_hash}"

    def _extract_snippet(self, chunk: Dict[str, Any], max_length: int = 400) -> str:
        """
        Extract and sanitize snippet from chunk.

        Args:
            chunk: Chunk data
            max_length: Maximum snippet length

        Returns:
            Sanitized text snippet
        """
        text = chunk.get('text', '').strip()

        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length - 3] + "..."

        # Basic sanitization
        text = text.replace('\n\n\n', '\n\n')  # Remove excessive newlines
        text = text.replace('\t', ' ')  # Replace tabs with spaces

        return text

    def _calculate_confidence(self, chunk: Dict[str, Any]) -> float:
        """
        Calculate confidence score for this evidence candidate.

        Args:
            chunk: Chunk data

        Returns:
            Confidence score 0-1
        """
        confidence = 0.5  # Base confidence

        # Boost for structured content
        if chunk.get('type') == 'table':
            confidence += 0.2

        # Boost for specific terms
        text_lower = chunk.get('text', '').lower()
        evidence_terms = ['approved', 'authorized', 'indication', 'contraindication',
                         'phase 3', 'primary endpoint', 'maholder', 'marketing authorization']

        term_matches = sum(1 for term in evidence_terms if term in text_lower)
        confidence += min(term_matches * 0.1, 0.3)

        # Boost for numbers and dates
        if any(char.isdigit() for char in chunk.get('text', '')):
            confidence += 0.1

        return min(confidence, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "evidence_id": self.evidence_id,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "page": self.page,
            "snippet": self.snippet,
            "chunk_type": self.chunk_type,
            "confidence": self.confidence
        }


class EvidenceCandidatesBuilder:
    """
    Builds evidence candidates from retrieved chunks.
    Ensures deterministic evidence IDs and proper snippet extraction.
    """

    def __init__(self):
        self.max_snippet_length = 400
        self.min_confidence_threshold = 0.3

    def build_candidates(self, retrieved_chunks: List[Dict[str, Any]],
                        doc_id: str, doc_title: Optional[str] = None) -> List[EvidenceCandidate]:
        """
        Build evidence candidates from retrieved chunks.

        Args:
            retrieved_chunks: List of retrieved chunk dictionaries
            doc_id: Document identifier
            doc_title: Optional document title

        Returns:
            List of EvidenceCandidate objects
        """
        candidates = []

        for chunk in retrieved_chunks:
            try:
                candidate = EvidenceCandidate(chunk, doc_id, doc_title)

                # Filter low-confidence candidates
                if candidate.confidence >= self.min_confidence_threshold:
                    candidates.append(candidate)

            except Exception as e:
                logger.warning(f"Failed to create evidence candidate from chunk: {e}")
                continue

        # Remove duplicates by evidence_id
        unique_candidates = self._deduplicate_candidates(candidates)

        logger.info(f"Built {len(unique_candidates)} evidence candidates from {len(retrieved_chunks)} chunks")
        return unique_candidates

    def build_candidates_from_multiple_docs(self, retrieved_chunks_by_doc: Dict[str, List[Dict[str, Any]]],
                                          doc_titles: Optional[Dict[str, str]] = None) -> List[EvidenceCandidate]:
        """
        Build evidence candidates from chunks retrieved from multiple documents.

        Args:
            retrieved_chunks_by_doc: Dict mapping doc_id to list of retrieved chunks
            doc_titles: Optional dict mapping doc_id to document titles

        Returns:
            Combined list of EvidenceCandidate objects
        """
        all_candidates = []

        for doc_id, chunks in retrieved_chunks_by_doc.items():
            doc_title = doc_titles.get(doc_id) if doc_titles else None
            doc_candidates = self.build_candidates(chunks, doc_id, doc_title)
            all_candidates.extend(doc_candidates)

        # Final deduplication across all documents
        unique_candidates = self._deduplicate_candidates(all_candidates)

        logger.info(f"Built {len(unique_candidates)} total evidence candidates from {len(retrieved_chunks_by_doc)} documents")
        return unique_candidates

    def _deduplicate_candidates(self, candidates: List[EvidenceCandidate]) -> List[EvidenceCandidate]:
        """
        Remove duplicate candidates by evidence_id, keeping the highest confidence one.

        Args:
            candidates: List of candidates (may have duplicates)

        Returns:
            Deduplicated list
        """
        candidate_map = {}

        for candidate in candidates:
            evidence_id = candidate.evidence_id

            if evidence_id not in candidate_map:
                candidate_map[evidence_id] = candidate
            else:
                # Keep the one with higher confidence
                existing = candidate_map[evidence_id]
                if candidate.confidence > existing.confidence:
                    candidate_map[evidence_id] = candidate

        return list(candidate_map.values())

    def candidates_to_prompt_format(self, candidates: List[EvidenceCandidate]) -> str:
        """
        Convert candidates to format suitable for LLM prompting.

        Args:
            candidates: List of evidence candidates

        Returns:
            Formatted string for prompt inclusion
        """
        if not candidates:
            return "No evidence available."

        formatted_parts = []

        for candidate in candidates:
            doc_info = f"Doc: {candidate.doc_title or candidate.doc_id}"
            page_info = f"Page {candidate.page}"
            snippet = candidate.snippet

            part = f"{doc_info} ({page_info}): {snippet}"
            formatted_parts.append(part)

        return "\n\n".join(formatted_parts)

    def validate_evidence_references(self, claims: List[Dict[str, Any]],
                                   candidates: List[EvidenceCandidate]) -> Dict[str, Any]:
        """
        Validate that claims reference valid evidence IDs.

        Args:
            claims: List of claims with evidence_ids
            candidates: List of valid evidence candidates

        Returns:
            Validation result with errors and valid claims
        """
        valid_evidence_ids = {c.evidence_id for c in candidates}
        valid_claims = []
        invalid_claims = []

        for claim in claims:
            evidence_ids = claim.get('evidence_ids', [])

            if not evidence_ids:
                invalid_claims.append({
                    'claim': claim,
                    'error': 'No evidence_ids provided'
                })
                continue

            invalid_ids = [eid for eid in evidence_ids if eid not in valid_evidence_ids]

            if invalid_ids:
                invalid_claims.append({
                    'claim': claim,
                    'error': f'Invalid evidence_ids: {invalid_ids}'
                })
            else:
                valid_claims.append(claim)

        return {
            'valid_claims': valid_claims,
            'invalid_claims': invalid_claims,
            'total_claims': len(claims),
            'valid_count': len(valid_claims),
            'invalid_count': len(invalid_claims)
        }
