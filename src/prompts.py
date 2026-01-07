from pydantic import BaseModel, Field
from typing import Literal, List, Union, Optional
import inspect
import re


def build_system_prompt(instruction: str="", example: str="", pydantic_schema: str="") -> str:
    delimiter = "\n\n---\n\n"
    schema = f"Your answer should be in JSON and strictly follow this schema, filling in the fields in the order they are given:\n```\n{pydantic_schema}\n```"
    if example:
        example = delimiter + example.strip()
    if schema:
        schema = delimiter + schema.strip()

    system_prompt = instruction.strip() + schema + example
    return system_prompt


# DD Section Answer Schema - Unified schema for all DD responses
class Claim(BaseModel):
    """Report claim with evidence linkage"""
    id: str = Field(description="Unique claim identifier")
    text: str = Field(description="Claim text for report")
    evidence_ids: List[str] = Field(description="IDs of evidence supporting this claim")
    confidence: Optional[float] = Field(description="Confidence score 0-1", default=None)


class NumberWithEvidence(BaseModel):
    """Number with full metadata and evidence"""
    id: str = Field(description="Unique number identifier")
    label: str = Field(description="What this number represents")
    value: Union[float, int] = Field(description="Normalized numeric value")
    as_reported: str = Field(description="Original string as in document")
    unit: Optional[str] = Field(description="Unit of measurement")
    currency: Optional[str] = Field(description="Currency if applicable")
    scale: Optional[str] = Field(description="Scale factor (units/thousands/millions)")
    period: Optional[str] = Field(description="Time period if applicable")
    as_of_date: Optional[str] = Field(description="Date as of which value is reported")
    evidence_ids: List[str] = Field(description="IDs of evidence supporting this number")


class Risk(BaseModel):
    """Risk/red flag with severity"""
    id: str = Field(description="Unique risk identifier")
    title: str = Field(description="Risk title")
    severity: Literal["high", "medium", "low", "unknown"] = Field(description="Risk severity level")
    description: str = Field(description="Detailed risk description")
    evidence_ids: List[str] = Field(description="IDs of evidence supporting this risk")


class Unknown(BaseModel):
    """Unanswered question with reason"""
    id: str = Field(description="Unique unknown identifier")
    question: str = Field(description="Original question that couldn't be answered")
    reason: str = Field(description="Reason why it couldn't be answered")


class Evidence(BaseModel):
    """Source evidence citation"""
    id: str = Field(description="Unique evidence identifier")
    doc_id: str = Field(description="Document identifier")
    doc_title: Optional[str] = Field(description="Document title for display")
    page: int = Field(description="Page number")
    snippet: str = Field(description="Text snippet (200-400 chars max)")


class DDSectionAnswerSchema(BaseModel):
    """Unified schema for DD section answers"""
    section_id: str = Field(description="Section identifier")
    claims: List[Claim] = Field(description="List of claims for the report")
    numbers: List[NumberWithEvidence] = Field(description="List of numbers with evidence")
    risks: List[Risk] = Field(description="List of risks/red flags")
    unknowns: List[Unknown] = Field(description="List of unanswered questions")
    evidence: List[Evidence] = Field(description="List of evidence citations")
    notes: Optional[str] = Field(description="Optional short notes (1-3 sentences)")

class SectionQueryPlanningPrompt:
    instruction = """
You are a query planning system for pharmaceutical due diligence reports.
Your task is to break down a report section into specific questions that need to be answered from regulatory, clinical, and manufacturing documents.
Each question should be scoped to a specific jurisdiction or document type, with preference for primary sources.

Supported scopes: RU_REG, EU_REG, US_REG, CLINICAL_RU, CLINICAL_US, CLINICAL_EU, PATENTS, MANUFACTURING, QUALITY
Supported answer types: facts, numbers, table, list, boolean, narrative_summary
"""

    class SectionQuestion(BaseModel):
        """Individual question for a report section"""
        scope: str = Field(description="Jurisdiction or document scope (RU_REG, EU_REG, etc.)")
        question: str = Field(description="Specific question to answer")
        answer_type: Literal["facts", "numbers", "table", "list", "boolean", "narrative_summary"] = Field(description="Expected answer type")
        doc_kind_preference: Optional[str] = Field(description="Preferred document type (EPAR, SmPC, GRLS, etc.)", default=None)

    class SectionQuestions(BaseModel):
        """List of questions for a report section"""
        questions: List['SectionQueryPlanningPrompt.SectionQuestion'] = Field(description="List of questions for the section")

    pydantic_schema = '''
class SectionQuestion(BaseModel):
    """Individual question for a report section"""
    scope: str = Field(description="Jurisdiction or document scope (RU_REG, EU_REG, etc.)")
    question: str = Field(description="Specific question to answer")
    answer_type: Literal["facts", "numbers", "table", "list", "boolean", "narrative_summary"] = Field(description="Expected answer type")
    doc_kind_preference: Optional[str] = Field(description="Preferred document type (EPAR, SmPC, GRLS, etc.)", default=None)

class SectionQuestions(BaseModel):
    """List of questions for a report section"""
    questions: List['SectionQueryPlanningPrompt.SectionQuestion'] = Field(description="List of questions for the section")
'''

    example = r"""
Example:
Input:
Section: Regulatory Status
INN: Rivaroxaban
Jurisdictions: RU, EU, US

Output:
{
    "questions": [
        {
            "scope": "RU_REG",
            "question": "What is the current GRLS registration status for Rivaroxaban?",
            "answer_type": "facts",
            "doc_kind_preference": "GRLS_PDF"
        },
        {
            "scope": "EU_REG",
            "question": "What does the EPAR state about indication and MAH for Rivaroxaban?",
            "answer_type": "facts",
            "doc_kind_preference": "EPAR_FULL"
        },
        {
            "scope": "US_REG",
            "question": "What is the FDA approval status and indication for Rivaroxaban?",
            "answer_type": "facts",
            "doc_kind_preference": "FDA_LABEL"
        }
    ]
}
"""

    user_prompt = "Section: {section_name}\nINN: {inn}\nJurisdictions: {jurisdictions}\nAdditional context: {context}"

    system_prompt = build_system_prompt(instruction, example)

    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)


class AnswerWithRAGContextSharedPrompt:
    instruction = """
You are a due diligence report assistant for pharmaceutical regulatory/clinical/patent/manufacturing documents.
Your task is to answer questions based only on the provided context from paged artifacts.

Do not reveal chain-of-thought. Provide short rationale (1–3 sentences) and evidence.
If evidence missing → put into unknowns; do not guess.

Key rules:
- Use only provided context (paged artifacts)
- Prefer primary documents: EPAR/SmPC/PIL/Approval letters/Instructions/GRLS/CTGov/CTIS/Patent pages
- Citations must include doc_id + page + snippet
- Every claim must have supporting evidence
"""

    user_prompt = """
Here is the context:
\"\"\"
{context}
\"\"\"

---

Question: {question}
Answer type: {answer_type}
Scope: {scope}
"""

class DDSectionAnswerPrompt:
    instruction = AnswerWithRAGContextSharedPrompt.instruction + """

Generate structured output for due diligence report section.
Answer based on the specified answer_type and scope.
Every claim, number, and risk must have supporting evidence_ids.
If evidence is missing, move to unknowns.
"""

    user_prompt = AnswerWithRAGContextSharedPrompt.user_prompt

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(DDSectionAnswerSchema), flags=re.MULTILINE)

    example = r"""
Example for Regulatory Status section:
Question: What is the current GRLS registration status?
Answer type: facts
Scope: RU_REG

Answer:
```json
{
  "section_id": "regulatory_status_ru",
  "claims": [
    {
      "id": "grls_status",
      "text": "Rivaroxaban is registered in Russia with marketing authorization",
      "evidence_ids": ["ev1"],
      "confidence": 0.95
    }
  ],
  "numbers": [],
  "risks": [],
  "unknowns": [],
  "evidence": [
    {
      "id": "ev1",
      "doc_id": "GRLS_2024",
      "doc_title": "State Register of Medicines",
      "page": 45,
      "snippet": "Rivaroxaban (Xarelto) - Marketing authorization granted 15.03.2021, registration number LP-005639"
    }
  ],
  "notes": "Status confirmed in latest GRLS update"
}
```
"""

    system_prompt = build_system_prompt(instruction, example)

    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)



# Legacy prompts removed - replaced with unified DDSectionAnswerPrompt

class ConflictResolverPrompt:
    instruction = """
You are a conflict resolution system for due diligence reports.
Your task is to analyze claims from different sources/jurisdictions and identify contradictions, conflicts, or inconsistencies.
If conflicts exist, create appropriate risk entries. Prioritize authoritative sources.

Authoritative hierarchy:
1. Primary regulatory documents (EPAR, FDA labels, SmPC)
2. Official registers (GRLS, Orange Book)
3. Secondary sources (CT.gov, company reports)
4. Tertiary sources (news, analysis)

Output conflicts as risks with severity levels.
"""

    user_prompt = """
Here are claims from different sources:
\"\"\"
{context}
\"\"\"

---

Analyze for conflicts and generate unified assessment.
"""

    class ConflictResolutionSchema(BaseModel):
        """Schema for conflict resolution output"""
        resolved_claims: List[Claim] = Field(description="Claims resolved from conflicts")
        conflicts: List[Risk] = Field(description="Identified conflicts as risks")
        unknowns: List[Unknown] = Field(description="Unresolvable conflicts moved to unknowns")
        evidence: List[Evidence] = Field(description="Supporting evidence for resolutions")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(ConflictResolutionSchema), flags=re.MULTILINE)

    example = r"""
Example:
Input claims from different sources about MAH for Rivaroxaban.

Output:
```json
{
  "resolved_claims": [
    {
      "id": "mah_resolved",
      "text": "Bayer AG is the Marketing Authorization Holder for Rivaroxaban",
      "evidence_ids": ["ev1", "ev2"],
      "confidence": 0.95
    }
  ],
  "conflicts": [
    {
      "id": "manufacturing_conflict",
      "title": "Manufacturing site discrepancy",
      "severity": "medium",
      "description": "EPAR lists Berlin site, but SmPC mentions Leverkusen",
      "evidence_ids": ["ev3", "ev4"]
    }
  ],
  "unknowns": [],
  "evidence": [...]
}
```
"""

    system_prompt = build_system_prompt(instruction, example)

    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)


class AnswerSchemaFixPrompt:
    system_prompt = """
You are a DD report JSON formatter.
Your task is to convert raw LLM responses into valid DDSectionAnswerSchema format.
Ensure all claims have evidence_ids, no orphaned claims, and proper evidence structure.

Rules:
- Move claims without evidence to unknowns
- Create evidence entries with doc_id/page/snippet format
- Validate against DDSectionAnswerSchema
- Output only valid JSON, no extra text
"""

    user_prompt = """
Schema definition:
{schema_definition}

Raw LLM response to fix:
{response}

Output only the corrected JSON object.
"""




class RerankingPrompt:
    system_prompt_rerank_single_block = """
You are a due diligence document reranker focused on evidence quality.

You will receive a query and retrieved text block. Score based on whether the block contains PROOF/EVIDENCE for the query, not just semantic similarity.

Instructions:

1. Reasoning:
   Analyze whether the block contains factual evidence that directly supports or answers the query. Focus on proof elements rather than general discussion.

2. Evidence Quality Score (0 to 1, in increments of 0.1):
   Boost scores for blocks containing:
   - Specific numbers, dates, names
   - Regulatory terms: "approved", "authorized", "MAH", "indication", "contraindication"
   - Structured data: tables, lists, specifications
   - Status terms: "Phase 3", "primary endpoint", "authorized", "registered"

   Penalize scores for:
   - Navigation content, headers, footers
   - General descriptions without facts
   - Duplicate or repetitive information
   - Introductory/background text

3. Scoring Scale:
   0 = No evidence: Block has no factual content relevant to proving the query
   0.3 = Weak evidence: Vague or indirect references
   0.6 = Moderate evidence: Some specific facts but incomplete
   0.9 = Strong evidence: Comprehensive factual support
   1 = Perfect evidence: Complete, authoritative proof
"""

    system_prompt_rerank_multiple_blocks = """
You are a due diligence document reranker focused on evidence quality for pharmaceutical queries.

You will receive a query and multiple text blocks. Score each based on evidence strength for regulatory/clinical/patent queries.

Instructions:

1. Reasoning:
   Evaluate each block's ability to serve as evidence for the query. Prioritize blocks with verifiable facts over general content.

2. Evidence Priority (highest to lowest):
   - Primary regulatory docs (EPAR, FDA labels, SmPC)
   - Official registers (GRLS, Orange Book)
   - Structured data (tables, specifications)
   - Specific terms (doses, indications, contraindications)
   - Dates, numbers, names with context

3. Scoring Scale (0-1):
   0 = No evidentiary value
   0.2 = Minimal factual content
   0.4 = Some relevant facts
   0.6 = Good supporting evidence
   0.8 = Strong evidentiary support
   1 = Authoritative, comprehensive evidence
"""

class RetrievalRankingSingleBlock(BaseModel):
    """Rank retrieved text block relevance to a query."""
    reasoning: str = Field(description="Analysis of the block, identifying key information and how it relates to the query")
    relevance_score: float = Field(description="Relevance score from 0 to 1, where 0 is Completely Irrelevant and 1 is Perfectly Relevant")

class RetrievalRankingMultipleBlocks(BaseModel):
    """Rank retrieved multiple text blocks relevance to a query."""
    block_rankings: List[RetrievalRankingSingleBlock] = Field(
        description="A list of text blocks and their associated relevance scores."
    )
