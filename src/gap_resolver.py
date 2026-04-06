"""
Gap Resolver — Sprint 22 WS1
==============================
Reads dossier.json + unknowns + coverage_ledger + field_source_map.yaml +
source_contracts.json and builds an AcquisitionPlan for source completion.

Also provides controlled execution layer for supported official sources
(attach → fetch → index → check stop condition).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Config paths ────────────────────────────────────────────────────────────

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_FIELD_SOURCE_MAP_PATH = _CONFIG_DIR / "field_source_map.yaml"
_CONTRACTS_PATH = _CONFIG_DIR / "source_contracts.json"

# ── Stop condition evaluators ───────────────────────────────────────────────

_STOP_EVALUATORS = {
    "field_has_value": lambda v: v is not None and v != "" and v != [],
    "found_verified_primary": lambda v: v is not None and v != "" and v != [],
    "all_regions_covered": lambda v: isinstance(v, list) and len(v) >= 2,
    "study_card_assembled": lambda v: isinstance(v, dict) and bool(v.get("title")),
    "family_identified": lambda v: v is not None and v != "",
    "synthesis_extracted": lambda v: isinstance(v, list) and len(v) >= 1,
}


# ── Models ──────────────────────────────────────────────────────────────────

class AcquisitionItem(BaseModel):
    """Single gap→source action item."""
    field_path: str
    section: str
    reason: str
    target_source: str
    source_tier: str  # tier1 | tier2 | fallback
    source_class: str  # primary_official | registry | secondary | manual
    mode: str  # attach_missing | retry_fetch | retry_index | refresh_stale | backfill_secondary | no_action
    priority: int
    stop_condition: str
    current_state: str  # filled | empty | partial | not_applicable
    allow_provisional: bool
    supports_execution: bool
    jurisdiction: Optional[str] = None
    context_id: Optional[str] = None


class AcquisitionPlan(BaseModel):
    """Full gap resolution plan for a case."""
    case_id: str
    generated_at: str
    inn: Optional[str] = None
    items: List[AcquisitionItem] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _load_field_source_map(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or _FIELD_SOURCE_MAP_PATH
    if not p.exists():
        logger.warning("field_source_map.yaml not found at %s", p)
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_contracts(path: Optional[Path] = None) -> Dict[str, Dict]:
    p = path or _CONTRACTS_PATH
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {c["source_id"]: c for c in data.get("contracts", [])}


def _section_from_field(field_path: str) -> str:
    """Extract section name from field path."""
    first = field_path.split(".")[0]
    # Normalize array-pattern fields
    first = re.sub(r"\[\*\]$", "", first)
    return first


def _jurisdiction_from_field(field_path: str) -> Optional[str]:
    """Extract jurisdiction from field path like registrations.EU.status."""
    parts = field_path.split(".")
    if len(parts) >= 2 and parts[0] == "registrations":
        region = parts[1]
        if region in ("US", "EU", "RU", "EAEU"):
            return region
    return None


def _resolve_dossier_field(dossier: Dict[str, Any], field_path: str) -> Any:
    """
    Navigate dossier dict using dot-notation field_path.
    Handles array patterns like clinical_studies[*].phase.
    Returns the resolved value (or None if not found).
    """
    parts = field_path.split(".")
    current = dossier

    for i, part in enumerate(parts):
        if current is None:
            return None

        # Handle array wildcard: clinical_studies[*]
        if part.endswith("[*]"):
            key = part[:-3]
            arr = current.get(key, [])
            if not isinstance(arr, list) or not arr:
                return None
            # For array fields, collect the sub-field from all items
            remaining = ".".join(parts[i + 1:])
            if remaining:
                values = []
                for item in arr:
                    if isinstance(item, dict):
                        v = _resolve_dossier_field(item, remaining)
                        if v is not None:
                            values.append(v)
                return values if values else None
            return arr

        # Handle region-qualified registrations: registrations.EU.status
        if isinstance(current, list):
            # Search in list items (e.g., registrations is a list)
            for item in current:
                if isinstance(item, dict):
                    if item.get("region") == part:
                        current = item
                        break
            else:
                return None
            continue

        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None

    # Unwrap EvidencedValue: {value: ..., evidence_refs: [...]}
    if isinstance(current, dict) and "value" in current:
        return current.get("value")
    return current


def _classify_field_state(value: Any) -> str:
    """Classify a field value into state categories."""
    if value is None:
        return "empty"
    if isinstance(value, list):
        if not value:
            return "empty"
        # Check if list items have actual values
        real_values = [v for v in value if v is not None and v != ""]
        if not real_values:
            return "empty"
        return "filled"
    if isinstance(value, str) and not value.strip():
        return "empty"
    if isinstance(value, dict):
        if not value:
            return "empty"
        # Check if it's a mostly-empty dict
        non_null = sum(1 for v in value.values() if v is not None and v != "" and v != [])
        if non_null == 0:
            return "empty"
        if non_null < len(value) // 2:
            return "partial"
    return "filled"


def _source_status_from_ledger(
    source_id: str,
    coverage_ledger: Optional[Dict[str, Any]],
    docs_by_source: Dict[str, List[Dict]],
) -> str:
    """
    Determine source status from coverage ledger / documents.
    Returns: not_declared | not_attached | attached_not_fetched |
             fetched_not_indexed | indexed_but_not_evidenced | evidenced | unsupported
    """
    # Check source_breakdown in coverage ledger
    if coverage_ledger:
        for sb in coverage_ledger.get("source_breakdown", []):
            if sb.get("source_id") == source_id:
                status = sb.get("status", "not_declared")
                if status in ("unsupported", "not_reachable"):
                    return "unsupported"
                verdict = sb.get("verdict", "")
                if verdict == "PARK":
                    return "unsupported"
                # Check document-level status
                attached = sb.get("attached_docs", 0)
                indexed = sb.get("indexed_docs", 0)
                evidenced = sb.get("evidenced_docs", 0)
                if evidenced > 0:
                    return "evidenced"
                if indexed > 0:
                    return "indexed_but_not_evidenced"
                if attached > 0:
                    return "attached_not_fetched"  # or fetched_not_indexed
                return "not_attached"

    # Fallback: check documents directly
    docs = docs_by_source.get(source_id, [])
    if not docs:
        return "not_attached"

    statuses = {d.get("status", "unknown") for d in docs}
    if "indexed" in statuses or "parsed" in statuses:
        return "indexed_but_not_evidenced"
    if "rendered" in statuses:
        return "fetched_not_indexed"
    if "created" in statuses or "rendering" in statuses:
        return "attached_not_fetched"
    if statuses & {"failed", "blocked_paywall", "forbidden_403", "timeout"}:
        return "attached_not_fetched"
    return "not_attached"


def _source_class_from_contracts(
    source_id: str, contracts: Dict[str, Dict]
) -> str:
    """Get source_class from contracts."""
    c = contracts.get(source_id, {})
    role = c.get("source_role", "")
    if role == "primary":
        return "primary_official"
    if role == "enrichment":
        return "secondary"
    tier = c.get("tier", "")
    if tier == "tier1":
        return "primary_official"
    if tier == "tier2":
        return "registry"
    return "secondary"


def _determine_mode(
    source_status: str,
    is_fallback: bool,
) -> str:
    """Pick the right acquisition mode given current source status."""
    if source_status == "not_attached" or source_status == "not_declared":
        return "attach_missing"
    if source_status == "attached_not_fetched":
        return "retry_fetch"
    if source_status == "fetched_not_indexed":
        return "retry_index"
    if source_status == "indexed_but_not_evidenced":
        return "backfill_secondary" if is_fallback else "refresh_stale"
    if source_status == "evidenced":
        return "no_action"
    if source_status == "unsupported":
        return "no_action"
    return "attach_missing"


# ── Main resolver ───────────────────────────────────────────────────────────

class GapResolver:
    """
    Reads dossier + coverage_ledger + field_source_map and builds
    an acquisition plan for closing coverage gaps.
    """

    def __init__(
        self,
        field_source_map_path: Optional[Path] = None,
        contracts_path: Optional[Path] = None,
    ):
        raw = _load_field_source_map(field_source_map_path)
        self.field_map: Dict[str, Dict] = raw.get("field_source_map", {})
        self.source_doc_kinds: Dict[str, List[str]] = raw.get("source_doc_kinds", {})
        self.safe_sources: set = set(raw.get("safe_execution_sources", []))
        self.contracts = _load_contracts(contracts_path)

    def plan(
        self,
        dossier: Dict[str, Any],
        case_id: str,
        coverage_ledger: Optional[Dict[str, Any]] = None,
        db_documents: Optional[List[Dict[str, Any]]] = None,
        unknowns: Optional[List[Dict[str, Any]]] = None,
    ) -> AcquisitionPlan:
        """
        Build acquisition plan from dossier gaps.

        Args:
            dossier: Dossier v3 JSON dict.
            case_id: Case UUID.
            coverage_ledger: Output of CoverageLedgerBuilder.build().
            db_documents: All documents for this case (from DB).
            unknowns: Explicit unknowns list (or taken from dossier).
        """
        if unknowns is None:
            unknowns = dossier.get("unknowns", [])

        # Build source→docs index from db_documents
        docs_by_source = defaultdict(list)
        if db_documents:
            for d in db_documents:
                # Map doc_kind to source_id (reverse lookup)
                dk = d.get("doc_kind", "")
                for sid, kinds in self.source_doc_kinds.items():
                    if dk in kinds:
                        docs_by_source[sid].append(d)
                        break

        # Build unknown fields set for priority boosting
        unknown_fields = {u.get("field_path", "") for u in unknowns}

        items: List[AcquisitionItem] = []
        priority_counter = 0

        for field_path, cfg in self.field_map.items():
            preferred = cfg.get("preferred_chain", [])
            fallback = cfg.get("fallback_chain", [])
            stop_cond = cfg.get("stop_condition", "field_has_value")
            allow_prov = cfg.get("allow_provisional", False)
            supports_exec = cfg.get("supports_execution", False)

            section = _section_from_field(field_path)
            jurisdiction = _jurisdiction_from_field(field_path)

            # Check current field state
            value = _resolve_dossier_field(dossier, field_path)
            state = _classify_field_state(value)

            # Evaluate stop condition
            evaluator = _STOP_EVALUATORS.get(stop_cond, _STOP_EVALUATORS["field_has_value"])
            stop_met = evaluator(value)

            if state == "filled" and stop_met:
                continue  # Already satisfied, skip

            # Check if there's an explicit unknown for this field
            is_explicit_unknown = field_path in unknown_fields

            # Walk preferred chain
            chain_closed = False
            for source_id in preferred:
                src_status = _source_status_from_ledger(
                    source_id, coverage_ledger, docs_by_source
                )

                if src_status == "evidenced":
                    # Primary source already evidenced — stop chain (AC-WS1-2)
                    chain_closed = True
                    break

                if src_status == "unsupported":
                    continue

                mode = _determine_mode(src_status, is_fallback=False)
                if mode == "no_action":
                    continue

                priority_counter += 1
                # Boost priority for explicit unknowns
                priority = priority_counter
                if is_explicit_unknown:
                    priority = max(1, priority - 50)

                src_class = _source_class_from_contracts(source_id, self.contracts)
                can_execute = supports_exec and (source_id in self.safe_sources)

                reason = self._build_reason(field_path, state, source_id, src_status, is_explicit_unknown)

                items.append(AcquisitionItem(
                    field_path=field_path,
                    section=section,
                    reason=reason,
                    target_source=source_id,
                    source_tier="tier1",
                    source_class=src_class,
                    mode=mode,
                    priority=priority,
                    stop_condition=stop_cond,
                    current_state=state,
                    allow_provisional=allow_prov,
                    supports_execution=can_execute,
                    jurisdiction=jurisdiction,
                ))
                break  # Only first actionable source in preferred chain

            if chain_closed:
                continue

            # Fallback chain — only if preferred chain exhausted
            if not any(i.field_path == field_path for i in items):
                for source_id in fallback:
                    src_status = _source_status_from_ledger(
                        source_id, coverage_ledger, docs_by_source
                    )
                    if src_status in ("evidenced", "unsupported"):
                        continue

                    mode = _determine_mode(src_status, is_fallback=True)
                    if mode == "no_action":
                        continue

                    priority_counter += 1
                    src_class = _source_class_from_contracts(source_id, self.contracts)
                    can_execute = supports_exec and (source_id in self.safe_sources)

                    reason = (
                        f"Preferred sources exhausted for {field_path} (state={state}). "
                        f"Fallback: {source_id} ({src_status}). PROVISIONAL."
                    )

                    items.append(AcquisitionItem(
                        field_path=field_path,
                        section=section,
                        reason=reason,
                        target_source=source_id,
                        source_tier="fallback",
                        source_class=src_class,
                        mode=mode,
                        priority=priority_counter,
                        stop_condition=stop_cond,
                        current_state=state,
                        allow_provisional=True,
                        supports_execution=can_execute,
                        jurisdiction=jurisdiction,
                    ))
                    break

        # Sort by priority (lower = higher priority)
        items.sort(key=lambda x: x.priority)
        # Re-number sequentially
        for i, item in enumerate(items, 1):
            item.priority = i

        summary = self._build_summary(items)

        # Extract INN from dossier for URL templates
        inn = None
        passport = dossier.get("passport", {})
        inn_raw = passport.get("inn")
        if isinstance(inn_raw, dict):
            inn = inn_raw.get("value")
        elif isinstance(inn_raw, str):
            inn = inn_raw

        return AcquisitionPlan(
            case_id=case_id,
            generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            inn=inn,
            items=items,
            summary=summary,
        )

    def _build_reason(
        self, field_path: str, state: str, source_id: str,
        src_status: str, is_unknown: bool,
    ) -> str:
        """Build human-readable reason for an acquisition item."""
        parts = []
        if is_unknown:
            parts.append(f"Explicit unknown: {field_path}")
        else:
            parts.append(f"Field {field_path} is {state}")
        parts.append(f"Next source: {source_id} (status={src_status})")
        return ". ".join(parts)

    def _build_summary(self, items: List[AcquisitionItem]) -> Dict[str, Any]:
        """Build summary statistics from plan items."""
        by_section = defaultdict(int)
        by_mode = defaultdict(int)
        by_tier = defaultdict(int)
        executable = 0
        provisional = 0

        for item in items:
            by_section[item.section] += 1
            by_mode[item.mode] += 1
            by_tier[item.source_tier] += 1
            if item.supports_execution:
                executable += 1
            if item.source_tier == "fallback":
                provisional += 1

        return {
            "total_items": len(items),
            "by_section": dict(by_section),
            "by_mode": dict(by_mode),
            "by_tier": dict(by_tier),
            "executable_items": executable,
            "provisional_items": provisional,
        }


# ── Controlled Execution Layer (WS1-D3) ────────────────────────────────────

class ExecutionLog(BaseModel):
    """Log entry for a single execution step."""
    item_index: int
    field_path: str
    target_source: str
    action: str  # attach | fetch | index | skip
    outcome: str  # success | failed | skipped | stop_condition_met
    details: str = ""
    timestamp: str = ""


class ExecutionResult(BaseModel):
    """Result of controlled execution run."""
    case_id: str
    executed_items: int
    successful: int
    failed: int
    skipped: int
    stop_conditions_met: int
    log: List[ExecutionLog] = Field(default_factory=list)


class ControlledExecutor:
    """
    Executes acquisition plan items for supported official sources.

    Requires:
      - gateway_base_url: API gateway URL (e.g. http://localhost:8085)
      - auth_token: Bearer token for gateway API calls
      - tenant_id: Tenant ID for case operations

    Only executes items where:
      - supports_execution=True
      - source is in safe_execution_sources
      - source is NOT secondary web fallback
    """

    def __init__(
        self,
        gateway_base_url: str = "http://localhost:8085",
        auth_token: str = "",
        tenant_id: str = "demo",
    ):
        self.gateway = gateway_base_url.rstrip("/")
        self.auth_token = auth_token
        self.tenant_id = tenant_id
        self._current_inn: Optional[str] = None

    def execute(
        self,
        plan: AcquisitionPlan,
        max_items: int = 10,
        dry_run: bool = False,
    ) -> ExecutionResult:
        """
        Execute priority-1 items from acquisition plan.

        Args:
            plan: The acquisition plan from GapResolver.plan().
            max_items: Max items to execute in one pass.
            dry_run: If True, log but don't actually call APIs.
        """
        self._current_inn = plan.inn

        result = ExecutionResult(
            case_id=plan.case_id,
            executed_items=0,
            successful=0,
            failed=0,
            skipped=0,
            stop_conditions_met=0,
        )

        for i, item in enumerate(plan.items[:max_items]):
            if not item.supports_execution:
                entry = ExecutionLog(
                    item_index=i,
                    field_path=item.field_path,
                    target_source=item.target_source,
                    action="skip",
                    outcome="skipped",
                    details=f"Source {item.target_source} does not support auto-execution",
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                )
                result.log.append(entry)
                result.skipped += 1
                continue

            if item.mode == "no_action":
                continue

            result.executed_items += 1

            if dry_run:
                entry = ExecutionLog(
                    item_index=i,
                    field_path=item.field_path,
                    target_source=item.target_source,
                    action=item.mode,
                    outcome="skipped",
                    details="DRY RUN — would execute: " + item.mode,
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                )
                result.log.append(entry)
                result.skipped += 1
                continue

            # Actual execution
            try:
                outcome = self._execute_item(item, plan.case_id)
                entry = ExecutionLog(
                    item_index=i,
                    field_path=item.field_path,
                    target_source=item.target_source,
                    action=item.mode,
                    outcome=outcome["status"],
                    details=outcome.get("details", ""),
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                )
                result.log.append(entry)
                if outcome["status"] == "success":
                    result.successful += 1
                elif outcome["status"] == "stop_condition_met":
                    result.stop_conditions_met += 1
                    result.successful += 1
                else:
                    result.failed += 1
            except Exception as e:
                logger.error("Execution failed for item %d (%s): %s", i, item.field_path, e)
                entry = ExecutionLog(
                    item_index=i,
                    field_path=item.field_path,
                    target_source=item.target_source,
                    action=item.mode,
                    outcome="failed",
                    details=str(e),
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                )
                result.log.append(entry)
                result.failed += 1

        return result

    def _execute_item(self, item: AcquisitionItem, case_id: str) -> Dict[str, str]:
        """Execute a single acquisition item via gateway API."""
        import requests

        if item.mode == "attach_missing":
            return self._attach_source(item, case_id)
        elif item.mode == "retry_fetch":
            return self._retry_fetch(item, case_id)
        elif item.mode == "retry_index":
            return self._retry_index(item, case_id)
        elif item.mode == "refresh_stale":
            return self._attach_source(item, case_id)
        else:
            return {"status": "skipped", "details": f"Unsupported mode: {item.mode}"}

    # URL templates for known API sources. {inn} is replaced at runtime.
    _SOURCE_URL_TEMPLATES: Dict[str, Dict[str, str]] = {
        "openfda": {
            "us_fda": "https://api.fda.gov/drug/label.json?search=openfda.generic_name:{inn}&limit=5",
            "label": "https://api.fda.gov/drug/label.json?search=openfda.generic_name:{inn}&limit=5",
        },
        "pubchem": {
            "pubchem": "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{inn}/JSON",
        },
        "ctgov": {
            "ctgov": "https://clinicaltrials.gov/api/v2/studies?query.term={inn}&pageSize=20",
        },
        "ctgov_results": {
            "ctgov_results": "https://clinicaltrials.gov/api/v2/studies?query.term={inn}&filter.overallStatus=COMPLETED&pageSize=10",
        },
        "ctgov_protocol": {
            "ctgov_protocol": "https://clinicaltrials.gov/api/v2/studies?query.term={inn}&pageSize=10",
        },
        "chembl": {
            "chembl": "https://www.ebi.ac.uk/chembl/api/data/molecule/search?q={inn}&format=json",
        },
    }

    def _resolve_source_url(self, source_id: str, doc_kind: str, inn: Optional[str]) -> Optional[str]:
        """Resolve a source-specific URL from templates."""
        templates = self._SOURCE_URL_TEMPLATES.get(source_id, {})
        tmpl = templates.get(doc_kind)
        if not tmpl:
            return None
        if not inn:
            return None
        return tmpl.replace("{inn}", inn)

    def _attach_source(self, item: AcquisitionItem, case_id: str) -> Dict[str, str]:
        """Attach a missing source via POST /cases/{id}/sources:attach."""
        import requests

        doc_kinds = self.source_doc_kinds_for(item.target_source)
        if not doc_kinds:
            return {"status": "failed", "details": f"No doc_kinds mapped for {item.target_source}"}

        sources = []
        for dk in doc_kinds:
            url = self._resolve_source_url(item.target_source, dk, self._current_inn)
            if not url:
                logger.warning("No URL template for %s/%s (inn=%s) — skipping", item.target_source, dk, self._current_inn)
                continue
            sources.append({
                "url": url,
                "doc_kind": dk,
                "title": f"Auto-attached by WS1 gap resolver: {item.target_source}/{dk}",
                "region": item.jurisdiction or "",
                "retrieval_reason": f"unknown:{item.field_path}",
                "origin_channel": "registry_api",
                "source_class": item.source_class,
                "verification_status": (
                    "provisional_secondary" if item.source_tier == "fallback"
                    else "verified_primary"
                ),
            })

        if not sources:
            return {"status": "failed", "details": f"No resolvable URLs for {item.target_source} (inn={self._current_inn})"}

        api_url = f"{self.gateway}/ddkit/cases/{case_id}/sources:attach"
        headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json",
        }

        try:
            resp = requests.post(api_url, json={"sources": sources}, headers=headers, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                attached = sum(1 for r in results if r.get("document_id") and not r.get("error"))
                errors = [r.get("error") for r in results if r.get("error")]
                if errors:
                    return {"status": "partial" if attached > 0 else "failed",
                            "details": f"Attached {attached}, errors: {'; '.join(errors[:3])}"}
                return {"status": "success", "details": f"Attached {attached} docs for {item.target_source}"}
            else:
                return {"status": "failed", "details": f"HTTP {resp.status_code}: {resp.text[:200]}"}
        except Exception as e:
            return {"status": "failed", "details": str(e)}

    def _retry_fetch(self, item: AcquisitionItem, case_id: str) -> Dict[str, str]:
        """Retry fetch for a failed document — re-enqueue doc_fetch_render."""
        # In V1, retry = re-attach (which will re-enqueue if not duplicate)
        return self._attach_source(item, case_id)

    def _retry_index(self, item: AcquisitionItem, case_id: str) -> Dict[str, str]:
        """Retry index — currently same as attach (re-triggers pipeline)."""
        return self._attach_source(item, case_id)

    def source_doc_kinds_for(self, source_id: str) -> List[str]:
        """Get doc_kinds for a source_id from loaded config."""
        raw = _load_field_source_map()
        return raw.get("source_doc_kinds", {}).get(source_id, [])


# ── CLI / standalone runner ─────────────────────────────────────────────────

def run_plan_from_file(
    dossier_path: str,
    case_id: str,
    coverage_ledger_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> AcquisitionPlan:
    """
    Run gap resolver from file paths (for CLI / testing).

    Usage:
        python -m gap_resolver plan --dossier path/to/dossier.json --case-id UUID
    """
    with open(dossier_path, "r", encoding="utf-8") as f:
        dossier = json.load(f)

    coverage_ledger = None
    if coverage_ledger_path:
        with open(coverage_ledger_path, "r", encoding="utf-8") as f:
            coverage_ledger = json.load(f)

    resolver = GapResolver()
    plan = resolver.plan(
        dossier=dossier,
        case_id=case_id,
        coverage_ledger=coverage_ledger,
    )

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(plan.model_dump(), f, indent=2, ensure_ascii=False)
        logger.info("Acquisition plan written to %s (%d items)", output_path, len(plan.items))
    else:
        print(json.dumps(plan.model_dump(), indent=2, ensure_ascii=False))

    return plan


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="WS1 Gap Resolver")
    sub = parser.add_subparsers(dest="command")

    plan_cmd = sub.add_parser("plan", help="Build acquisition plan from dossier")
    plan_cmd.add_argument("--dossier", required=True, help="Path to dossier_v3.json")
    plan_cmd.add_argument("--case-id", required=True, help="Case UUID")
    plan_cmd.add_argument("--coverage-ledger", help="Path to coverage_ledger.json")
    plan_cmd.add_argument("--output", help="Output path for acquisition_plan.json")

    exec_cmd = sub.add_parser("execute", help="Execute acquisition plan")
    exec_cmd.add_argument("--plan", required=True, help="Path to acquisition_plan.json")
    exec_cmd.add_argument("--gateway", default="http://localhost:8085")
    exec_cmd.add_argument("--token", default="demo_token_9f83b2a1")
    exec_cmd.add_argument("--max-items", type=int, default=10)
    exec_cmd.add_argument("--dry-run", action="store_true")
    exec_cmd.add_argument("--output", help="Output path for execution_result.json")

    args = parser.parse_args()

    if args.command == "plan":
        run_plan_from_file(
            dossier_path=args.dossier,
            case_id=args.case_id,
            coverage_ledger_path=args.coverage_ledger,
            output_path=args.output,
        )
    elif args.command == "execute":
        with open(args.plan, "r", encoding="utf-8") as f:
            plan_data = json.load(f)
        plan = AcquisitionPlan(**plan_data)
        executor = ControlledExecutor(
            gateway_base_url=args.gateway,
            auth_token=args.token,
        )
        result = executor.execute(plan, max_items=args.max_items, dry_run=args.dry_run)
        out = result.model_dump()
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
        else:
            print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        parser.print_help()
