"""
Coverage Ledger Builder — Sprint 9
====================================
Computes a source-universe-aware coverage ledger for a dossier run.

Separates:
  - quality_v2 = quality of the output artifact (Sprint 7.5)
  - coverage_ledger = completeness relative to the declared source universe

Stages tracked per source:
  declared → reachable → attached → fetched → indexed → extracted → evidenced

Decision readiness per section:
  ready | partial | insufficient
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

# ── Config paths ─────────────────────────────────────────────────────────────

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_UNIVERSE_PATH = _CONFIG_DIR / "declared_source_universe.yaml"
_CONTRACTS_PATH = _CONFIG_DIR / "source_contracts.json"


# ── Pydantic-free lightweight models (dict-based for JSON serialization) ─────

def _load_universe(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or _UNIVERSE_PATH
    if not p.exists():
        logger.warning("declared_source_universe.yaml not found at %s", p)
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_contracts(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    p = path or _CONTRACTS_PATH
    if not p.exists():
        logger.warning("source_contracts.json not found at %s", p)
        return []
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("contracts", [])


# ── Source status determination ──────────────────────────────────────────────

# Terminal document statuses (from job_processors.py:210-214)
_TERMINAL_STATUSES = frozenset({
    "indexed", "failed", "parsed", "skipped", "unsupported",
    "blocked_paywall", "captcha", "forbidden_403", "rate_limited_429",
    "requires_login", "robots_denied", "timeout", "parsed_empty",
})

_FAILURE_STATUSES = frozenset({
    "failed", "blocked_paywall", "captcha", "forbidden_403",
    "rate_limited_429", "requires_login", "robots_denied", "timeout",
})

_INDEXED_STATUSES = frozenset({"indexed", "parsed"})


def _source_status(docs: List[Dict[str, Any]]) -> str:
    """Determine source-level status from its documents."""
    if not docs:
        return "not_attached"
    statuses = {d.get("status", "unknown") for d in docs}
    if statuses & _INDEXED_STATUSES:
        return "ok"
    if statuses & _FAILURE_STATUSES:
        # Pick most informative failure
        for s in ["blocked_paywall", "captcha", "forbidden_403", "rate_limited_429",
                   "requires_login", "robots_denied", "timeout", "failed"]:
            if s in statuses:
                return s
    if "rendered" in statuses or "created" in statuses:
        return "in_progress"
    return "unknown"


# ── Main builder ─────────────────────────────────────────────────────────────

class CoverageLedgerBuilder:
    """
    Builds coverage_ledger block for a dossier/report run.

    Usage:
        builder = CoverageLedgerBuilder(use_case="ra_regulatory_screening")
        ledger = builder.build(
            db_documents=<list of doc dicts from DB>,
            dossier_report=<DossierReport dict or None>,
            evidence_registry=<list of evidence dicts or None>,
        )
    """

    def __init__(
        self,
        use_case: str = "ra_regulatory_screening",
        universe_path: Optional[Path] = None,
        contracts_path: Optional[Path] = None,
    ):
        self.use_case = use_case
        self.universe = _load_universe(universe_path)
        self.contracts = {c["source_id"]: c for c in _load_contracts(contracts_path)}
        self._uc_config = self._resolve_use_case()

    def _resolve_use_case(self) -> Dict[str, Any]:
        """Get the use-case config from universe."""
        uc = self.universe.get("use_cases", {}).get(self.use_case, {})
        if not uc:
            logger.warning("Use case '%s' not found in declared_source_universe", self.use_case)
        return uc

    def build(
        self,
        db_documents: List[Dict[str, Any]],
        dossier_report: Optional[Dict[str, Any]] = None,
        evidence_registry: Optional[List[Dict[str, Any]]] = None,
        wave2_documents: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Build the full coverage ledger.

        Args:
            db_documents: All documents for this case from PostgreSQL (id, doc_kind, status, source_url, ...).
            dossier_report: The dossier v3 report dict (for field-level extraction checks).
            evidence_registry: List of evidence dicts from the report.
            wave2_documents: Subset of db_documents that came from Wave 2 (for delta accounting).
        """
        sections = self._uc_config.get("sections", {})
        section_coverage = {}
        all_source_breakdown = []
        total_declared = 0
        total_reachable = 0
        total_attached = 0
        total_indexed = 0
        total_fields_expected = 0
        total_fields_filled = 0
        total_fields_evidenced = 0
        total_critical_unknowns = 0

        # Index documents by doc_kind for quick lookup
        docs_by_kind = defaultdict(list)
        for d in db_documents:
            dk = d.get("doc_kind", "unknown")
            docs_by_kind[dk].append(d)

        # Evidence set (for checking if a field has evidence)
        ev_set = set()
        if evidence_registry:
            for ev in evidence_registry:
                ev_set.add(ev.get("evidence_id") or ev.get("id", ""))

        for section_id, section_cfg in sections.items():
            sc = self._build_section_coverage(
                section_id, section_cfg, docs_by_kind, dossier_report, ev_set
            )
            section_coverage[section_id] = sc
            total_declared += sc["declared_sources"]
            total_reachable += sc["reachable_sources"]
            total_attached += sc["attached_docs"]
            total_indexed += sc["indexed_docs"]
            total_fields_expected += sc["fields_expected"]
            total_fields_filled += sc["fields_filled"]
            total_fields_evidenced += sc["fields_evidenced"]
            total_critical_unknowns += sc["critical_unknowns"]
            all_source_breakdown.extend(sc.get("source_breakdown", []))

        # Wave 2 delta
        wave2_delta = self._build_wave2_delta(
            db_documents, wave2_documents, dossier_report, ev_set
        )

        # Overall decision readiness
        overall_readiness = self._compute_overall_readiness(section_coverage)

        ledger = {
            "basis": f"declared_source_universe_v{self.universe.get('version', '1.0')}",
            "use_case": self.use_case,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "totals": {
                "declared_sources": total_declared,
                "reachable_sources": total_reachable,
                "attached_docs": total_attached,
                "indexed_docs": total_indexed,
                "fields_expected": total_fields_expected,
                "fields_filled": total_fields_filled,
                "fields_evidenced": total_fields_evidenced,
                "critical_unknowns": total_critical_unknowns,
                "decision_readiness": overall_readiness,
            },
            "section_coverage": section_coverage,
            "source_breakdown": all_source_breakdown,
            "wave2_delta": wave2_delta,
            "limitations": self._build_limitations(section_coverage, all_source_breakdown),
        }
        return ledger

    def _build_section_coverage(
        self,
        section_id: str,
        section_cfg: Dict[str, Any],
        docs_by_kind: Dict[str, List[Dict]],
        dossier_report: Optional[Dict[str, Any]],
        ev_set: set,
    ) -> Dict[str, Any]:
        """Build coverage for a single section."""
        declared_sources = 0
        reachable_sources = 0
        attached_docs = 0
        indexed_docs = 0
        source_breakdown = []

        # Collect all sources from regions or flat lists
        all_sources = self._collect_section_sources(section_cfg)

        for src in all_sources:
            src_id = src.get("source_id", "unknown")
            contract = self.contracts.get(src_id, {})
            expected_kinds = src.get("doc_kinds", contract.get("doc_kinds_expected", []))
            tier = src.get("tier", contract.get("tier", "unknown"))

            declared_sources += 1

            # Check if source is unsupported
            unsupported_list = self._get_unsupported(section_cfg, src_id)
            if unsupported_list:
                source_breakdown.append({
                    "source_id": src_id,
                    "tier": tier,
                    "status": "unsupported",
                    "reason": unsupported_list[0].get("reason", "Not implemented"),
                    "doc_kinds_expected": len(expected_kinds),
                    "doc_kinds_attached": 0,
                    "docs_indexed": 0,
                    "fields_contributed": 0,
                    "critical_missing": contract.get("must_have_fields", []),
                })
                continue

            reachable_sources += 1

            # Count attached and indexed docs matching expected kinds
            src_docs = []
            for ek in expected_kinds:
                src_docs.extend(docs_by_kind.get(ek, []))

            src_attached = len(src_docs)
            src_indexed = sum(1 for d in src_docs if d.get("status") in _INDEXED_STATUSES)
            attached_docs += src_attached
            indexed_docs += src_indexed

            status = _source_status(src_docs)

            # Check contract fields
            must_fields = contract.get("must_have_fields", [])
            fields_contributed = 0
            critical_missing_fields = []
            if dossier_report and must_fields:
                for fpath in must_fields:
                    val = self._resolve_field(dossier_report, fpath)
                    if val is not None:
                        fields_contributed += 1
                    else:
                        critical_missing_fields.append(fpath)

            source_breakdown.append({
                "source_id": src_id,
                "tier": tier,
                "status": status,
                "doc_kinds_expected": len(expected_kinds),
                "doc_kinds_attached": len(set(d.get("doc_kind") for d in src_docs)),
                "docs_attached": src_attached,
                "docs_indexed": src_indexed,
                "fields_contributed": fields_contributed,
                "critical_missing": critical_missing_fields,
                "known_limitations": contract.get("known_limitations", []),
            })

        # Field-level coverage from dossier
        fields_expected, fields_filled, fields_evidenced = self._count_section_fields(
            section_id, dossier_report, ev_set
        )

        critical_unknowns = self._count_critical_unknowns(section_id, dossier_report)

        decision_readiness = self._section_readiness(
            indexed_docs, declared_sources, fields_filled, fields_expected, critical_unknowns
        )

        return {
            "declared_sources": declared_sources,
            "reachable_sources": reachable_sources,
            "attached_docs": attached_docs,
            "indexed_docs": indexed_docs,
            "fields_expected": fields_expected,
            "fields_filled": fields_filled,
            "fields_evidenced": fields_evidenced,
            "critical_unknowns": critical_unknowns,
            "decision_readiness": decision_readiness,
            "source_breakdown": source_breakdown,
        }

    def _collect_section_sources(self, section_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect all tier1 + tier2 sources from a section config (with or without regions)."""
        sources = []
        regions = section_cfg.get("regions", {})
        if regions:
            for region_id, region_cfg in regions.items():
                for src in region_cfg.get("tier1_sources", []):
                    sources.append({**src, "tier": "tier1", "region": region_id})
                for src in region_cfg.get("tier2_sources", []):
                    sources.append({**src, "tier": "tier2", "region": region_id})
        else:
            for src in section_cfg.get("tier1_sources", []):
                sources.append({**src, "tier": "tier1"})
            for src in section_cfg.get("tier2_sources", []):
                sources.append({**src, "tier": "tier2"})
        return sources

    def _get_unsupported(self, section_cfg: Dict[str, Any], source_id: str) -> List[Dict]:
        """Check if source is in unsupported list."""
        result = []
        # Check top-level unsupported
        for u in section_cfg.get("unsupported", []):
            if u.get("source_id") == source_id:
                result.append(u)
        # Check per-region unsupported
        for region_cfg in section_cfg.get("regions", {}).values():
            for u in region_cfg.get("unsupported", []):
                if u.get("source_id") == source_id:
                    result.append(u)
        return result

    # ── Field-level counting ─────────────────────────────────────────────────

    _SECTION_FIELD_MAP = {
        "registrations": [
            "passport.fda_approval_date", "passport.fda_indication",
            "passport.route_of_administration", "passport.dosage_forms",
            "passport.chemical_formula", "passport.drug_class",
            "passport.mechanism_of_action", "passport.mah_holders",
            "passport.trade_names", "passport.registered_where",
            "passport.key_dosages",
            "registrations.US.status", "registrations.US.mah",
            "registrations.EU.status", "registrations.EU.mah",
            "registrations.RU.status", "registrations.RU.mah",
        ],
        "clinical": [
            "clinical_studies[].study_id", "clinical_studies[].phase",
            "clinical_studies[].status", "clinical_studies[].n_enrolled",
            "clinical_studies[].efficacy_keypoints",
        ],
        "patents": [
            "patent_families[].representative_pub", "patent_families[].priority_date",
            "patent_families[].assignees", "patent_families[].expiry_by_country",
            "patent_families[].what_blocks",
        ],
        "safety": [
            "passport.fda_indication",
        ],
        "manufacturing": [],
        "chemistry": [
            "passport.chemical_formula", "passport.smiles",
            "passport.inchi_key", "passport.molecular_weight",
        ],
    }

    def _count_section_fields(
        self, section_id: str, dossier: Optional[Dict], ev_set: set
    ) -> Tuple[int, int, int]:
        """Count expected / filled / evidenced fields for a section."""
        fields = self._SECTION_FIELD_MAP.get(section_id, [])
        if not fields or not dossier:
            return len(fields), 0, 0

        filled = 0
        evidenced = 0
        for fpath in fields:
            val = self._resolve_field(dossier, fpath)
            if val is not None:
                filled += 1
                # Check if there's evidence
                if isinstance(val, dict) and val.get("evidence_refs"):
                    if any(ref in ev_set for ref in val["evidence_refs"]):
                        evidenced += 1
                elif isinstance(val, list) and len(val) > 0:
                    # For list fields, check if any item has evidence
                    for item in val:
                        if isinstance(item, dict) and item.get("evidence_refs"):
                            if any(ref in ev_set for ref in item["evidence_refs"]):
                                evidenced += 1
                                break
        return len(fields), filled, evidenced

    def _count_critical_unknowns(self, section_id: str, dossier: Optional[Dict]) -> int:
        """Count critical unknowns relevant to this section."""
        if not dossier:
            return 0
        unknowns = dossier.get("unknowns", [])
        fields = self._SECTION_FIELD_MAP.get(section_id, [])
        count = 0
        for unk in unknowns:
            fp = unk.get("field_path", "")
            for f in fields:
                base = f.replace("[]", "")
                if fp.startswith(base.split(".")[0]):
                    count += 1
                    break
        return count

    def _resolve_field(self, dossier: Dict, field_path: str) -> Any:
        """Resolve a dot-path field from dossier dict. Returns None if missing."""
        parts = field_path.replace("[]", "").split(".")
        current = dossier
        for part in parts:
            if current is None:
                return None
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list) and len(current) > 0:
                # For list fields like registrations.US.status, find matching region
                found = None
                for item in current:
                    if isinstance(item, dict):
                        if item.get("region", "").upper() == part.upper():
                            found = item
                            break
                current = found
            else:
                return None
        return current

    # ── Decision readiness ───────────────────────────────────────────────────

    def _section_readiness(
        self,
        indexed: int,
        declared: int,
        fields_filled: int,
        fields_expected: int,
        critical_unknowns: int,
    ) -> str:
        """Compute decision readiness for a section."""
        if declared == 0:
            return "ready"  # No sources declared = N/A

        if critical_unknowns > 2:
            return "insufficient"
        if indexed == 0:
            return "insufficient"

        fill_ratio = fields_filled / fields_expected if fields_expected > 0 else 1.0
        source_ratio = indexed / declared if declared > 0 else 0.0

        if fill_ratio >= 0.7 and source_ratio >= 0.5 and critical_unknowns == 0:
            return "ready"
        if fill_ratio >= 0.3 or source_ratio >= 0.3:
            return "partial"
        return "insufficient"

    def _compute_overall_readiness(self, section_coverage: Dict[str, Dict]) -> str:
        """Compute overall decision readiness from section readiness."""
        statuses = [sc.get("decision_readiness", "insufficient") for sc in section_coverage.values()]
        if not statuses:
            return "insufficient"
        if all(s == "ready" for s in statuses):
            return "ready"
        if any(s == "insufficient" for s in statuses):
            return "insufficient"
        return "partial"

    # ── Wave 2 delta ─────────────────────────────────────────────────────────

    def _build_wave2_delta(
        self,
        all_docs: List[Dict],
        wave2_docs: Optional[List[Dict]],
        dossier: Optional[Dict],
        ev_set: set,
    ) -> Dict[str, Any]:
        """Build Wave 2 outside-source delta accounting."""
        if not wave2_docs:
            return {
                "wave2_enabled": False,
                "wave2_new_docs": 0,
                "wave2_new_facts": 0,
                "wave2_critical_new_facts": 0,
                "wave2_promoted_to_dossier": 0,
                "wave2_noise_docs": 0,
                "blind_spots": [],
            }

        wave2_ids = {d.get("id") for d in wave2_docs}
        wave1_ids = {d.get("id") for d in all_docs if d.get("id") not in wave2_ids}

        wave2_indexed = [d for d in wave2_docs if d.get("status") in _INDEXED_STATUSES]
        wave2_failed = [d for d in wave2_docs if d.get("status") in _FAILURE_STATUSES]

        # Count facts from Wave 2 in evidence
        wave2_facts = 0
        wave2_critical = 0
        wave2_promoted = 0
        wave2_doc_ids = {d.get("id") for d in wave2_docs}

        if dossier and dossier.get("evidence_registry"):
            for ev in dossier["evidence_registry"]:
                if ev.get("doc_id") in wave2_doc_ids:
                    wave2_facts += 1
                    wave2_promoted += 1

        # Wave 2 doc_kinds not in Wave 1
        wave1_kinds = {d.get("doc_kind") for d in all_docs if d.get("id") not in wave2_ids}
        wave2_only_kinds = {d.get("doc_kind") for d in wave2_indexed} - wave1_kinds

        blind_spots = []
        for dk in sorted(wave2_only_kinds):
            blind_spots.append({
                "doc_kind": dk,
                "found_in": "wave2_only",
                "docs_count": sum(1 for d in wave2_indexed if d.get("doc_kind") == dk),
                "useful": dk in {d.get("doc_kind") for d in wave2_docs
                                  if d.get("id") in wave2_doc_ids and d.get("status") in _INDEXED_STATUSES},
            })

        return {
            "wave2_enabled": True,
            "wave2_new_docs": len(wave2_docs),
            "wave2_indexed_docs": len(wave2_indexed),
            "wave2_failed_docs": len(wave2_failed),
            "wave2_new_facts": wave2_facts,
            "wave2_critical_new_facts": wave2_critical,
            "wave2_promoted_to_dossier": wave2_promoted,
            "wave2_noise_docs": len(wave2_docs) - len(wave2_indexed),
            "wave2_only_doc_kinds": sorted(wave2_only_kinds),
            "blind_spots": blind_spots,
        }

    # ── Limitations ──────────────────────────────────────────────────────────

    def _build_limitations(
        self,
        section_coverage: Dict[str, Dict],
        source_breakdown: List[Dict],
    ) -> List[Dict[str, Any]]:
        """Build human-readable limitations from coverage data."""
        limitations = []

        for sb in source_breakdown:
            if sb.get("status") == "unsupported":
                limitations.append({
                    "source_id": sb["source_id"],
                    "type": "unsupported",
                    "severity": "high",
                    "message": sb.get("reason", "Source not implemented"),
                    "impact": f"Missing fields: {', '.join(sb.get('critical_missing', []))}",
                })
            elif sb.get("status") in _FAILURE_STATUSES:
                limitations.append({
                    "source_id": sb["source_id"],
                    "type": "fetch_failure",
                    "severity": "medium",
                    "message": f"Source fetch failed with status: {sb['status']}",
                    "impact": f"Missing fields: {', '.join(sb.get('critical_missing', []))}",
                })
            elif sb.get("status") == "not_attached":
                limitations.append({
                    "source_id": sb["source_id"],
                    "type": "not_attached",
                    "severity": "medium",
                    "message": "Source was declared but no documents were attached",
                    "impact": f"Missing fields: {', '.join(sb.get('critical_missing', []))}",
                })

        for section_id, sc in section_coverage.items():
            if sc["decision_readiness"] == "insufficient":
                limitations.append({
                    "section": section_id,
                    "type": "insufficient_coverage",
                    "severity": "high",
                    "message": (
                        f"Section '{section_id}' has insufficient coverage: "
                        f"{sc['indexed_docs']}/{sc['declared_sources']} sources indexed, "
                        f"{sc['fields_filled']}/{sc['fields_expected']} fields filled, "
                        f"{sc['critical_unknowns']} critical unknowns"
                    ),
                })

        return limitations
