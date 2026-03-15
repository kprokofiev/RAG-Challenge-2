"""
Coverage Ledger Builder — Sprint 9 + Sprint 14 Source Verdicts
================================================================
Computes a source-universe-aware coverage ledger for a dossier run.

Separates:
  - quality_v2 = quality of the output artifact (Sprint 7.5)
  - coverage_ledger = completeness relative to the declared source universe

Stages tracked per source:
  declared → reachable → attached → fetched → indexed → extracted → evidenced

Source verdicts (Sprint 14):
  KEEP     — live, contributing real fields
  CONNECT  — mandatory but not attached / not contributing
  FIX      — valuable but broken (integration/parser/deployment)
  DEMOTE   — little/no core fields; enrichment only
  PARK     — not needed for core dossier quality now

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

# ── Source verdicts (Sprint 14) ──────────────────────────────────────────────
# Authoritative classification from source audit matrix.
# Verdict logic: computed from runtime status + fields contributed + contract.

_VERDICT_KEEP = "KEEP"
_VERDICT_CONNECT = "CONNECT"
_VERDICT_FIX = "FIX"
_VERDICT_DEMOTE = "DEMOTE"
_VERDICT_PARK = "PARK"

# Static verdicts for sources where the classification is audit-confirmed
# and unlikely to change at runtime. These override dynamic computation.
_STATIC_VERDICTS: Dict[str, str] = {
    # KEEP — live, contributing
    "openfda": _VERDICT_KEEP,
    "dailymed": _VERDICT_KEEP,
    "ctgov": _VERDICT_KEEP,
    "pubchem": _VERDICT_KEEP,
    "epo_ops": _VERDICT_KEEP,
    "patentsview": _VERDICT_KEEP,
    "fda_label_safety": _VERDICT_KEEP,
    # FIX — valuable but broken
    "drugsatfda": _VERDICT_FIX,
    "grls": _VERDICT_FIX,
    "eaeu_portal": _VERDICT_FIX,
    "epo_register": _VERDICT_FIX,
    "rospatent": _VERDICT_FIX,
    "ctis": _VERDICT_FIX,
    # CONNECT — mandatory, not yet attached
    "ema_smpc": _VERDICT_CONNECT,
    "ema_pil": _VERDICT_CONNECT,
    "ema_epar": _VERDICT_CONNECT,
    "grls_instruction": _VERDICT_CONNECT,
    "ru_clinical": _VERDICT_CONNECT,
    # DEMOTE — enrichment only
    "pubmed": _VERDICT_DEMOTE,
    "lens_org": _VERDICT_DEMOTE,
    # PARK — not needed for core dossier
    "rems": _VERDICT_PARK,
    "ru_quality_letter": _VERDICT_PARK,
    "google_patents": _VERDICT_PARK,
    "eapo": _VERDICT_PARK,
    "manufacturing_svc": _VERDICT_PARK,
    "esklp": _VERDICT_PARK,
    "pharmacompass": _VERDICT_PARK,
}


def _compute_source_verdict(
    source_id: str,
    status: str,
    fields_contributed: int,
    must_have_fields: List[str],
    tier: str,
) -> str:
    """
    Compute source verdict dynamically.
    Static overrides take precedence, then heuristic classification.
    """
    if source_id in _STATIC_VERDICTS:
        return _STATIC_VERDICTS[source_id]

    # Dynamic fallback for sources not in static map
    if status == "unsupported":
        return _VERDICT_FIX
    if status == "ok" and fields_contributed > 0:
        return _VERDICT_KEEP
    if status == "ok" and fields_contributed == 0:
        return _VERDICT_DEMOTE if tier == "tier2" else _VERDICT_CONNECT
    if status in _FAILURE_STATUSES:
        return _VERDICT_FIX if must_have_fields else _VERDICT_PARK
    if status == "not_attached":
        if must_have_fields and tier == "tier1":
            return _VERDICT_CONNECT
        return _VERDICT_PARK
    return _VERDICT_PARK


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

        # Verdict summary (Sprint 14)
        verdict_counts: Dict[str, int] = {}
        seen_source_ids: set = set()
        for sb in all_source_breakdown:
            sid = sb.get("source_id", "")
            if sid in seen_source_ids:
                continue  # Deduplicate — same source can appear in multiple sections
            seen_source_ids.add(sid)
            v = sb.get("verdict", _VERDICT_PARK)
            verdict_counts[v] = verdict_counts.get(v, 0) + 1

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
                "verdict_summary": verdict_counts,
            },
            "section_coverage": section_coverage,
            "source_breakdown": all_source_breakdown,
            "family_health": self._build_family_health(all_source_breakdown),
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
                verdict = _compute_source_verdict(
                    src_id, "unsupported", 0,
                    contract.get("must_have_fields", []), tier,
                )
                source_breakdown.append({
                    "source_id": src_id,
                    "tier": tier,
                    "status": "unsupported",
                    "verdict": verdict,
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

            verdict = _compute_source_verdict(
                src_id, status, fields_contributed,
                contract.get("must_have_fields", []), tier,
            )
            source_breakdown.append({
                "source_id": src_id,
                "tier": tier,
                "status": status,
                "verdict": verdict,
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
            indexed_docs, declared_sources, fields_filled, fields_expected,
            critical_unknowns, source_breakdown,
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
        source_breakdown: Optional[List[Dict]] = None,
    ) -> str:
        """Compute decision readiness for a section.

        Sprint 14: PARK sources excluded from source_ratio denominator
        so they don't unfairly penalize readiness.
        """
        if declared == 0:
            return "ready"  # No sources declared = N/A

        if critical_unknowns > 2:
            return "insufficient"
        if indexed == 0:
            return "insufficient"

        # Sprint 14: exclude PARK sources from denominator
        effective_declared = declared
        if source_breakdown:
            non_park = sum(
                1 for sb in source_breakdown
                if sb.get("verdict", _VERDICT_PARK) != _VERDICT_PARK
            )
            effective_declared = max(non_park, 1)

        fill_ratio = fields_filled / fields_expected if fields_expected > 0 else 1.0
        source_ratio = indexed / effective_declared if effective_declared > 0 else 0.0

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

    # ── Source family health (Sprint 14 P2.2) ──────────────────────────────

    _SOURCE_FAMILIES: Dict[str, str] = {
        "openfda": "US Regulatory", "dailymed": "US Regulatory",
        "drugsatfda": "US Regulatory", "rems": "US Regulatory",
        "fda_label_safety": "US Regulatory",
        "ema_smpc": "EU Regulatory", "ema_pil": "EU Regulatory",
        "ema_epar": "EU Regulatory",
        "grls": "RU Regulatory", "grls_instruction": "RU Regulatory",
        "ru_quality_letter": "RU Regulatory",
        "eaeu_portal": "EAEU Regulatory",
        "ctgov": "US Clinical", "pubmed": "Literature",
        "ctis": "EU Clinical", "ru_clinical": "RU Clinical",
        "epo_ops": "EU Patent", "epo_register": "EU Patent",
        "google_patents": "Patent", "patentsview": "US Patent",
        "lens_org": "Global Patent", "rospatent": "RU Patent",
        "eapo": "EAPO Patent",
        "pubchem": "Chemistry",
        "manufacturing_svc": "Manufacturing", "esklp": "Manufacturing",
        "pharmacompass": "Manufacturing",
    }

    def _build_family_health(self, source_breakdown: List[Dict]) -> Dict[str, Dict]:
        """Aggregate source verdicts by family for bottleneck analysis."""
        families: Dict[str, Dict] = {}
        seen: set = set()

        for sb in source_breakdown:
            sid = sb.get("source_id", "")
            if sid in seen:
                continue
            seen.add(sid)

            family = self._SOURCE_FAMILIES.get(sid, "Other")
            if family not in families:
                families[family] = {
                    "sources": [],
                    "total": 0,
                    "ok": 0,
                    "fields_contributed": 0,
                    "verdicts": {},
                }
            fam = families[family]
            fam["sources"].append(sid)
            fam["total"] += 1
            if sb.get("status") == "ok":
                fam["ok"] += 1
            fam["fields_contributed"] += sb.get("fields_contributed", 0)
            v = sb.get("verdict", _VERDICT_PARK)
            fam["verdicts"][v] = fam["verdicts"].get(v, 0) + 1

        # Compute health per family
        for fam in families.values():
            ok_ratio = fam["ok"] / fam["total"] if fam["total"] > 0 else 0
            has_fix = fam["verdicts"].get(_VERDICT_FIX, 0) > 0
            has_connect = fam["verdicts"].get(_VERDICT_CONNECT, 0) > 0
            if ok_ratio >= 0.5 and fam["fields_contributed"] > 0:
                fam["health"] = "GREEN"
            elif ok_ratio > 0 or fam["fields_contributed"] > 0:
                fam["health"] = "YELLOW"
            elif has_fix or has_connect:
                fam["health"] = "RED"
            else:
                fam["health"] = "GREY"  # All PARK

        return families

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
        """Build human-readable limitations from coverage data with verdict-aware severity."""
        limitations = []

        for sb in source_breakdown:
            verdict = sb.get("verdict", _VERDICT_PARK)
            # PARK sources should not generate high-severity limitations
            severity_map = {
                _VERDICT_KEEP: "high",
                _VERDICT_CONNECT: "high",
                _VERDICT_FIX: "high",
                _VERDICT_DEMOTE: "low",
                _VERDICT_PARK: "info",
            }
            base_severity = severity_map.get(verdict, "medium")

            if sb.get("status") == "unsupported":
                limitations.append({
                    "source_id": sb["source_id"],
                    "type": "unsupported",
                    "severity": base_severity,
                    "verdict": verdict,
                    "message": sb.get("reason", "Source not implemented"),
                    "impact": f"Missing fields: {', '.join(sb.get('critical_missing', []))}",
                })
            elif sb.get("status") in _FAILURE_STATUSES:
                limitations.append({
                    "source_id": sb["source_id"],
                    "type": "fetch_failure",
                    "severity": base_severity,
                    "verdict": verdict,
                    "message": f"Source fetch failed with status: {sb['status']}",
                    "impact": f"Missing fields: {', '.join(sb.get('critical_missing', []))}",
                })
            elif sb.get("status") == "not_attached":
                # Only report non-PARK sources as limitations
                if verdict != _VERDICT_PARK:
                    limitations.append({
                        "source_id": sb["source_id"],
                        "type": "not_attached",
                        "severity": base_severity,
                        "verdict": verdict,
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
