"""
Thin orchestrator for the exec decision engine.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)

try:
    from src.exec_decision_engine import ExecDecisionEngine
    from src.render_exec_decision_report import render_exec_decision_report
except ImportError:  # pragma: no cover
    from exec_decision_engine import ExecDecisionEngine  # type: ignore
    from render_exec_decision_report import render_exec_decision_report  # type: ignore


def _engine_enabled() -> bool:
    return (os.getenv("DDKIT_EXEC_ENGINE_ENABLED", "true") or "").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _render_enabled(mode: str) -> bool:
    env_mode = (os.getenv("DDKIT_EXEC_OUTPUT_MODE", "both") or "").strip().lower()
    if mode == "customer":
        return env_mode in {"customer", "both"} and (os.getenv("DDKIT_EXEC_RENDER_CUSTOMER", "true") or "").strip().lower() not in {"0", "false", "no", "off"}
    if mode == "internal":
        return env_mode in {"internal", "both"} and (os.getenv("DDKIT_EXEC_RENDER_INTERNAL", "true") or "").strip().lower() not in {"0", "false", "no", "off"}
    return False


def _report_summary(report: Dict[str, Any]) -> str:
    topline = report.get("topline", {})
    parts = []
    for key in ("asset_attractiveness", "rf_entry", "eaeu_entry"):
        item = topline.get(key, {})
        if item.get("verdict"):
            parts.append(f"{key}={item['verdict']}")
    if not parts:
        return "No topline verdicts available."
    return "; ".join(parts)


def _resolve_optional_path(path_value: Optional[Union[str, Path]]) -> Optional[Path]:
    if path_value is None:
        return None
    text = str(path_value).strip()
    if not text:
        return None
    return Path(text)


def _build_exec_retriever(
    retriever: Any = None,
    vector_db_dir: Optional[Union[str, Path]] = None,
    documents_dir: Optional[Union[str, Path]] = None,
) -> Any:
    if retriever is not None:
        return retriever
    resolved_vector_db_dir = _resolve_optional_path(vector_db_dir) or _resolve_optional_path(os.getenv("DDKIT_EXEC_VECTOR_DB_DIR"))
    resolved_documents_dir = _resolve_optional_path(documents_dir) or _resolve_optional_path(os.getenv("DDKIT_EXEC_DOCUMENTS_DIR"))
    if not resolved_vector_db_dir or not resolved_documents_dir:
        return None
    if not resolved_vector_db_dir.exists() or not resolved_documents_dir.exists():
        logger.warning(
            "exec_retriever_not_initialized vector_db_dir=%s documents_dir=%s",
            resolved_vector_db_dir,
            resolved_documents_dir,
        )
        return None
    try:
        try:
            from src.retrieval import HybridRetriever
        except ImportError:  # pragma: no cover
            from retrieval import HybridRetriever  # type: ignore
        return HybridRetriever(resolved_vector_db_dir, resolved_documents_dir)
    except Exception as exc:  # pragma: no cover
        logger.warning("exec_retriever_init_failed: %s", exc)
        return None


def _render_report_bundle(report: Dict[str, Any], output_dir: Optional[str]) -> Dict[str, str]:
    if not output_dir:
        return {}
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    rendered: Dict[str, str] = {}
    inn = report.get("inn") or "asset"
    safe_inn = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(inn))
    if _render_enabled("customer"):
        customer_path = str(Path(output_dir) / f"{safe_inn}_exec_customer.pdf")
        render_exec_decision_report(report, customer_path, mode="customer")
        rendered["customer_pdf"] = customer_path
    if _render_enabled("internal"):
        internal_path = str(Path(output_dir) / f"{safe_inn}_exec_internal.pdf")
        render_exec_decision_report(report, internal_path, mode="internal")
        rendered["internal_pdf"] = internal_path
    return rendered


def _run_legacy_exec_pipeline(
    dossier: dict,
    question_id: str,
    case_id: str,
    lens: str = "",
    allow_ws1: bool = False,
) -> dict:
    try:
        from question_router import QuestionRouter
        from scope_resolver import ScopeResolver
        from coverage_checker import CoverageChecker
        from evidence_planner import EvidencePlanner
        from claim_builder import ClaimBuilder
        from exec_writer import ExecWriter
    except ImportError:  # pragma: no cover
        from src.question_router import QuestionRouter
        from src.scope_resolver import ScopeResolver
        from src.coverage_checker import CoverageChecker
        from src.evidence_planner import EvidencePlanner
        from src.claim_builder import ClaimBuilder
        from src.exec_writer import ExecWriter

    t0 = time.time()
    router = QuestionRouter()
    routed = router.route(question_id)
    scope_resolver = ScopeResolver()
    resolved_scope = scope_resolver.resolve(routed, dossier)
    checker = CoverageChecker()
    coverage_ledger = dossier.get("coverage_ledger")
    coverage_decision = checker.check(routed, resolved_scope, dossier, coverage_ledger)
    planner = EvidencePlanner(retriever=None)
    evidence_pack = planner.plan(routed, resolved_scope, coverage_decision, dossier)
    builder = ClaimBuilder()
    answer_frame = builder.build(routed, resolved_scope, coverage_decision, evidence_pack, dossier)
    writer = ExecWriter(mode="template")
    exec_result = writer.write(answer_frame, lens_profile=lens or None)
    elapsed = time.time() - t0
    passport = dossier.get("passport", {})
    inn_raw = passport.get("inn")
    inn = inn_raw.get("value") if isinstance(inn_raw, dict) else inn_raw
    return {
        "inn": inn or "Unknown",
        "case_id": case_id,
        "question_id": question_id,
        "answer": exec_result.markdown,
        "short_summary": exec_result.short_summary,
        "answer_frame": answer_frame.model_dump(),
        "claims": [c.model_dump() for c in answer_frame.claims],
        "unknowns": answer_frame.unknowns,
        "confidence": answer_frame.confidence,
        "scope": answer_frame.scope,
        "coverage_decision": coverage_decision.model_dump(),
        "pipeline_ms": int(elapsed * 1000),
    }


def run_exec_pipeline(
    dossier: dict,
    question_id: str = "",
    case_id: str = "",
    lens: str = "",
    allow_ws1: bool = False,
    output_dir: Optional[str] = None,
    retriever: Any = None,
    vector_db_dir: Optional[Union[str, Path]] = None,
    documents_dir: Optional[Union[str, Path]] = None,
) -> dict:
    if not _engine_enabled():
        return _run_legacy_exec_pipeline(dossier, question_id, case_id, lens=lens, allow_ws1=allow_ws1)

    engine = ExecDecisionEngine(
        retriever=_build_exec_retriever(
            retriever=retriever,
            vector_db_dir=vector_db_dir,
            documents_dir=documents_dir,
        )
    )
    report = engine.generate(dossier, case_id=case_id or None)
    report_dict = report.model_dump()
    rendered = _render_report_bundle(report_dict, output_dir)
    return {
        "inn": report_dict.get("inn") or "Unknown",
        "case_id": case_id,
        "question_id": question_id,
        "answer": _report_summary(report_dict),
        "short_summary": _report_summary(report_dict),
        "report": report_dict,
        "topline": report_dict.get("topline", {}),
        "decision_blocks": report_dict.get("decision_blocks", []),
        "key_risks": report_dict.get("key_risks", []),
        "decision_blockers": report_dict.get("decision_blockers", []),
        "recommended_next_actions": report_dict.get("recommended_next_actions", []),
        "evidence_sufficiency": report_dict.get("evidence_sufficiency", {}),
        "verification": report_dict.get("verification", {}),
        "engine_manifest": report_dict.get("engine_manifest", {}),
        **rendered,
    }


def main():
    parser = argparse.ArgumentParser(description="Exec decision engine runner")
    parser.add_argument("--question-id", default="")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--lens", default="")
    parser.add_argument("--allow-ws1", action="store_true")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--vector-db-dir", default="")
    parser.add_argument("--documents-dir", default="")
    args = parser.parse_args()

    dossier_json = sys.stdin.read()
    if not dossier_json.strip():
        print(json.dumps({"error": "No dossier JSON provided on stdin"}), file=sys.stdout)
        sys.exit(1)

    try:
        dossier = json.loads(dossier_json)
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": f"Invalid JSON: {exc}"}), file=sys.stdout)
        sys.exit(1)

    try:
        result = run_exec_pipeline(
            dossier=dossier,
            question_id=args.question_id,
            case_id=args.case_id,
            lens=args.lens,
            allow_ws1=args.allow_ws1,
            output_dir=args.output_dir or None,
            vector_db_dir=args.vector_db_dir or None,
            documents_dir=args.documents_dir or None,
        )
        if args.output_dir:
            output_path = Path(args.output_dir) / "exec_result.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(result, ensure_ascii=False, default=str), file=sys.stdout)
    except KeyError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stdout)
        sys.exit(1)
    except Exception as exc:
        print(json.dumps({"error": f"Pipeline error: {exc}"}), file=sys.stdout)
        sys.exit(1)


if __name__ == "__main__":
    main()
