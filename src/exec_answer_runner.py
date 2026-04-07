"""
Exec Answer Runner — Sprint 22 WS2
=====================================
Standalone entry point called by Go api-gateway via subprocess.
Reads dossier JSON from stdin, runs the full WS2 exec pipeline,
outputs answer JSON to stdout.

Usage:
    echo '{"passport":...}' | python src/exec_answer_runner.py \
        --question-id q_reg_4geo --case-id UUID [--lens BD]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from question_router import QuestionRouter
from scope_resolver import ScopeResolver
from coverage_checker import CoverageChecker
from evidence_planner import EvidencePlanner
from claim_builder import ClaimBuilder
from exec_writer import ExecWriter

logger = logging.getLogger(__name__)


def run_exec_pipeline(
    dossier: dict,
    question_id: str,
    case_id: str,
    lens: str = "",
    allow_ws1: bool = False,
) -> dict:
    """
    Run the full WS2 exec Q&A pipeline.

    Steps:
    1. Route question from library
    2. Resolve scope from dossier product_contexts
    3. Check coverage
    4. Plan and collect evidence
    5. Build claims → answer_frame
    6. Write executive answer

    Returns JSON-serializable dict with answer + answer_frame + metadata.
    """
    t0 = time.time()

    # 1. Route question
    router = QuestionRouter()
    routed = router.route(question_id)

    # 2. Resolve scope
    scope_resolver = ScopeResolver()
    resolved_scope = scope_resolver.resolve(routed, dossier)

    # 3. Check coverage
    checker = CoverageChecker()
    coverage_ledger = dossier.get("coverage_ledger")
    coverage_decision = checker.check(routed, resolved_scope, dossier, coverage_ledger)

    # 4. Plan evidence
    planner = EvidencePlanner(retriever=None)  # No retriever in subprocess mode
    evidence_pack = planner.plan(routed, resolved_scope, coverage_decision, dossier)

    # 5. Build claims
    builder = ClaimBuilder()
    answer_frame = builder.build(routed, resolved_scope, coverage_decision, evidence_pack, dossier)

    # 6. Write executive answer
    writer = ExecWriter(mode="template")
    exec_result = writer.write(answer_frame, lens_profile=lens or None)

    elapsed = time.time() - t0

    # Extract INN from dossier for downstream consumers (PDF renderer, etc.)
    passport = dossier.get("passport", {})
    inn_raw = passport.get("inn")
    inn = inn_raw.get("value") if isinstance(inn_raw, dict) else inn_raw

    # Build response matching TZ spec
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


def main():
    parser = argparse.ArgumentParser(description="WS2 Exec Answer Runner")
    parser.add_argument("--question-id", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--lens", default="")
    parser.add_argument("--allow-ws1", action="store_true")
    args = parser.parse_args()

    # Read dossier from stdin
    dossier_json = sys.stdin.read()
    if not dossier_json.strip():
        print(json.dumps({"error": "No dossier JSON provided on stdin"}), file=sys.stdout)
        sys.exit(1)

    try:
        dossier = json.loads(dossier_json)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON: {e}"}), file=sys.stdout)
        sys.exit(1)

    try:
        result = run_exec_pipeline(
            dossier=dossier,
            question_id=args.question_id,
            case_id=args.case_id,
            lens=args.lens,
            allow_ws1=args.allow_ws1,
        )
        print(json.dumps(result, ensure_ascii=False, default=str), file=sys.stdout)
    except KeyError as e:
        print(json.dumps({"error": str(e)}), file=sys.stdout)
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": f"Pipeline error: {e}"}), file=sys.stdout)
        sys.exit(1)


if __name__ == "__main__":
    main()
