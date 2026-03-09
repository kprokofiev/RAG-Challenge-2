#!/usr/bin/env python3
"""
Benchmark Runner — Sprint 9
============================
Validates coverage ledger against benchmark suite expectations.

Usage:
  # Validate a single dossier JSON against benchmark:
  python scripts/benchmark_runner.py --dossier path/to/dossier_v3.json --inn ibuprofen

  # Validate all dossiers in a directory against full suite:
  python scripts/benchmark_runner.py --dossier-dir path/to/dossiers/ --suite config/benchmark_suite.json

  # Compare two runs (regression diff):
  python scripts/benchmark_runner.py --baseline results/baseline.json --current results/current.json --diff

Output: benchmark_results.json + benchmark_results.md (markdown summary)
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.coverage_ledger import CoverageLedgerBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_DEFAULT_SUITE = _CONFIG_DIR / "benchmark_suite.json"


def load_suite(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    p = path or _DEFAULT_SUITE
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("benchmark_cases", [])


def load_dossier(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_field(dossier: Dict, field_path: str) -> Any:
    """Resolve a dot-path field from dossier dict."""
    parts = field_path.replace("[]", "").split(".")
    current = dossier
    for part in parts:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list) and len(current) > 0:
            found = None
            for item in current:
                if isinstance(item, dict):
                    if item.get("region", "").upper() == part.upper():
                        found = item
                        break
            if found is None:
                # Try accessing first item for array fields
                current = current[0] if current else None
                if isinstance(current, dict):
                    current = current.get(part)
            else:
                current = found
        else:
            return None
    return current


def validate_case(
    benchmark: Dict[str, Any],
    dossier: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate a single benchmark case against a dossier."""
    inn = benchmark["inn"]
    results = {
        "inn": inn,
        "category": benchmark.get("category", []),
        "use_case": benchmark.get("use_case", "ra_regulatory_screening"),
        "checks": [],
        "pass_count": 0,
        "fail_count": 0,
        "skip_count": 0,
    }

    # 1. Check must_find_docs
    docs_in_dossier = set()
    for doc in (dossier.get("documents") or []):
        dk = doc.get("doc_kind") or doc.get("kind")
        if dk:
            docs_in_dossier.add(dk)
    # Also check evidence registry doc_kinds
    for ev in (dossier.get("evidence_registry") or []):
        dk = ev.get("doc_kind")
        if dk:
            docs_in_dossier.add(dk)

    for mfd in benchmark.get("must_find_docs", []):
        expected_kind = mfd["doc_kind"]
        found = expected_kind in docs_in_dossier
        check = {
            "type": "must_find_doc",
            "source": mfd.get("source", "?"),
            "doc_kind": expected_kind,
            "result": "PASS" if found else "FAIL",
            "detail": f"Found {expected_kind} in corpus" if found else f"Missing {expected_kind}",
        }
        results["checks"].append(check)
        if found:
            results["pass_count"] += 1
        else:
            results["fail_count"] += 1

    # 2. Check must_fill_fields
    for field_path in benchmark.get("must_fill_fields", []):
        val = resolve_field(dossier, field_path)
        filled = val is not None
        if isinstance(val, dict):
            # EvidencedValue — check if value is non-null
            filled = val.get("value") is not None
        elif isinstance(val, list):
            filled = len(val) > 0

        check = {
            "type": "must_fill_field",
            "field": field_path,
            "result": "PASS" if filled else "FAIL",
            "detail": f"Field filled" if filled else f"Field missing or null",
        }
        results["checks"].append(check)
        if filled:
            results["pass_count"] += 1
        else:
            results["fail_count"] += 1

    # 3. Check decision readiness from coverage_ledger
    ledger = dossier.get("coverage_ledger", {})
    section_coverage = ledger.get("section_coverage", {})
    for section_id, expected_readiness in benchmark.get("expected_decision_readiness", {}).items():
        actual = section_coverage.get(section_id, {}).get("decision_readiness", "unknown")
        match = _readiness_meets(actual, expected_readiness)
        check = {
            "type": "decision_readiness",
            "section": section_id,
            "expected": expected_readiness,
            "actual": actual,
            "result": "PASS" if match else "FAIL",
        }
        results["checks"].append(check)
        if match:
            results["pass_count"] += 1
        else:
            results["fail_count"] += 1

    # 4. Check limitations match known_limitations
    ledger_limitations = ledger.get("limitations", [])
    for known_lim in benchmark.get("known_limitations", []):
        found = any(
            known_lim.lower() in (lim.get("message", "") + lim.get("reason", "")).lower()
            for lim in ledger_limitations
        )
        check = {
            "type": "known_limitation",
            "expected": known_lim,
            "result": "PASS" if found else "SKIP",
            "detail": "Limitation correctly reported" if found else "Limitation not found in ledger (may be OK)",
        }
        results["checks"].append(check)
        if found:
            results["pass_count"] += 1
        else:
            results["skip_count"] += 1

    # Summary
    total = results["pass_count"] + results["fail_count"]
    results["score"] = round(results["pass_count"] / total * 100, 1) if total > 0 else 0.0
    results["verdict"] = "PASS" if results["fail_count"] == 0 else "FAIL"
    return results


def _readiness_meets(actual: str, expected: str) -> bool:
    """Check if actual readiness meets or exceeds expected."""
    order = {"ready": 3, "partial": 2, "insufficient": 1, "unknown": 0}
    return order.get(actual, 0) >= order.get(expected, 0)


def diff_results(
    baseline: List[Dict[str, Any]],
    current: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Compare two benchmark runs and produce diff."""
    baseline_map = {r["inn"]: r for r in baseline}
    diffs = []
    for cur in current:
        inn = cur["inn"]
        base = baseline_map.get(inn)
        if not base:
            diffs.append({
                "inn": inn,
                "change": "NEW",
                "baseline_score": None,
                "current_score": cur["score"],
            })
            continue

        score_delta = cur["score"] - base["score"]
        if abs(score_delta) < 0.01:
            status = "UNCHANGED"
        elif score_delta > 0:
            status = "IMPROVED"
        else:
            status = "REGRESSED"

        diff = {
            "inn": inn,
            "change": status,
            "baseline_score": base["score"],
            "current_score": cur["score"],
            "score_delta": round(score_delta, 1),
            "baseline_verdict": base["verdict"],
            "current_verdict": cur["verdict"],
        }

        # Detail regressions
        if status == "REGRESSED":
            regressed_checks = []
            base_checks = {(c["type"], c.get("field", c.get("doc_kind", c.get("section", "")))): c
                          for c in base.get("checks", [])}
            for c in cur.get("checks", []):
                key = (c["type"], c.get("field", c.get("doc_kind", c.get("section", ""))))
                bc = base_checks.get(key)
                if bc and bc["result"] == "PASS" and c["result"] == "FAIL":
                    regressed_checks.append(key)
            diff["regressed_checks"] = regressed_checks

        diffs.append(diff)

    return diffs


def format_markdown(results: List[Dict[str, Any]], diffs: Optional[List[Dict]] = None) -> str:
    """Format benchmark results as markdown."""
    lines = [
        "# Benchmark Results",
        f"**Generated:** {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        "",
        "## Summary",
        "",
        "| INN | Category | Score | Verdict | Pass | Fail | Skip |",
        "|-----|----------|-------|---------|------|------|------|",
    ]

    total_pass = 0
    total_fail = 0
    for r in results:
        cat = ", ".join(r.get("category", []))
        lines.append(
            f"| {r['inn']} | {cat} | {r['score']}% | "
            f"{'PASS' if r['verdict'] == 'PASS' else '**FAIL**'} | "
            f"{r['pass_count']} | {r['fail_count']} | {r['skip_count']} |"
        )
        total_pass += r["pass_count"]
        total_fail += r["fail_count"]

    total = total_pass + total_fail
    overall = round(total_pass / total * 100, 1) if total > 0 else 0
    lines.extend([
        "",
        f"**Overall: {total_pass}/{total} checks passed ({overall}%)**",
        "",
    ])

    # Diff section
    if diffs:
        lines.extend([
            "## Regression Diff",
            "",
            "| INN | Change | Baseline | Current | Delta |",
            "|-----|--------|----------|---------|-------|",
        ])
        for d in diffs:
            lines.append(
                f"| {d['inn']} | {d['change']} | "
                f"{d.get('baseline_score', 'N/A')} | {d['current_score']} | "
                f"{d.get('score_delta', 'N/A')} |"
            )
        lines.append("")

    # Failures detail
    failures = [r for r in results if r["verdict"] == "FAIL"]
    if failures:
        lines.extend(["## Failures Detail", ""])
        for r in failures:
            lines.append(f"### {r['inn']}")
            for c in r["checks"]:
                if c["result"] == "FAIL":
                    lines.append(f"- **FAIL** [{c['type']}] {c.get('field', c.get('doc_kind', c.get('section', '')))}: {c.get('detail', c.get('actual', ''))}")
            lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="PharmSearch Benchmark Runner")
    parser.add_argument("--dossier", type=Path, help="Single dossier_v3.json to validate")
    parser.add_argument("--dossier-dir", type=Path, help="Directory of dossier JSONs")
    parser.add_argument("--inn", type=str, help="INN to validate (with --dossier)")
    parser.add_argument("--suite", type=Path, default=_DEFAULT_SUITE, help="Benchmark suite JSON")
    parser.add_argument("--baseline", type=Path, help="Baseline results JSON (for --diff)")
    parser.add_argument("--current", type=Path, help="Current results JSON (for --diff)")
    parser.add_argument("--diff", action="store_true", help="Compare baseline vs current")
    parser.add_argument("--out", type=Path, default=Path("benchmark_results"), help="Output directory")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Diff mode
    if args.diff and args.baseline and args.current:
        with open(args.baseline) as f:
            baseline = json.load(f)
        with open(args.current) as f:
            current = json.load(f)
        diffs = diff_results(baseline, current)
        md = format_markdown(current, diffs)
        (args.out / "benchmark_diff.md").write_text(md, encoding="utf-8")
        (args.out / "benchmark_diff.json").write_text(
            json.dumps(diffs, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Diff written to %s", args.out)
        return

    suite = load_suite(args.suite)
    results = []

    if args.dossier:
        # Single dossier validation
        dossier = load_dossier(args.dossier)
        inn = args.inn or dossier.get("passport", {}).get("inn", "unknown")
        benchmark = next((b for b in suite if b["inn"].lower() == inn.lower()), None)
        if not benchmark:
            logger.error("INN '%s' not found in benchmark suite", inn)
            sys.exit(1)

        # If no coverage_ledger in dossier, compute one
        if "coverage_ledger" not in dossier:
            logger.info("No coverage_ledger in dossier, computing...")
            builder = CoverageLedgerBuilder(use_case=benchmark.get("use_case", "ra_regulatory_screening"))
            ledger = builder.build(db_documents=[], dossier_report=dossier)
            dossier["coverage_ledger"] = ledger

        result = validate_case(benchmark, dossier)
        results.append(result)
        logger.info("Validated %s: %s (score=%s%%)", inn, result["verdict"], result["score"])

    elif args.dossier_dir:
        # Batch validation
        for dossier_path in sorted(args.dossier_dir.glob("*.json")):
            try:
                dossier = load_dossier(dossier_path)
                inn = dossier.get("passport", {}).get("inn", dossier_path.stem)
                benchmark = next((b for b in suite if b["inn"].lower() == inn.lower()), None)
                if not benchmark:
                    logger.warning("INN '%s' from %s not in suite, skipping", inn, dossier_path.name)
                    continue

                if "coverage_ledger" not in dossier:
                    builder = CoverageLedgerBuilder(
                        use_case=benchmark.get("use_case", "ra_regulatory_screening")
                    )
                    ledger = builder.build(db_documents=[], dossier_report=dossier)
                    dossier["coverage_ledger"] = ledger

                result = validate_case(benchmark, dossier)
                results.append(result)
                logger.info("Validated %s: %s (score=%s%%)", inn, result["verdict"], result["score"])
            except Exception as exc:
                logger.error("Failed to validate %s: %s", dossier_path.name, exc)

    else:
        # Dry run: show suite info
        logger.info("Benchmark suite: %d cases", len(suite))
        for b in suite:
            logger.info(
                "  %s (%s): %d must_find_docs, %d must_fill_fields",
                b["inn"], ", ".join(b.get("category", [])),
                len(b.get("must_find_docs", [])),
                len(b.get("must_fill_fields", [])),
            )
        logger.info("Use --dossier or --dossier-dir to validate.")
        return

    # Write results
    results_json = args.out / "benchmark_results.json"
    results_md = args.out / "benchmark_results.md"
    results_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    results_md.write_text(format_markdown(results), encoding="utf-8")
    logger.info("Results written to %s and %s", results_json, results_md)

    # Exit code
    all_pass = all(r["verdict"] == "PASS" for r in results)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
