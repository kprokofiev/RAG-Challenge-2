from __future__ import annotations

import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

APP_ROOT = Path("/app")
SRC_ROOT = APP_ROOT / "src"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.coverage_ledger import CoverageLedgerBuilder
from src.exec_answer_runner import run_exec_pipeline
from src.job_processors import DossierGenerateProcessor
from src.question_router import QuestionRouter
from src.dossier_report_generator import DossierReportGenerator


TENANT_ID = "demo"
CASE_ID = "e3598016-0f03-4680-a9fe-0e1ae69b82a4"
INN = "apixaban"
OUT_DIR = Path("/app/out/apixaban_settled_case_commercial_run")


def _doc_summary(processor: DossierGenerateProcessor) -> dict:
    docs = processor.ddkit_db.list_case_documents(TENANT_ID, CASE_ID)
    by_status = Counter(str(d.get("status", "")).lower() for d in docs)
    by_kind_status = Counter((d.get("doc_kind", "unknown"), str(d.get("status", "")).lower()) for d in docs)
    return {
        "status_counts": dict(sorted(by_status.items())),
        "doc_kind_status_counts": [
            {"doc_kind": kind, "status": status, "count": count}
            for (kind, status), count in sorted(by_kind_status.items())
        ],
        "total_documents": len(docs),
    }


def _source_verdicts(processor: DossierGenerateProcessor, temp_path: Path) -> dict:
    key = f"tenants/{TENANT_ID}/cases/{CASE_ID}/dossier/dossier.json"
    local = temp_path / "gateway_dossier.json"
    if processor.storage_client.download_to_path(key, local):
        with open(local, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data.get("source_verdicts", {}) or {}
    return {}


def _doc_titles_from_evidence(dossier_json: dict) -> list[str]:
    titles: set[str] = set()
    for item in dossier_json.get("evidence_registry") or []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("doc_title") or "").strip()
        if title:
            titles.add(title)
    return sorted(titles)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    processor = DossierGenerateProcessor()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        ok = processor._download_case_artifacts(temp_path, TENANT_ID, CASE_ID)
        if not ok:
            raise RuntimeError("Failed to download case artifacts from storage")

        source_verdicts = _source_verdicts(processor, temp_path)
        generator = DossierReportGenerator(
            vector_db_dir=temp_path / "databases" / "vector_dbs",
            documents_dir=temp_path / "databases" / "chunked_reports",
            inn=INN,
            tenant_id=TENANT_ID,
            case_id=CASE_ID,
        )
        dossier = generator.generate(
            case_id=CASE_ID,
            run_id=f"settled_commercial_{int(time.time())}",
            deadline=None,
            legacy_sections=None,
            completeness=None,
            source_verdicts=source_verdicts,
        )

    coverage_ledger = CoverageLedgerBuilder(use_case="ra_regulatory_screening").build(
        db_documents=processor.ddkit_db.list_case_documents(TENANT_ID, CASE_ID),
        dossier_report=dossier.model_dump(),
        evidence_registry=[ev.model_dump() for ev in dossier.evidence_registry or []],
    )
    dossier.coverage_ledger = coverage_ledger

    dossier_json = dossier.model_dump(mode="json")
    (OUT_DIR / "dossier_v3.json").write_text(
        json.dumps(dossier_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    router = QuestionRouter()
    exec_results = []
    for question in router.list_questions():
        exec_results.append(
            run_exec_pipeline(
                dossier=dossier_json,
                question_id=question["id"],
                case_id=CASE_ID,
                lens="BD",
            )
        )

    (OUT_DIR / "exec_results.json").write_text(
        json.dumps(exec_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    confidence_counts = Counter(
        str(result.get("confidence", {}).get("overall", "unknown")).lower()
        for result in exec_results
    )
    evidence_titles = _doc_titles_from_evidence(dossier_json)
    summary = {
        "case_id": CASE_ID,
        "inn": INN,
        "elapsed_s": round(time.time() - t0, 1),
        "registrations": len(dossier_json.get("registrations") or []),
        "clinical_studies": len(dossier_json.get("clinical_studies") or []),
        "patent_families": len(dossier_json.get("patent_families") or []),
        "synthesis_steps": len(dossier_json.get("synthesis_steps") or []),
        "commercial_signals": len(dossier_json.get("commercial_signals") or []),
        "unknowns": len(dossier_json.get("unknowns") or []),
        "evidence_registry": len(dossier_json.get("evidence_registry") or []),
        "confidence_counts": dict(sorted(confidence_counts.items())),
        "dossier_quality_v2": dossier_json.get("dossier_quality_v2") or {},
        "doc_summary": _doc_summary(processor),
        "evidence_titles": evidence_titles,
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    run_log_lines = [
        f"INN: {INN}",
        f"Case: {CASE_ID}",
        f"Elapsed seconds: {summary['elapsed_s']}",
        f"Registrations: {summary['registrations']}",
        f"Clinical studies: {summary['clinical_studies']}",
        f"Patent families: {summary['patent_families']}",
        f"Synthesis steps: {summary['synthesis_steps']}",
        f"Commercial signals: {summary['commercial_signals']}",
        f"Unknowns: {summary['unknowns']}",
        f"Evidence registry: {summary['evidence_registry']}",
        f"Exec confidence counts: {summary['confidence_counts']}",
        f"Dossier quality v2: {summary['dossier_quality_v2']}",
        f"Document status counts: {summary['doc_summary']['status_counts']}",
    ]
    (OUT_DIR / "run_log.md").write_text("\n".join(run_log_lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
