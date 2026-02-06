"""
E2E (local) for CaseView v2 + PubMed as a source.

Why "local":
- Docker engine is not guaranteed on this machine.
- This script validates the full *data path* for PubMed -> (local) attach/parse/index -> retrieval -> LLM extraction
  -> case_view_v2 JSON output.

It uses:
- PubMed E-utilities (ESearch/ESummary/EFetch)
- OpenAI embeddings + LLM (via existing DDKit retriever + generator)

Run:
  python scripts/e2e_case_view_v2_pubmed.py --inn metformin --pubmed-top 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from src.case_view_v2_generator import CaseViewV2Generator
from src.pubmed_client import PubMedEutilsClient, PubMedArticle


def _tokenize(text: str) -> List[str]:
    return [t for t in (text or "").lower().split() if t]


def _build_min_snapshot(inn: str) -> dict:
    """
    Minimal frontend snapshot that:
    - satisfies passport gate (>=5 fields with citations)
    - provides 1 patent family w/ coverage_by_country
    - provides 1 complete RU trial card
    """
    inn = (inn or "").strip().lower()
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return {
        "inn": inn,
        "markets": ["ru", "eu", "us"],
        "generatedAt": now,
        "summary": {"inn": inn},
        "regulatory": {
            "us": {"status": "registered"},
            "eu": {"status": "registered", "countries": ["DE", "FR"]},
        },
        "fda_approval": f"{now[:10]} – approved (demo snapshot)",
        "chemical_formula": "C4H11N5",
        "drug_class": "demo class / demo MoA",
        "ruSections": {
            "regulatory": {
                "items": [
                    {
                        "reg_no": "RU-DEMO-0001",
                        "holder": "ООО Демо Холдер",
                        "status": "registered",
                        "forms": ["tablet"],
                        "strengths": ["500 mg"],
                        "routes": ["oral"],
                        "authorized_presentations": [],
                        "trade_name": "ДемоТН",
                        "links": {},
                    }
                ]
            },
            "clinical": {
                "studies": [
                    {
                        "id": "RU-DEMO-TRIAL-1",
                        "title": "Demo RU trial (snapshot)",
                        "phase": "Phase 3",
                        "study_type": "randomized",
                        "countries": ["RU"],
                        "enrollment": "120",
                        "comparator": "placebo",
                        "regimen": f"{inn} 500 mg BID",
                        "status": "completed",
                        "efficacy_key_points": ["Demo endpoint improved vs placebo (snapshot)"],
                        "conclusion": "Demo conclusion (snapshot)",
                        "where_conducted": "Russia",
                    }
                ]
            },
            "patents": {
                "patent_families": [
                    {
                        "familyId": "DEMO-FAM-1",
                        "members": [
                            {
                                "jurisdiction": "US",
                                "expiryDateBase": "2030-01-01",
                                "publicationNumber": "US-DEMO-123",
                            }
                        ],
                        "summary": {"mainStatus": "active"},
                    }
                ]
            },
        },
    }


def _pubmed_fetch_and_rerank(inn: str, retmax: int, top_n: int) -> List[PubMedArticle]:
    client = PubMedEutilsClient()
    term = f"{inn}[Title/Abstract]"
    pmids = client.search_pmids(term=term, retmax=retmax, sort="relevance")
    if not pmids:
        return []

    pmids = pmids[: max(top_n * 6, 30)]
    summaries = client.fetch_summaries(pmids)
    abstracts = client.fetch_abstracts(pmids)

    articles: List[PubMedArticle] = []
    for pmid in pmids:
        s = summaries.get(str(pmid), {}) or {}
        title = str(s.get("title") or "").strip()
        journal = str(s.get("fulljournalname") or "").strip()
        pubdate = str(s.get("pubdate") or s.get("epubdate") or "").strip()
        pub_types = [str(x) for x in (s.get("pubtype") or []) if str(x).strip()]
        doi = None
        for item in s.get("articleids", []) or []:
            if isinstance(item, dict) and item.get("idtype") == "doi":
                doi = str(item.get("value") or "").strip() or None
                break
        authors = ""
        au = s.get("authors") or []
        if isinstance(au, list):
            names = []
            for a in au:
                if isinstance(a, dict) and a.get("name"):
                    names.append(str(a["name"]))
            authors = "; ".join(names)

        abstract = abstracts.get(str(pmid), "") or ""
        url = client.pubmed_url(str(pmid))
        articles.append(
            PubMedArticle(
                pmid=str(pmid),
                title=title,
                journal=journal,
                pubdate=pubdate,
                pub_types=pub_types,
                doi=doi,
                authors=authors,
                abstract=abstract,
                source_url=url,
            )
        )

    # Simple BM25 rerank on title+abstract. (Fast + deterministic)
    rerank_query = f"{inn} randomized trial comparative real-world observational combination therapy"
    bm25 = BM25Okapi([_tokenize(a.to_text()) for a in articles])
    scores = bm25.get_scores(_tokenize(rerank_query))
    scored = list(zip(articles, scores))
    scored.sort(key=lambda x: float(x[1]), reverse=True)
    return [a for a, _ in scored[:top_n]]


def _ingest_article_local(
    doc_id: str,
    doc_kind: str,
    title: str,
    source_url: str,
    content_text: str,
    documents_dir: Path,
    vector_dir: Path,
) -> None:
    # Delayed import because ingestion depends on WorkerSettings (expects some env vars).
    from src.ingestion import VectorDBIngestor

    metainfo = {
        "sha1_name": doc_id,
        "doc_id": doc_id,
        "doc_kind": doc_kind,
        "tenant_id": "e2e",
        "case_id": "e2e",
        "title": title,
        "source_url": source_url,
        "region": "global",
    }
    # Create a minimal "already chunked" report:
    # - pages: for page-level display / citations
    # - chunks: for embeddings/retrieval
    report = {
        "metainfo": metainfo,
        "content": {
            "pages": [{"page": 1, "text": content_text}],
            "chunks": [{"page": 1, "text": content_text}],
        },
    }

    chunk_path = documents_dir / f"{doc_id}.json"
    with open(chunk_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    vdb = VectorDBIngestor()
    vdb.process_single_report(chunk_path, vector_dir)

    if not (vector_dir / f"{doc_id}.faiss").exists():
        raise RuntimeError(f"vector DB missing for {doc_id}")


def _ingest_demo_label(inn: str, documents_dir: Path, vector_dir: Path) -> None:
    """
    Minimal synthetic FDA label to drive doc-based regulatory + passport fields in the e2e run.
    """
    label_text = (
        f"FDA LABEL\n"
        f"Brand Name: GLUCOFAGE\n"
        f"Active Ingredient (INN): {inn}\n"
        f"Marketing Authorization Holder: Example Pharma Inc.\n"
        f"Dosage Forms and Strengths: tablets 500 mg, 850 mg, 1000 mg.\n"
        f"FDA Approval: 2016-01-01 for treatment of type 2 diabetes mellitus.\n"
        f"Drug Class / MoA: Biguanide; activates AMPK and reduces hepatic glucose production.\n"
        f"Indications:\n"
        f"- Type 2 diabetes mellitus in adults.\n"
        f"- Adjunct to diet and exercise for glycemic control.\n"
        f"- Combination therapy with insulin when needed.\n"
        f"- Combination therapy with sulfonylureas when needed.\n"
        f"Dosing:\n"
        f"- Start 500 mg twice daily with meals.\n"
        f"- Titrate by 500 mg weekly based on tolerance.\n"
        f"- Maximum 2000 mg/day in divided doses.\n"
        f"- Extended-release 500-1000 mg once daily with evening meal.\n"
        f"Warnings:\n"
        f"- Risk of lactic acidosis.\n"
        f"- Contraindicated in severe renal impairment.\n"
        f"- Use caution with hepatic impairment.\n"
    )
    _ingest_article_local(
        doc_id=str(uuid.uuid4()),
        doc_kind="label",
        title=f"FDA label (synthetic): {inn}",
        source_url="https://example.com/fda-label",
        content_text=label_text,
        documents_dir=documents_dir,
        vector_dir=vector_dir,
    )


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inn", default="metformin")
    ap.add_argument("--pubmed-retmax", type=int, default=60)
    ap.add_argument("--pubmed-top", type=int, default=6)
    ap.add_argument("--out", default="")
    ap.add_argument("--env", default=r"C:\GItHub\pharm_search\.env")
    args = ap.parse_args()

    load_dotenv(args.env, override=True)
    # WorkerSettings (imported indirectly by ingestion) requires these env vars even for local scripts.
    os.environ.setdefault("STORAGE_ENDPOINT_URL", "http://localhost:9000")
    os.environ.setdefault("STORAGE_ACCESS_KEY", "dummy")
    os.environ.setdefault("STORAGE_SECRET_KEY", "dummy")
    os.environ.setdefault("REDIS_URL", os.getenv("REDIS_URL") or "redis://localhost:6379/0")
    _assert(bool(os.getenv("OPENAI_API_KEY")), "OPENAI_API_KEY missing (load env first)")

    inn = (args.inn or "").strip().lower()
    _assert(bool(inn), "--inn is required")

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        documents_dir = base / "documents"
        vector_dir = base / "vectors"
        documents_dir.mkdir(parents=True, exist_ok=True)
        vector_dir.mkdir(parents=True, exist_ok=True)

        # 0) Synthetic FDA label to keep passport/regulatory doc-backed in e2e.
        _ingest_demo_label(inn, documents_dir, vector_dir)

        # 1) PubMed -> local ingest (attach/parse/index)
        arts = _pubmed_fetch_and_rerank(inn, retmax=args.pubmed_retmax, top_n=max(1, args.pubmed_top))
        _assert(bool(arts), f"PubMed returned no articles for inn={inn}")

        for art in arts:
            doc_id = str(uuid.uuid4())
            _ingest_article_local(
                doc_id=doc_id,
                doc_kind="publication",
                title=art.title or f"PubMed {art.pmid}",
                source_url=art.source_url,
                content_text=art.to_text(),
                documents_dir=documents_dir,
                vector_dir=vector_dir,
            )

        # 2) Generate case_view_v2 using snapshot + web enrichment
        snapshot = _build_min_snapshot(inn)
        gen = CaseViewV2Generator(
            documents_dir=documents_dir,
            vector_db_dir=vector_dir,
            tenant_id="e2e",
            case_id="e2e",
        )
        deadline = time.time() + 900
        payload = gen.generate_case_view(snapshot=snapshot, query=inn, inn=inn, use_web=True, use_snapshot=True, deadline=deadline)

        # 3) Assertions (basic UI readiness + pubmed populated)
        _assert(payload.get("schema_version") == "2.0", "schema_version must be 2.0")

        stats = payload.get("source_stats") or {}
        _assert("facts_total" in stats, "source_stats.facts_total missing")
        _assert(stats.get("ready_for_ui") is True, "ready_for_ui expected to be true for the demo snapshot")

        pubmed = (((payload.get("sections") or {}).get("clinical") or {}).get("pubmed") or {})
        pubs_total = sum(len(pubmed.get(k) or []) for k in ("comparative", "abstracts", "real_world", "combination"))
        _assert(pubs_total > 0, "clinical.pubmed is empty (expected PubMed docs to be used)")

        out_path = args.out.strip()
        if not out_path:
            out_path = str(Path.cwd() / f"case_view_v2_e2e_{inn}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"[OK] E2E finished. PubMed articles ingested: {len(arts)}; pubmed items in case_view: {pubs_total}")
        print(f"[OK] Output: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
