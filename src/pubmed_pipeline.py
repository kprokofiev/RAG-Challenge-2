import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

from src.ddkit_db import DDKitDB
from src.pubmed_client import PubMedArticle, PubMedEutilsClient
from src.storage_client import StorageClient
from src.text_splitter import TextSplitter
from src.ingestion import VectorDBIngestor


logger = logging.getLogger(__name__)


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _extract_pub_year(pubdate: str) -> Optional[int]:
    if not pubdate:
        return None
    parts = str(pubdate).split()
    for p in parts:
        if p.isdigit() and len(p) == 4:
            return _safe_int(p)
    return None


def _extract_authors(summary_doc: Dict[str, Any]) -> str:
    authors = summary_doc.get("authors") or []
    names: List[str] = []
    if isinstance(authors, list):
        for a in authors:
            if isinstance(a, dict):
                name = a.get("name")
                if name:
                    names.append(str(name))
    return "; ".join(names)


def _extract_doi(summary_doc: Dict[str, Any]) -> Optional[str]:
    for item in summary_doc.get("articleids", []) or []:
        if isinstance(item, dict) and item.get("idtype") == "doi":
            return str(item.get("value") or "").strip() or None
    elocation = summary_doc.get("elocationid")
    if isinstance(elocation, str) and elocation.lower().startswith("doi:"):
        return elocation.split("doi:", 1)[-1].strip() or None
    return None


def _tokenize(text: str) -> List[str]:
    return [t for t in (text or "").lower().split() if t]


@dataclass
class PubMedIngestResult:
    attached_docs: int = 0
    skipped_existing: int = 0
    fetched_pmids: int = 0


class PubMedIngestor:
    """
    PubMed as a first-class "source":
      search -> rerank -> attach -> parse (chunk+vector) -> upload artifacts

    We ingest PubMed records as text documents (not PDFs), indexed into the case vector store.
    """

    def __init__(
        self,
        storage: Optional[StorageClient] = None,
        db: Optional[DDKitDB] = None,
        client: Optional[PubMedEutilsClient] = None,
    ):
        self.storage = storage or StorageClient()
        self.db = db or DDKitDB()
        self.client = client or PubMedEutilsClient()

    def ingest_for_inn(
        self,
        tenant_id: str,
        case_id: str,
        inn: str,
        documents_dir: Path,
        vector_dir: Path,
        retmax: int = 80,
        attach_top_n: int = 12,
        deadline: Optional[float] = None,
    ) -> Tuple[PubMedIngestResult, List[Dict[str, Any]]]:
        """
        Returns (result, doc_meta_list).
        """
        from time import time as now

        inn = (inn or "").strip()
        if not inn:
            return PubMedIngestResult(), []

        if deadline is not None and now() > deadline:
            return PubMedIngestResult(), []

        term = f"{inn}[Title/Abstract]"
        pmids = self.client.search_pmids(term=term, retmax=retmax, sort="relevance")
        if not pmids:
            return PubMedIngestResult(), []

        # Limit total candidates (rerank stage happens below).
        pmids = pmids[: max(attach_top_n * 6, 30)]

        summaries = self.client.fetch_summaries(pmids)
        # EFetch abstracts in one batch for rerank; PubMed allows batch ids.
        abstracts = self.client.fetch_abstracts(pmids)

        articles: List[PubMedArticle] = []
        for pmid in pmids:
            s = summaries.get(str(pmid), {}) or {}
            title = str(s.get("title") or "").strip()
            journal = str(s.get("fulljournalname") or "").strip()
            pubdate = str(s.get("pubdate") or s.get("epubdate") or "").strip()
            pub_year = _extract_pub_year(pubdate)
            pub_types = [str(x) for x in (s.get("pubtype") or []) if str(x).strip()]
            doi = _extract_doi(s)
            authors = _extract_authors(s)
            abstract = abstracts.get(str(pmid), "") or ""
            url = self.client.pubmed_url(str(pmid))
            articles.append(
                PubMedArticle(
                    pmid=str(pmid),
                    title=title,
                    journal=journal,
                    pubdate=pubdate,
                    pub_year=pub_year,
                    pub_types=pub_types,
                    doi=doi,
                    authors=authors,
                    abstract=abstract,
                    source_url=url,
                )
            )

        # Rerank (BM25) on title+abstract.
        rerank_query = f"{inn} clinical trial randomized comparative real-world observational combination therapy"
        docs_tok = [_tokenize(a.to_text()) for a in articles]
        bm25 = BM25Okapi(docs_tok)
        scores = bm25.get_scores(_tokenize(rerank_query))
        scored = list(zip(articles, scores))
        scored.sort(key=lambda x: float(x[1]), reverse=True)
        top_articles = [a for a, _ in scored[:attach_top_n]]

        result = PubMedIngestResult(fetched_pmids=len(pmids))
        attached_meta: List[Dict[str, Any]] = []

        for art in top_articles:
            if deadline is not None and now() > deadline:
                break

            doc_title = art.title or f"PubMed {art.pmid}"
            doc_kind = "publication"
            published_at = str(art.pub_year) if art.pub_year else None

            # Upsert document row (id may be existing from previous runs).
            doc_id, _is_dup = self.db.upsert_document_by_source_url(
                tenant_id=tenant_id,
                case_id=case_id,
                doc_kind=doc_kind,
                title=doc_title,
                source_type="pubmed",
                source_url=art.source_url,
                status="parsed",
                language="en",
            )
            if not doc_id:
                continue

            chunk_path = documents_dir / f"{doc_id}.json"
            vector_path = vector_dir / f"{doc_id}.faiss"
            if chunk_path.exists() and vector_path.exists():
                result.skipped_existing += 1
                attached_meta.append(
                    {
                        "doc_id": doc_id,
                        "doc_kind": doc_kind,
                        "title": doc_title,
                        "source_url": art.source_url,
                        "region": "global",
                        "published_at": published_at,
                    }
                )
                continue

            ok = self._ingest_single_article(
                tenant_id=tenant_id,
                case_id=case_id,
                doc_id=doc_id,
                doc_kind=doc_kind,
                title=doc_title,
                source_url=art.source_url,
                published_at=published_at,
                content_text=art.to_text(),
                documents_dir=documents_dir,
                vector_dir=vector_dir,
            )
            if ok:
                result.attached_docs += 1
                attached_meta.append(
                    {
                        "doc_id": doc_id,
                        "doc_kind": doc_kind,
                        "title": doc_title,
                        "source_url": art.source_url,
                        "region": "global",
                        "published_at": published_at,
                    }
                )

        return result, attached_meta

    def _ingest_single_article(
        self,
        tenant_id: str,
        case_id: str,
        doc_id: str,
        doc_kind: str,
        title: str,
        source_url: str,
        published_at: Optional[str],
        content_text: str,
        documents_dir: Path,
        vector_dir: Path,
    ) -> bool:
        """
        Create a merged report JSON, chunk it, build vectors, and upload artifacts to S3.
        """
        metainfo = {
            "sha1_name": doc_id,
            "doc_id": doc_id,
            "doc_kind": doc_kind,
            "tenant_id": tenant_id,
            "case_id": case_id,
            "title": title,
            "source_url": source_url,
            "region": "global",
        }
        if published_at:
            metainfo["published_at"] = published_at

        pages = [{"page": 1, "text": content_text}]
        merged_report = {"metainfo": metainfo, "content": {"pages": pages, "chunks": None}}

        with tempfile.TemporaryDirectory() as td:
            merged_dir = Path(td) / "merged"
            merged_dir.mkdir(parents=True, exist_ok=True)
            merged_path = merged_dir / f"{doc_id}.json"
            with open(merged_path, "w", encoding="utf-8") as f:
                json.dump(merged_report, f, ensure_ascii=False, indent=2)

            # Chunk -> write chunked report into documents_dir.
            splitter = TextSplitter()
            _ = splitter.split_all_reports(merged_dir, documents_dir)

        chunk_path = documents_dir / f"{doc_id}.json"
        if not chunk_path.exists():
            return False

        # Vectorize only this report.
        vdb = VectorDBIngestor()
        try:
            vdb.process_single_report(chunk_path, vector_dir)
        except Exception as exc:
            logger.warning("PubMed vector ingestion failed for %s: %s", doc_id, exc)
            return False

        vector_path = vector_dir / f"{doc_id}.faiss"
        if not vector_path.exists():
            return False

        # Upload chunk bundle + vector.
        try:
            chunks_bundle_key, parsed_key = self._upload_pubmed_artifacts(
                tenant_id=tenant_id,
                case_id=case_id,
                doc_id=doc_id,
                chunk_json_path=chunk_path,
                vector_path=vector_path,
                parsed_payload=merged_report,
            )
            if self.db.is_configured():
                # Store a lightweight "parsed JSON" key for traceability.
                self.db.update_document_parsed(doc_id, parsed_key, pages_count=1)
            logger.info("PubMed ingested doc=%s chunks=%s", doc_id, chunks_bundle_key)
        except Exception as exc:
            logger.warning("PubMed artifact upload failed for %s: %s", doc_id, exc)
            return False

        return True

    def _upload_pubmed_artifacts(
        self,
        tenant_id: str,
        case_id: str,
        doc_id: str,
        chunk_json_path: Path,
        vector_path: Path,
        parsed_payload: Dict[str, Any],
    ) -> Tuple[str, str]:
        import tarfile

        # Upload a parsed JSON (not docling), so citations can link to pages if needed.
        parsed_key = f"tenants/{tenant_id}/cases/{case_id}/documents/{doc_id}/parsed/pubmed.json"
        self.storage.upload_bytes(parsed_key, json.dumps(parsed_payload, ensure_ascii=False).encode("utf-8"))

        # Bundle chunked report JSON into tar.gz (DDKit convention).
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            bundle_path = td_path / "chunks_bundle.tar.gz"
            with tarfile.open(bundle_path, "w:gz") as tar:
                tar.add(chunk_json_path, arcname=chunk_json_path.name)

            bundle_key = f"tenants/{tenant_id}/cases/{case_id}/documents/{doc_id}/chunks/chunks_bundle.tar.gz"
            if not self.storage.upload_file(bundle_key, bundle_path):
                raise RuntimeError("failed to upload chunks_bundle")

            manifest = {"doc_id": doc_id, "chunks_count": 1, "format": "tar.gz", "schema_version": 1}
            manifest_path = td_path / "manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
            manifest_key = f"tenants/{tenant_id}/cases/{case_id}/documents/{doc_id}/chunks/manifest.json"
            if not self.storage.upload_file(manifest_key, manifest_path):
                raise RuntimeError("failed to upload chunks manifest")

        vector_key = f"tenants/{tenant_id}/cases/{case_id}/documents/{doc_id}/vectors/{vector_path.name}"
        if not self.storage.upload_file(vector_key, vector_path):
            raise RuntimeError("failed to upload vector")

        return bundle_key, parsed_key

