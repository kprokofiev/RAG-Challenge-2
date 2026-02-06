import os
import time
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import xml.etree.ElementTree as ET

import requests


logger = logging.getLogger(__name__)


PUBMED_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


@dataclass
class PubMedArticle:
    pmid: str
    title: str = ""
    journal: str = ""
    pubdate: str = ""
    pub_year: Optional[int] = None
    pub_types: List[str] = None  # type: ignore[assignment]
    authors: str = ""
    doi: Optional[str] = None
    abstract: str = ""
    source_url: str = ""

    def to_text(self) -> str:
        parts: List[str] = []
        if self.title:
            parts.append(f"Title: {self.title}")
        meta_bits: List[str] = []
        if self.journal:
            meta_bits.append(self.journal)
        if self.pubdate:
            meta_bits.append(self.pubdate)
        if self.doi:
            meta_bits.append(f"DOI: {self.doi}")
        if meta_bits:
            parts.append("Meta: " + " | ".join(meta_bits))
        if self.authors:
            parts.append(f"Authors: {self.authors}")
        if self.pub_types:
            parts.append("Publication types: " + ", ".join(self.pub_types))
        if self.abstract:
            parts.append("\nAbstract:\n" + self.abstract.strip())
        if self.source_url:
            parts.append(f"\nPubMed URL: {self.source_url}")
        return "\n".join(parts).strip()


class _RateLimiter:
    """
    Simple per-process rate limiter.

    NCBI guidance: with api_key up to ~10 requests/sec, without api_key ~3 req/sec.
    """

    def __init__(self, max_rps: float):
        self.min_interval = 1.0 / max(0.1, float(max_rps))
        self._last = 0.0

    def wait(self) -> None:
        now = time.perf_counter()
        delta = now - self._last
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self._last = time.perf_counter()


class PubMedEutilsClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = PUBMED_EUTILS_BASE,
        timeout_s: int = 30,
    ):
        self.api_key = api_key or os.getenv("PUBMED_API_KEY") or None
        self.base_url = base_url
        self.timeout_s = timeout_s
        max_rps = 10.0 if self.api_key else 3.0
        self._rl = _RateLimiter(max_rps=max_rps)
        self._session = requests.Session()

    @staticmethod
    def pubmed_url(pmid: str) -> str:
        pmid = str(pmid).strip()
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

    def _get(self, path: str, params: Dict[str, Any]) -> requests.Response:
        self._rl.wait()
        common: Dict[str, Any] = {}
        if self.api_key:
            common["api_key"] = self.api_key
        full = dict(common)
        full.update(params or {})
        url = self.base_url + path
        resp = self._session.get(url, params=full, timeout=self.timeout_s)
        resp.raise_for_status()
        return resp

    def search_pmids(self, term: str, retmax: int = 100, sort: str = "relevance") -> List[str]:
        params = {
            "db": "pubmed",
            "term": term,
            "retmode": "json",
            "retmax": int(retmax),
            "sort": sort,
        }
        data = self._get("esearch.fcgi", params=params).json()
        es = (data or {}).get("esearchresult", {}) or {}
        pmids = es.get("idlist", []) or []
        return [str(x) for x in pmids if str(x).strip()]

    def fetch_summaries(self, pmids: List[str]) -> Dict[str, Dict[str, Any]]:
        if not pmids:
            return {}
        ids = ",".join([str(x) for x in pmids])
        params = {"db": "pubmed", "id": ids, "retmode": "json"}
        data = self._get("esummary.fcgi", params=params).json()
        result = (data or {}).get("result", {}) or {}
        out: Dict[str, Dict[str, Any]] = {}
        for pmid in pmids:
            doc = result.get(str(pmid))
            if isinstance(doc, dict):
                out[str(pmid)] = doc
        return out

    def fetch_abstracts(self, pmids: List[str]) -> Dict[str, str]:
        """
        Fetch abstracts via EFetch XML and return pmid->abstract_text (may be empty).
        """
        if not pmids:
            return {}
        ids = ",".join([str(x) for x in pmids])
        params = {"db": "pubmed", "id": ids, "retmode": "xml"}
        xml_text = self._get("efetch.fcgi", params=params).text
        out: Dict[str, str] = {}
        try:
            root = ET.fromstring(xml_text)
        except Exception as exc:
            logger.warning("PubMed efetch XML parse failed: %s", exc)
            return out

        # PubmedArticleSet/PubmedArticle/MedlineCitation/PMID
        for art in root.findall(".//PubmedArticle"):
            pmid_el = art.find(".//MedlineCitation/PMID")
            pmid = (pmid_el.text or "").strip() if pmid_el is not None else ""
            if not pmid:
                continue
            abstract_parts: List[str] = []
            for abs_el in art.findall(".//Article/Abstract/AbstractText"):
                txt = "".join(abs_el.itertext()).strip()
                label = abs_el.attrib.get("Label") or abs_el.attrib.get("NlmCategory") or ""
                if label and txt:
                    abstract_parts.append(f"{label}: {txt}")
                elif txt:
                    abstract_parts.append(txt)
            out[pmid] = "\n".join([p for p in abstract_parts if p]).strip()
        return out

