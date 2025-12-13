from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests


@dataclass
class PaperRecord:
    title: str
    doi: Optional[str]
    year: Optional[int]
    n_citations: Optional[int]
    url_landing: Optional[str]
    url_pdf: Optional[str]
    venue: Optional[str]
    authors: List[str]
    source: str


class ProviderError(RuntimeError):
    pass


def _norm_doi(doi: Optional[str]) -> Optional[str]:
    if not doi:
        return None
    d = doi.strip()
    if not d:
        return None
    if d.lower().startswith("https://doi.org/"):
        d = d[len("https://doi.org/") :]
    return d.strip() or None


class SemanticScholarProvider:
    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        self.base = "https://api.semanticscholar.org/graph/v1"

    def _headers(self) -> Dict[str, str]:
        h: Dict[str, str] = {"User-Agent": "topicnode/0.1"}
        if self.api_key:
            h["x-api-key"] = self.api_key
        return h

    def search(self, query: str, limit: int = 25) -> List[PaperRecord]:
        fields = "title,year,citationCount,externalIds,url,venue,authors,openAccessPdf"
        params = {"query": query, "limit": str(limit), "fields": fields}
        url = f"{self.base}/paper/search"

        r = requests.get(url, params=params, headers=self._headers(), timeout=30)
        if r.status_code != 200:
            raise ProviderError(f"Semantic Scholar search failed: {r.status_code} {r.text[:200]}")
        data = r.json()
        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return []

        out: List[PaperRecord] = []
        for it in items:
            if not isinstance(it, dict):
                continue

            title = str(it.get("title") or "").strip()
            if not title:
                continue

            year = it.get("year") if isinstance(it.get("year"), int) else None
            ccount = it.get("citationCount") if isinstance(it.get("citationCount"), int) else None
            venue = it.get("venue") if isinstance(it.get("venue"), str) else None
            landing = it.get("url") if isinstance(it.get("url"), str) else None

            ext = it.get("externalIds") if isinstance(it.get("externalIds"), dict) else {}
            doi = _norm_doi(ext.get("DOI") if isinstance(ext, dict) else None)

            oa = it.get("openAccessPdf") if isinstance(it.get("openAccessPdf"), dict) else {}
            oa_url = oa.get("url") if isinstance(oa, dict) else None
            pdf_url = oa_url if isinstance(oa_url, str) else None

            authors: List[str] = []
            a = it.get("authors")
            if isinstance(a, list):
                for au in a:
                    if isinstance(au, dict) and isinstance(au.get("name"), str):
                        authors.append(au["name"])

            out.append(
                PaperRecord(
                    title=title,
                    doi=doi,
                    year=year,
                    n_citations=ccount,
                    url_landing=landing,
                    url_pdf=pdf_url,
                    venue=venue,
                    authors=authors,
                    source="semantic_scholar",
                )
            )

        return out
