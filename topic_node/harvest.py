from __future__ import annotations

import csv
import json
import os
import shutil
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

from topic_node.openai_requests import run_text_prompt
from topic_node.search_providers import SemanticScholarProvider


@dataclass
class HarvestDecision:
    message: str
    exit_code: int = 0


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _list_pdfs(dirpath: str) -> List[str]:
    if not os.path.isdir(dirpath):
        return []
    return sorted([n for n in os.listdir(dirpath) if n.lower().endswith(".pdf")])


def _topic_to_queries(topic_text: str, model: str = "gpt-5") -> Dict[str, Any]:
    prompt = f"""You generate literature search queries.

Goal: produce a diverse set of search queries that find papers related to the topic from different angles.
Some queries should be tight to the topic as written; others should target the core methods or concepts that
are central to the topic even if they do not mention the most specific entities in the topic.

Return JSON ONLY:
{{
  "queries": [
    {{
      "q": "search query string",
      "weight": 1.0,
      "bucket": "direct|method_core|adjacent",
      "focus": "one short phrase describing what this query targets"
    }}
  ],
  "notes": "optional"
}}

Constraints:
- 8 to 14 queries total.
- Include 2–4 queries in each bucket:
  - direct: tightly matches the topic as written.
  - method_core: drop or generalize the most specific instance/entity constraints but keep the core method/problem.
  - adjacent: nearby concepts that are commonly required in practice but still clearly related.
- Queries must be concise and specific.
- Do not include negative/exclusion terms.
- Return JSON only.

TOPIC:
{topic_text}
"""
    out = run_text_prompt(prompt, model=model)
    if isinstance(out, dict) and isinstance(out.get("queries"), list):
        return out
    return {"queries": [{"q": topic_text, "weight": 1.0, "bucket": "direct"}], "notes": "fallback"}


def _write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _write_tsv(path: str, rows: list[dict], fields: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _score_seed_candidate(hit, query_weight: float) -> float:
    c = getattr(hit, "n_citations", None)
    if not isinstance(c, int):
        c = 0

    y = getattr(hit, "year", None)
    recency = 0.0
    if isinstance(y, int):
        recency = max(0.0, min(1.0, (y - 1995) / 30.0))

    qw = float(query_weight) if isinstance(query_weight, (int, float)) else 1.0
    return qw * (1.0 + (c ** 0.5) / 10.0) * (0.7 + 0.3 * recency)


def _search_seed_candidates(topic_text: str, harvest_dir: str) -> Tuple[Dict[str, Any], list[dict]]:
    queries_obj = _topic_to_queries(topic_text)
    _write_json(os.path.join(harvest_dir, "seed_queries.json"), queries_obj)

    provider = SemanticScholarProvider(api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY"))
    rows: list[dict] = []

    for qobj in (queries_obj.get("queries") or [])[:14]:
        if not isinstance(qobj, dict):
            continue

        q = qobj.get("q")
        if not isinstance(q, str) or not q.strip():
            continue

        bucket = qobj.get("bucket", "")
        weight = qobj.get("weight", 1.0)

        q_base = q.strip()

        try:
            hits = provider.search(q_base, limit=25)
        except Exception:
            hits = []

        for h in hits:
            score = _score_seed_candidate(h, weight)
            rows.append(
                {
                    "bucket": bucket,
                    "score": score,
                    "title": getattr(h, "title", "") or "",
                    "year": getattr(h, "year", "") if isinstance(getattr(h, "year", None), int) else "",
                    "venue": getattr(h, "venue", "") or "",
                    "doi": getattr(h, "doi", "") or "",
                    "url_landing": getattr(h, "url_landing", "") or getattr(h, "url", "") or "",
                    "url_pdf": getattr(h, "url_pdf", "") or "",
                    "n_citations": getattr(h, "n_citations", "") if isinstance(getattr(h, "n_citations", None), int) else "",
                    "hit_queries": q_base,
                    "provider": "semantic_scholar",
                }
            )

    rows.sort(key=lambda r: float(r.get("score") or 0.0), reverse=True)

    dedup: dict[str, dict] = {}
    for r in rows:
        key = (r.get("doi") or "").strip().lower()
        if not key:
            key = (r.get("url_landing") or "").strip().lower()
        if not key:
            key = (r.get("title") or "").strip().lower() + "::" + str(r.get("year") or "")
        if key in dedup:
            continue
        dedup[key] = r

    final = list(dedup.values())[:200]
    for i, r in enumerate(final, start=1):
        r["rank"] = i
        r["score"] = str(float(r.get("score") or 0.0))

    fields = [
        "rank",
        "bucket",
        "score",
        "title",
        "year",
        "venue",
        "doi",
        "url_landing",
        "url_pdf",
        "n_citations",
        "hit_queries",
        "provider",
    ]
    _write_tsv(os.path.join(harvest_dir, "seed_candidates.tsv"), final, fields)
    return queries_obj, final


def _triage_seed_candidates(topic_text: str, candidates: list[dict], model: str = "gpt-5") -> Dict[str, Any]:
    by_bucket: dict[str, list[dict]] = {}
    for r in candidates:
        b = (r.get("bucket") or "").strip() or "unknown"
        by_bucket.setdefault(b, []).append(r)

    sample: list[dict] = []
    sample.extend(candidates[:30])
    for b in ["direct", "method_core", "adjacent"]:
        if b in by_bucket:
            sample.extend(by_bucket[b][:15])

    seen = set()
    unique_sample = []
    for r in sample:
        key = (r.get("doi") or "").strip().lower()
        if not key:
            key = (r.get("url_landing") or "").strip().lower()
        if not key:
            key = (r.get("title") or "").strip().lower() + "::" + str(r.get("year") or "")
        if key in seen:
            continue
        seen.add(key)
        unique_sample.append(r)
    sample = unique_sample[:60]

    items = []
    for r in sample:
        items.append(
            {
                "rank": r.get("rank"),
                "bucket": r.get("bucket"),
                "title": r.get("title"),
                "year": r.get("year"),
                "venue": r.get("venue"),
                "doi": r.get("doi"),
                "n_citations": r.get("n_citations"),
                "url_landing": r.get("url_landing"),
                "url_pdf": r.get("url_pdf"),
            }
        )

    prompt = f"""You are selecting seed papers for citation expansion.

We will use the selected seeds for:
- backward citations (papers they cite)
- forward citations (papers that cite them)

Goal: pick 5 to 10 papers that are most likely to sit near the center of the citation graph for the topic and
lead to many relevant papers, while still being aligned with the topic.

Selection guidance:
- Prefer influential, method-defining, framework, and widely-used protocol papers closely related to the topic.
- Prefer papers with high citation counts when relevance is comparable.
- Include a mix across buckets (direct + method_core) when possible.
- Avoid narrow application papers or papers likely far from the topic’s core.

Return JSON ONLY with:
{{
  "selected": [
    {{
      "rank": <rank from the list>,
      "reason": "one sentence reason"
    }}
  ],
  "notes": "optional"
}}

TOPIC:
{topic_text}

CANDIDATES (ranked subset):
{json.dumps(items, ensure_ascii=False)}
"""
    out = run_text_prompt(prompt, model=model)
    if isinstance(out, dict) and isinstance(out.get("selected"), list):
        return out
    return {"selected": [], "notes": "fallback"}


def _write_seed_shortlist(harvest_dir: str, candidates: list[dict], triage: Dict[str, Any]) -> list[dict]:
    rank_map: dict[int, dict] = {}
    for r in candidates:
        try:
            rk = int(r.get("rank"))
        except Exception:
            continue
        rank_map[rk] = r

    shortlist = []
    for item in triage.get("selected", []):
        if not isinstance(item, dict):
            continue
        try:
            rk = int(item.get("rank"))
        except Exception:
            continue
        r = rank_map.get(rk)
        if not r:
            continue

        title = (r.get("title") or "").strip()
        url_pdf = (r.get("url_pdf") or "").strip()
        url_landing = (r.get("url_landing") or "").strip()
        doi = (r.get("doi") or "").strip()
        url = url_pdf or url_landing or (f"https://doi.org/{doi}" if doi else "")

        out = {
            "title": title,
            "url": url,
            "reason": (item.get("reason") or "").strip(),
        }
        shortlist.append(out)

    seen = set()
    final = []
    for r in shortlist:
        key = (r.get("url") or "").strip().lower() or (r.get("title") or "").strip().lower()
        if key in seen:
            continue
        seen.add(key)
        final.append(r)

    fields = ["title", "url", "reason"]
    _write_tsv(os.path.join(harvest_dir, "seed_shortlist.tsv"), final, fields)
    return final


def decide_next_harvest_step(topic_root: str) -> Optional[HarvestDecision]:
    topic_root = os.path.abspath(topic_root)
    _ensure_dir(topic_root)

    topic_path = os.path.join(topic_root, "topic.txt")
    harvest_dir = os.path.join(topic_root, "harvest")
    pdfs_dir = os.path.join(topic_root, "pdfs")
    root_dir = os.path.join(topic_root, "root")
    root_papers_dir = os.path.join(root_dir, "papers")
    root_children_dir = os.path.join(root_dir, "children")
    seed_dir = os.path.join(root_dir, "seed")
    seed_rejects_dir = os.path.join(root_dir, "seed_rejects")

    for p in [harvest_dir, pdfs_dir, root_papers_dir, root_children_dir, seed_dir, seed_rejects_dir]:
        _ensure_dir(p)

    if not os.path.exists(topic_path):
        return HarvestDecision(
            message="Please create topic.txt in this folder describing the literature topic, then run again.",
            exit_code=2,
        )

    topic_text = _read_text(topic_path)
    if not topic_text:
        return HarvestDecision(
            message="topic.txt is empty. Please write the topic description, then run again.",
            exit_code=2,
        )

    seed_pdfs = _list_pdfs(seed_dir)
    if not seed_pdfs:
        _ = _search_seed_candidates(topic_text=topic_text, harvest_dir=harvest_dir)
        _, candidates = _
        triage = _triage_seed_candidates(topic_text, candidates)
        shortlist = _write_seed_shortlist(harvest_dir, candidates, triage)

        msg = "Wrote harvest/seed_queries.json and harvest/seed_candidates.tsv."
        if shortlist:
            msg += " Also wrote harvest/seed_shortlist.tsv. Next: download PDFs for each row in seed_shortlist.tsv into root/seed/ and run again.\n\n"
            msg += "\n".join([r["url"] for r in shortlist if r.get("url")]) + "\n"
        else:
            msg += " Next: download PDFs for a few top rows into root/seed/ and run again."
        print(msg)
        return HarvestDecision(message=msg, exit_code=2)

    copied = 0
    for name in seed_pdfs:
        src = os.path.join(seed_dir, name)
        dst = os.path.join(pdfs_dir, name)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            copied += 1

    if copied > 0:
        return HarvestDecision(
            message=f"Copied {copied} seed PDFs into pdfs/. Next: run again to process them.",
            exit_code=2,
        )

    return None
