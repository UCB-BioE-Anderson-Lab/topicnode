from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

from topic_node.engine import (
    ensure_dir,
    load_json,
    save_json,
    list_child_nodes,
    get_node_description,
    read_text_required,
)
from topic_node.llm_interface import (
    LLMError,
    summarize_node_from_papers,
    summarize_node_from_child_summaries,
)


@dataclass
class SummaryStats:
    leaf_nodes_summarized: List[str] = field(default_factory=list)
    internal_nodes_summarized: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def _is_leaf(node_dir: str) -> bool:
    """A node is a leaf if it has no child directories under children/."""
    return len(list_child_nodes(node_dir)) == 0


def _load_node_meta(node_dir: str) -> Dict[str, Any]:
    meta_path = os.path.join(node_dir, "node_meta.json")
    if os.path.exists(meta_path):
        return load_json(meta_path)
    # Fallback minimal meta if somehow missing
    return {
        "node_id": os.path.basename(node_dir),
        "relative_path": os.path.relpath(node_dir, os.path.dirname(node_dir)).replace(
            os.sep, "/"
        ),
        "node_type": "leaf",
        "status": "active",
    }


def _summarize_leaf_node(
    topic_root: str,
    node_dir: str,
    topic_text: str,
    stats: SummaryStats,
) -> None:
    """Summarize a leaf node directly from its papers' explanations/features."""
    summaries_dir = os.path.join(node_dir, "summaries")
    summary_path = os.path.join(summaries_dir, "node_summary.json")

    # If a summary already exists, do nothing.
    if os.path.exists(summary_path):
        return

    papers_dir = os.path.join(node_dir, "papers")
    if not os.path.isdir(papers_dir):
        return

    node_meta = _load_node_meta(node_dir)
    node_description = get_node_description(node_meta, topic_root)

    leaf_papers: List[Dict[str, Any]] = []

    for name in sorted(os.listdir(papers_dir)):
        if not name.lower().endswith(".json"):
            continue
        json_path = os.path.join(papers_dir, name)
        data = load_json(json_path)

        pdf_filename = data.get("pdf_filename") or (os.path.splitext(name)[0] + ".pdf")

        features = data.get("features")
        if not isinstance(features, dict):
            features = {}

        # Skip triaged papers
        if features.get("triage") is True:
            continue

        # Prefer short_explanation from features, fall back to explanation
        short_expl = ""
        feat_short = features.get("short_explanation")
        if isinstance(feat_short, str) and feat_short.strip():
            short_expl = feat_short.strip()
        else:
            explanation = data.get("explanation")
            if isinstance(explanation, dict):
                expl_short = explanation.get("short_explanation")
                if isinstance(expl_short, str) and expl_short.strip():
                    short_expl = expl_short.strip()

        explanation = data.get("explanation")
        relevance_score = "unknown"
        if isinstance(explanation, dict):
            rs = explanation.get("relevance_score")
            if isinstance(rs, str) and rs:
                relevance_score = rs

        leaf_papers.append(
            {
                "pdf_filename": pdf_filename,
                "short_explanation": short_expl,
                "relevance_score": relevance_score,
                "features": features,
            }
        )

    # Nothing to summarize (no non-triaged papers)
    if not leaf_papers:
        summary_obj = {
            "node_relative_path": node_meta.get("relative_path"),
            "summary_text": "",
            "key_findings": [],
            "limitations": [],
            "citations": [],
            "raw_summary": {},
        }

        ensure_dir(summaries_dir)
        save_json(summary_path, summary_obj)
        stats.leaf_nodes_summarized.append(node_meta.get("relative_path", ""))
        return

    try:
        summary = summarize_node_from_papers(
            leaf_papers=leaf_papers,
            topic_text=topic_text,
            node_description=node_description,
        )
    except LLMError as exc:
        stats.errors.append(
            f"LLM error in summarize_node_from_papers for node "
            f"{node_meta.get('relative_path', '')}: {exc}"
        )
        return

    summary_obj = {
        "node_relative_path": node_meta.get("relative_path"),
        "summary_text": summary.get("summary_text", ""),
        "key_findings": summary.get("key_findings", []),
        "limitations": summary.get("limitations", []),
        "citations": summary.get("citations", []),
        "raw_summary": summary,
    }

    ensure_dir(summaries_dir)
    save_json(summary_path, summary_obj)
    stats.leaf_nodes_summarized.append(node_meta.get("relative_path", ""))


def _summarize_internal_node(
    topic_root: str,
    node_dir: str,
    topic_text: str,
    stats: SummaryStats,
) -> None:
    """Summarize an internal node from its child summaries."""
    summaries_dir = os.path.join(node_dir, "summaries")
    summary_path = os.path.join(summaries_dir, "node_summary.json")

    # If a summary already exists, do nothing.
    if os.path.exists(summary_path):
        return

    child_dirs = list_child_nodes(node_dir)
    if not child_dirs:
        return

    child_summaries: List[Dict[str, Any]] = []
    for child in child_dirs:
        child_summary_path = os.path.join(child, "summaries", "node_summary.json")
        if not os.path.exists(child_summary_path):
            # Child has no summary yet, so we skip summarizing this parent for now.
            return
        child_summaries.append(load_json(child_summary_path))

    if not child_summaries:
        return

    node_meta = _load_node_meta(node_dir)
    node_description = get_node_description(node_meta, topic_root)

    try:
        summary = summarize_node_from_child_summaries(
            child_summaries=child_summaries,
            topic_text=topic_text,
            node_description=node_description,
        )
    except LLMError as exc:
        stats.errors.append(
            f"LLM error in summarize_node_from_child_summaries for node "
            f"{node_meta.get('relative_path', '')}: {exc}"
        )
        return

    summary_obj = {
        "node_relative_path": node_meta.get("relative_path"),
        "summary_text": summary.get("summary_text", ""),
        "key_findings": summary.get("key_findings", []),
        "limitations": summary.get("limitations", []),
        "citations": summary.get("citations", []),
        "raw_summary": summary,
    }

    ensure_dir(summaries_dir)
    save_json(summary_path, summary_obj)
    stats.internal_nodes_summarized.append(node_meta.get("relative_path", ""))


def _summarize_node_recursive(
    topic_root: str,
    node_dir: str,
    topic_text: str,
    stats: SummaryStats,
) -> None:
    """Post-order traversal: summarize children first, then this node."""
    child_dirs = list_child_nodes(node_dir)

    if child_dirs:
        # Internal or future-internal node: process children first
        for child in child_dirs:
            _summarize_node_recursive(topic_root, child, topic_text, stats)

        # Then summarize this node from child summaries
        _summarize_internal_node(topic_root, node_dir, topic_text, stats)
    else:
        # Leaf node: summarize directly from papers
        _summarize_leaf_node(topic_root, node_dir, topic_text, stats)


def run_summarization(topic_root: str) -> SummaryStats:
    """Top-level entry point for hierarchical summarization.

    Assumes that engine has already:
      - assigned PDFs to nodes
      - generated explanations/features
      - routed papers into children

    This pass:
      - walks the tree rooted at topic_root/root
      - writes summaries/node_summary.json for leaf nodes (from papers)
      - then writes summaries/node_summary.json for internal nodes (from child summaries)
      - does not overwrite existing node_summary.json files
    """
    root_node_dir = os.path.join(topic_root, "root")
    if not os.path.isdir(root_node_dir):
        raise RuntimeError(f"Root node directory does not exist: {root_node_dir}")

    topic_path = os.path.join(topic_root, "topic.txt")
    topic_text = read_text_required(topic_path, "topic.txt")

    stats = SummaryStats()

    print(f"[summarizer] Starting summarization for topic root {topic_root}")
    _summarize_node_recursive(topic_root, root_node_dir, topic_text, stats)
    print(
        f"[summarizer] Summarization complete: "
        f"{len(stats.leaf_nodes_summarized)} leaf nodes, "
        f"{len(stats.internal_nodes_summarized)} internal nodes summarized, "
        f"{len(stats.errors)} errors"
    )

    return stats