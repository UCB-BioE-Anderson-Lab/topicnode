from __future__ import annotations

import argparse
import os
import sys

from topic_node.engine import run_topic_node
from topic_node.summarizer import run_summarization

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Iterative topic node processor for literature review trees."
    )
    parser.add_argument(
        "topic_root",
        help="Path to the topic root directory containing topic.txt, pdfs, and root.",
    )
    parser.add_argument(
        "pdfs_path",
        nargs="?",
        default=None,
        help="Optional explicit path to the pdfs directory. Defaults to <topic_root>/pdfs.",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help=(
            "Only run hierarchical summarization. "
            "Assumes the node tree has already been populated by the engine."
        ),
    )

    args = parser.parse_args(argv)
    summarize_only = args.summarize_only

    topic_root = os.path.abspath(args.topic_root)
    pdfs_path = os.path.abspath(args.pdfs_path) if args.pdfs_path else None

    stats = None
    if not summarize_only:
        try:
            stats = run_topic_node(topic_root, pdfs_path)
        except Exception as exc:
            sys.stderr.write(f"Error in engine: {exc}\n")
            return 1

        # Run report
        print("Run report")
        print("==========")
        print(f"Papers processed or reprocessed: {stats.papers_processed}")
        if stats.nodes_completed:
            print("\nNodes that transitioned to complete or had summaries refreshed:")
            for path in sorted(set(stats.nodes_completed)):
                print(f"  - {path}")
        if stats.conflicts:
            print("\nPapers requiring manual review due to rule conflicts or non matches:")
            for msg in stats.conflicts:
                print(f"  - {msg}")
        if stats.errors:
            print("\nErrors encountered:")
            for msg in stats.errors:
                print(f"  - {msg}")

    try:
        summary_stats = run_summarization(topic_root)
    except Exception as exc:
        sys.stderr.write(f"Error in summarization: {exc}\n")
        return 1

    print("\nSummary report")
    print("==============")
    print(
        f"Leaf nodes summarized: {len(summary_stats.leaf_nodes_summarized)}\n"
        f"Internal nodes summarized: {len(summary_stats.internal_nodes_summarized)}\n"
        f"Summarization errors: {len(summary_stats.errors)}"
    )

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
