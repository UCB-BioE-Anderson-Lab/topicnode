from __future__ import annotations

import json
import os
import random
import time
import hashlib
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from topic_node.llm_interface import (
    LLMError,
    characterize_paper,
    extract_paper_features,
    propose_node_splits,
)
from topic_node.rule_engine import evaluate_rule as _legacy_evaluate_rule

DEFAULT_SAMPLE_SIZE = 10


@dataclass
class RunStats:
    papers_processed: int = 0
    nodes_completed: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class TopicContext:
    topic_root: str
    pdfs_dir: str
    topic_text: str
    topic_mtime: float
    stats: RunStats


# ---------------------------------------------------------------------------
# Basic filesystem / JSON helpers
# ---------------------------------------------------------------------------


def read_text_required(path: str, description: str) -> str:
    if not os.path.exists(path):
        raise RuntimeError(f"Required {description} missing at {path}")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if not text.strip():
        raise RuntimeError(f"Required {description} at {path} is empty")
    return text


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def compute_node_signature(
    node_dir: str,
    topic_mtime: float,
    triage_prompt_file: Optional[str],
) -> Dict[str, Any]:
    """Compute a simple signature of the node's inputs.

    Used to detect when a node's inputs (papers, topic, triage prompt) have
    changed between runs.
    """

    papers_dir = os.path.join(node_dir, "papers")
    paper_files = sorted(
        [f for f in os.listdir(papers_dir)] if os.path.isdir(papers_dir) else []
    )
    items: List[Tuple[str, float]] = []
    for name in paper_files:
        path = os.path.join(papers_dir, name)
        try:
            mtime = os.path.getmtime(path)
        except FileNotFoundError:
            mtime = 0.0
        items.append((name, mtime))

    triage_mtime: Optional[float] = None
    if triage_prompt_file:
        triage_path = os.path.join(node_dir, triage_prompt_file)
        if os.path.exists(triage_path):
            triage_mtime = os.path.getmtime(triage_path)

    sig_obj = {
        "papers": items,
        "triage_mtime": triage_mtime,
        "topic_mtime": topic_mtime,
    }
    payload = json.dumps(sig_obj, sort_keys=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return {
        "paper_filenames": paper_files,
        "hash": digest,
    }


# ---------------------------------------------------------------------------
# JSON rule evaluation wrapper
# ---------------------------------------------------------------------------


def _eval_json_rule(rule: Any, features: Dict[str, Any]) -> bool:
    """Evaluate a rule expressed in the JSON rule language.

    Supported forms (exactly one key in each dict):
      - {"all": [subrule, ...]}  logical AND
      - {"any": [subrule, ...]}  logical OR
      - {"not": subrule}         logical NOT
      - {"eq": [feature_name, value]}
      - {"neq": [feature_name, value]}
      - {"in": [feature_name, [v1, v2, ...]]}
    """

    if not isinstance(rule, dict) or len(rule) != 1:
        return False

    op, value = next(iter(rule.items()))

    if op == "all":
        if not isinstance(value, list):
            return False
        return all(_eval_json_rule(r, features) for r in value)

    if op == "any":
        if not isinstance(value, list):
            return False
        return any(_eval_json_rule(r, features) for r in value)

    if op == "not":
        return not _eval_json_rule(value, features)

    if op in ("eq", "neq"):
        if not isinstance(value, list) or len(value) != 2:
            return False
        feature_name, literal = value
        fval = features.get(feature_name)
        if op == "eq":
            return fval == literal
        return fval != literal

    if op == "in":
        if not isinstance(value, list) or len(value) != 2:
            return False
        feature_name, allowed = value
        if not isinstance(allowed, list):
            return False
        fval = features.get(feature_name)
        return fval in allowed

    # Unknown operator
    return False


def evaluate_rule(rule: Any, features: Dict[str, Any]) -> bool:
    """Wrapper that supports both JSON rules and legacy string rules."""

    if isinstance(rule, dict):
        return _eval_json_rule(rule, features)
    # Fall back to legacy implementation (string expressions, etc.)
    return _legacy_evaluate_rule(rule, features)


# ---------------------------------------------------------------------------
# Paper discovery and node metadata
# ---------------------------------------------------------------------------


def discover_assigned_papers(root_node_dir: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root_node_dir):
        if os.path.basename(dirpath) == "papers":
            for name in filenames:
                if not name.lower().endswith(".json"):
                    continue
                pdf_name = os.path.splitext(name)[0] + ".pdf"
                mapping[pdf_name] = dirpath
    return mapping


def assign_unassigned_pdfs_to_root(
    topic_root: str,
    pdfs_dir: str,
    root_node_dir: str,
    stats: RunStats,
) -> None:
    """Ensure every PDF has a JSON stub in the root node."""

    assigned = discover_assigned_papers(root_node_dir)
    ensure_dir(os.path.join(root_node_dir, "papers"))
    for name in sorted(os.listdir(pdfs_dir)):
        if not name.lower().endswith(".pdf"):
            continue
        if name in assigned:
            continue
        json_name = os.path.splitext(name)[0] + ".json"
        json_path = os.path.join(root_node_dir, "papers", json_name)
        if os.path.exists(json_path):
            continue
        paper_record = {
            "pdf_filename": name,
            "node_relative_path": os.path.relpath(root_node_dir, topic_root).replace(
                os.sep, "/"
            ),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "updated_at": None,
            "topic_mtime": None,
            "triage_prompt_mtime": None,
            "pdf_mtime": os.path.getmtime(os.path.join(pdfs_dir, name)),
            "explanation": {},
            "features": {},
        }
        save_json(json_path, paper_record)
        # papers_processed is reserved for actual LLM processing


def ensure_node_meta(
    context: TopicContext,
    node_dir: str,
    is_root: bool,
) -> Dict[str, Any]:
    meta_path = os.path.join(node_dir, "node_meta.json")
    ensure_dir(node_dir)
    ensure_dir(os.path.join(node_dir, "papers"))
    ensure_dir(os.path.join(node_dir, "children"))

    topic_root = context.topic_root

    if os.path.exists(meta_path):
        meta = load_json(meta_path)
    else:
        node_id = os.path.basename(node_dir)
        relative_path = os.path.relpath(node_dir, topic_root).replace(os.sep, "/")
        meta = {
            "node_id": node_id,
            "relative_path": relative_path,
            "node_type": "root" if is_root else "leaf",
            "status": "active",
            "triage_prompt_file": None,
            "last_run_signature": {
                "paper_filenames": [],
                "hash": "",
            },
        }

    # Ensure required keys with defaults
    meta.setdefault("node_id", os.path.basename(node_dir))
    meta.setdefault(
        "relative_path",
        os.path.relpath(node_dir, topic_root).replace(os.sep, "/"),
    )
    meta.setdefault("node_type", "root" if is_root else "leaf")
    meta.setdefault("status", "active")
    meta.setdefault("triage_prompt_file", None)
    meta.setdefault("last_run_signature", {"paper_filenames": [], "hash": ""})

    # If triage_prompt_file is not set but a default file exists, use it
    if meta.get("triage_prompt_file") is None:
        default_triage = os.path.join(node_dir, "triage_prompt.txt")
        if os.path.exists(default_triage):
            meta["triage_prompt_file"] = "triage_prompt.txt"

    # Compute current signature
    current_sig = compute_node_signature(
        node_dir,
        topic_mtime=context.topic_mtime,
        triage_prompt_file=meta.get("triage_prompt_file"),
    )
    meta["current_signature"] = current_sig

    save_json(meta_path, meta)
    return meta


def load_paper_json(path: str) -> Dict[str, Any]:
    try:
        return load_json(path)
    except Exception:
        return {}


def save_paper_json(path: str, data: Dict[str, Any]) -> None:
    data.setdefault("created_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_json(path, data)


def list_child_nodes(node_dir: str) -> List[str]:
    children_dir = os.path.join(node_dir, "children")
    if not os.path.isdir(children_dir):
        return []
    return sorted(
        os.path.join(children_dir, name)
        for name in os.listdir(children_dir)
        if os.path.isdir(os.path.join(children_dir, name))
    )


def get_node_description(node_meta: Dict[str, Any], topic_root: str) -> str:
    relative_path = node_meta.get("relative_path", "")
    node_type = node_meta.get("node_type", "")
    status = node_meta.get("status", "")
    return f"Node path: {relative_path} (type: {node_type}, status: {status})"


# ---------------------------------------------------------------------------
# Per-node processing
# ---------------------------------------------------------------------------


def sample_papers(node_dir: str, k: int) -> List[str]:
    """Return up to k JSON filenames for papers in this node."""

    papers_dir = os.path.join(node_dir, "papers")
    all_json = [
        name
        for name in sorted(os.listdir(papers_dir))
        if name.lower().endswith(".json")
    ]
    if len(all_json) <= k:
        return all_json
    return random.sample(all_json, k)


def maybe_generate_triage_prompt(
    context: TopicContext,
    node_dir: str,
    node_meta: Dict[str, Any],
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Ensure this node has a triage prompt and split spec.

    If a triage prompt already exists on disk, load and return it. Otherwise:
    - Sample up to DEFAULT_SAMPLE_SIZE papers in this node.
    - Call characterize_paper on those PDFs.
    - Call propose_node_splits with the sampled characterizations.
    - Expect a JSON response that includes at least:
        - sampled_summary: textual summary (for human debugging).
        - features: a list of feature definition objects (without triage).
        - splits: list of group objects with group_id, display_name, rule.
    - Append a hard-coded boolean triage feature to the feature list.
    - Append a short_explanation feature for downstream summarization.
    - Construct a node-specific triage prompt text from the feature list.
    - Save triage_prompt.txt and attach splits/feature_schema into node_meta.
    """

    topic_root = context.topic_root

    # If we already have a triage prompt file, load it and reuse.
    triage_name = node_meta.get("triage_prompt_file")
    if triage_name:
        triage_path = os.path.join(node_dir, triage_name)
        if os.path.exists(triage_path):
            with open(triage_path, "r", encoding="utf-8") as f:
                triage_text = f.read()
            split_spec = node_meta.get("split_spec") or {}
            print(f"[topicnode] Reusing existing triage prompt for {node_meta.get('relative_path')}")
            return split_spec, triage_text

    # No existing prompt: generate from a sample of papers.
    papers_dir = os.path.join(node_dir, "papers")
    sampled_json = sample_papers(node_dir, DEFAULT_SAMPLE_SIZE)
    if not sampled_json:
        # No papers, nothing to do.
        print(f"[topicnode] Node {node_meta.get('relative_path')} has no papers to sample")
        return {}, None

    node_description = get_node_description(node_meta, topic_root)

    print(
        f"[topicnode] Generating triage prompt for {node_meta.get('relative_path')} "
        f"from {len(sampled_json)} sampled papers"
    )

    sampled_payload: List[Dict[str, Any]] = []
    for json_name in sampled_json:
        json_path = os.path.join(papers_dir, json_name)
        paper_data = load_paper_json(json_path)
        pdf_filename = paper_data.get("pdf_filename") or (
            os.path.splitext(json_name)[0] + ".pdf"
        )
        pdf_path = os.path.join(context.pdfs_dir, pdf_filename)
        if not os.path.exists(pdf_path):
            context.stats.errors.append(
                f"PDF {pdf_filename} for paper JSON {json_path} is missing under {context.pdfs_dir}"
            )
            continue
        try:
            expl = characterize_paper(
                pdf_path=pdf_path,
                topic_text=context.topic_text,
                node_description=node_description,
            )
        except LLMError as exc:
            context.stats.errors.append(
                f"LLM error in characterize_paper for {pdf_filename} in node {node_meta.get('relative_path')}: {exc}"
            )
            continue

        # Store explanation back on the paper for reuse
        if isinstance(expl, dict):
            paper_data["explanation"] = expl
        else:
            paper_data["explanation"] = {"raw": expl}
        save_paper_json(json_path, paper_data)

        sampled_payload.append(
            {
                "pdf_filename": pdf_filename,
                "explanation": expl,
            }
        )

    if not sampled_payload:
        return {}, None

    try:
        proposal = propose_node_splits(
            sampled_papers=sampled_payload,
            topic_text=context.topic_text,
            node_description=node_description,
        )
    except LLMError as exc:
        context.stats.errors.append(
            f"LLM error while proposing splits for node {node_meta.get('relative_path')}: {exc}"
        )
        return {}, None

    # Extract feature definitions and splits from the proposal.
    feature_defs = proposal.get("features") or []
    if not isinstance(feature_defs, list):
        feature_defs = []

    # Drop any accidental triage feature supplied by the model.
    feature_defs = [
        f for f in feature_defs
        if isinstance(f, dict) and f.get("name") != "triage"
    ]

    # Append the hard-coded triage feature.
    triage_feature = {
        "name": "triage",
        "description": "Whether the paper is clearly out of scope for the global topic or this node.",
        "value_type": "boolean",
        "allowed_values": [],
        "instructions": (
            "Set this to true if the paper is clearly irrelevant to the overall topic "
            "or to the current node's focus, based on the topic and node description. "
            "Otherwise set this to false."
        ),
    }
    feature_defs.append(triage_feature)

    # Insert a per-paper summary feature.
    summary_feature = {
        "name": "short_explanation",
        "description": (
            "A plain-language description of how this paper relates to the global topic "
            "and the current node, including the main experimental setup and key findings."
        ),
        "value_type": "string",
        "allowed_values": [],
        "instructions": (
            "Write a detailed plain-language description of how the paper relates to the overall topic "
            "and this node's focus. Describe the main experimental setup, important conditions or strains, "
            "and the key findings that are most relevant to the topic. Include as many concrete statements "
            "and results as needed so that this text alone contains enough information to summarize the paper later."
        ),
    }
    feature_defs.append(summary_feature)

    if not feature_defs:
        context.stats.errors.append(
            f"propose_node_splits for node {node_meta.get('relative_path')} "
            f"did not return any usable features"
        )
        return {}, None

    splits = proposal.get("splits") or []
    split_spec = {
        "features": feature_defs,
        "splits": splits,
        "sampled_summary": proposal.get("sampled_summary", ""),
    }

    # Build the triage prompt text that will be inserted into extract_paper_features.
    lines: List[str] = []
    lines.append(
        "You must return a JSON object with a top-level key \"features\" whose value "
        "is an object mapping feature names to values."
    )
    lines.append(
        "For each feature defined below, set features[\"name\"] according to its type, "
        "allowed values, and instructions. Use null only when a value truly cannot be determined."
    )
    lines.append("")
    lines.append("Feature definitions:")
    lines.append("")

    for feat in feature_defs:
        if not isinstance(feat, dict):
            continue
        name = feat.get("name", "")
        if not name:
            continue
        desc = feat.get("description", "")
        vtype = feat.get("value_type", "")
        allowed = feat.get("allowed_values") or []
        instr = feat.get("instructions", "")

        lines.append(f"- name: {name}")
        if desc:
            lines.append(f"  description: {desc}")
        if vtype:
            lines.append(f"  type: {vtype}")
        if allowed:
            allowed_str = ", ".join(repr(v) for v in allowed)
            lines.append(f"  allowed_values: [{allowed_str}]")
        if instr:
            lines.append(f"  instructions: {instr}")
        lines.append("")

    triage_prompt = "\n".join(lines)

    # Persist triage prompt and schema.
    triage_path = os.path.join(node_dir, "triage_prompt.txt")
    with open(triage_path, "w", encoding="utf-8") as f:
        f.write(triage_prompt)

    node_meta["triage_prompt_file"] = "triage_prompt.txt"
    node_meta["feature_schema"] = feature_defs
    node_meta["split_spec"] = split_spec
    save_json(os.path.join(node_dir, "node_meta.json"), node_meta)

    print(
        f"[topicnode] Wrote triage prompt and split spec for {node_meta.get('relative_path')} "
        f"with {len(feature_defs)} features and {len(splits)} splits"
    )

    return split_spec, triage_prompt


def extract_features_for_all_papers(
    context: TopicContext,
    node_dir: str,
    node_meta: Dict[str, Any],
    triage_prompt_text: str,
) -> None:
    """Run extract_paper_features on all papers in this node.

    The prompt is expected to define a features schema including a boolean
    "triage" field, which the engine will later use to drop irrelevant
    papers from routing.
    """

    papers_dir = os.path.join(node_dir, "papers")
    topic_mtime = context.topic_mtime

    triage_path = os.path.join(node_dir, node_meta.get("triage_prompt_file", ""))
    triage_mtime = os.path.getmtime(triage_path) if os.path.exists(triage_path) else None

    print(f"[topicnode] Extracting features for node {node_meta.get('relative_path')}")

    for name in sorted(os.listdir(papers_dir)):
        if not name.lower().endswith(".json"):
            continue
        json_path = os.path.join(papers_dir, name)
        data = load_paper_json(json_path)

        pdf_filename = data.get("pdf_filename") or (os.path.splitext(name)[0] + ".pdf")
        pdf_path = os.path.join(context.pdfs_dir, pdf_filename)
        if not os.path.exists(pdf_path):
            context.stats.errors.append(
                f"PDF {pdf_filename} for paper JSON {json_path} is missing under {context.pdfs_dir}"
            )
            continue

        pdf_mtime = os.path.getmtime(pdf_path)

        # Decide whether to recompute features
        features = data.get("features")
        if (
            isinstance(features, dict)
            and data.get("topic_mtime") == topic_mtime
            and data.get("triage_prompt_mtime") == triage_mtime
            and data.get("pdf_mtime") == pdf_mtime
        ):
            # Cached features are up to date
            continue

        node_description = get_node_description(node_meta, context.topic_root)

        try:
            feat = extract_paper_features(
                pdf_path=pdf_path,
                topic_text=context.topic_text,
                node_description=node_description,
                triage_prompt_text=triage_prompt_text,
            )
        except LLMError as exc:
            context.stats.errors.append(
                f"LLM error in extract_paper_features for {pdf_filename} in node {node_meta.get('relative_path')}: {exc}"
            )
            continue

        if isinstance(feat, dict):
            data["features_full"] = feat
            base = feat.get("features", feat)
            data["features"] = base if isinstance(base, dict) else {}
        else:
            data["features_full"] = {"raw": feat}
            data["features"] = {}

        # Ensure triage feature exists and is boolean; if not, default to False
        feats_obj = data.get("features")
        if isinstance(feats_obj, dict):
            triage_val = feats_obj.get("triage")
            if not isinstance(triage_val, bool):
                # Record a soft error and coerce to False
                context.stats.errors.append(
                    f"Missing or non-boolean triage feature for {pdf_filename} in node {node_meta.get('relative_path')}"
                )
                feats_obj["triage"] = False
        else:
            # No features dict at all; initialize with triage=False
            context.stats.errors.append(
                f"No features object returned for {pdf_filename} in node {node_meta.get('relative_path')}"
            )
            data["features"] = {"triage": False}

        data["pdf_filename"] = pdf_filename
        data["node_relative_path"] = node_meta.get("relative_path")
        data["topic_mtime"] = topic_mtime
        data["triage_prompt_mtime"] = triage_mtime
        data["pdf_mtime"] = pdf_mtime

        save_paper_json(json_path, data)
        context.stats.papers_processed += 1


def sanitize_dir_name(name: str) -> str:
    allowed = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            allowed.append(ch)
    s = "".join(allowed) or "group"
    return s[:64]


def prepare_child_nodes(
    context: TopicContext,
    node_dir: str,
    node_meta: Dict[str, Any],
    splits: List[Dict[str, Any]],
) -> List[Tuple[Dict[str, Any], str, Any]]:
    """Create child node directories and node_meta for each split.

    Returns a list of (child_meta, child_dir, rule) tuples, where rule is
    a JSON object representing the routing condition for that child.
    """

    children_dir = os.path.join(node_dir, "children")
    ensure_dir(children_dir)

    child_info: List[Tuple[Dict[str, Any], str, Any]] = []

    for split in splits:
        if not isinstance(split, dict):
            continue
        gid = sanitize_dir_name(split.get("group_id", "group"))
        display_name = split.get("display_name", gid)
        rule = split.get("rule", {})

        child_dir = os.path.join(children_dir, gid)
        ensure_dir(child_dir)

        child_meta_path = os.path.join(child_dir, "node_meta.json")
        if os.path.exists(child_meta_path):
            child_meta = load_json(child_meta_path)
        else:
            child_meta = {
                "node_id": gid,
                "relative_path": os.path.relpath(child_dir, context.topic_root).replace(
                    os.sep, "/"
                ),
                "node_type": "leaf",
                "status": "active",
                "triage_prompt_file": node_meta.get("triage_prompt_file"),
                "last_run_signature": {
                    "paper_filenames": [],
                    "hash": "",
                },
            }
        child_meta["display_name"] = display_name
        save_json(child_meta_path, child_meta)

        ensure_dir(os.path.join(child_dir, "papers"))
        ensure_dir(os.path.join(child_dir, "children"))

        # Copy triage prompt file if present
        triage_name = node_meta.get("triage_prompt_file")
        if triage_name:
            parent_triage_path = os.path.join(node_dir, triage_name)
            if os.path.exists(parent_triage_path):
                child_triage_path = os.path.join(child_dir, triage_name)
                if not os.path.exists(child_triage_path):
                    try:
                        import shutil

                        shutil.copy2(parent_triage_path, child_triage_path)
                    except Exception:
                        pass

        child_info.append((child_meta, child_dir, rule))

    return child_info


def route_papers_to_children(
    context: TopicContext,
    node_dir: str,
    node_meta: Dict[str, Any],
    splits: List[Dict[str, Any]],
) -> List[str]:
    """Route non-triaged papers to child nodes based on rules.

    Papers with features.triage == True are left in this node and not
    routed into any child. They are effectively dropped from further
    splitting, but their JSON is preserved.
    """

    if not splits:
        print(f"[topicnode] No splits defined for node {node_meta.get('relative_path')}")
        return []

    child_info = prepare_child_nodes(context, node_dir, node_meta, splits)

    children_dir = os.path.join(node_dir, "children")
    # Optional unclassified node for zero / multi matches
    unclassified_dir = os.path.join(children_dir, "unclassified")
    ensure_dir(unclassified_dir)
    ensure_dir(os.path.join(unclassified_dir, "papers"))
    unclassified_meta_path = os.path.join(unclassified_dir, "node_meta.json")
    if os.path.exists(unclassified_meta_path):
        un_meta = load_json(unclassified_meta_path)
    else:
        un_meta = {
            "node_id": "unclassified",
            "relative_path": os.path.relpath(unclassified_dir, context.topic_root).replace(
                os.sep, "/"
            ),
            "node_type": "leaf",
            "status": "active",
            "triage_prompt_file": node_meta.get("triage_prompt_file"),
            "last_run_signature": {
                "paper_filenames": [],
                "hash": "",
            },
        }
    save_json(unclassified_meta_path, un_meta)

    papers_dir = os.path.join(node_dir, "papers")
    child_dirs: List[str] = []

    for child_meta, child_dir, _ in child_info:
        child_dirs.append(child_dir)

    print(
        f"[topicnode] Routing papers for node {node_meta.get('relative_path')} "
        f"into {len(child_info)} children"
    )

    for name in sorted(os.listdir(papers_dir)):
        if not name.lower().endswith(".json"):
            continue
        json_path = os.path.join(papers_dir, name)
        data = load_paper_json(json_path)

        feats = data.get("features")
        if not isinstance(feats, dict):
            feats = {}

        # Drop triaged papers: leave them in this node and skip routing.
        triage_flag = feats.get("triage")
        if triage_flag is True:
            data["triage"] = True
            save_paper_json(json_path, data)
            continue

        matches: List[Tuple[Dict[str, Any], str]] = []
        for child_meta, child_dir, rule in child_info:
            if not rule:
                continue
            try:
                if evaluate_rule(rule, feats):
                    matches.append((child_meta, child_dir))
            except Exception as exc:
                context.stats.errors.append(
                    f"Rule evaluation error in node {node_meta.get('relative_path')} "
                    f"for child {child_meta.get('node_id')}: {exc}"
                )
                continue

        if len(matches) == 1:
            child_meta, child_dir = matches[0]
            target_dir = os.path.join(child_dir, "papers")
        else:
            # Zero or multiple matches
            context.stats.conflicts.append(
                f"Paper {name} in node {node_meta.get('relative_path')} was "
                f"{'unmatched' if len(matches) == 0 else 'matched by multiple rules'}; "
                f"assigned to unclassified."
            )
            target_dir = os.path.join(unclassified_dir, "papers")

        ensure_dir(target_dir)
        new_path = os.path.join(target_dir, name)
        os.replace(json_path, new_path)

    # After routing, mark this node as internal if it has any real children
    if child_dirs:
        node_meta["node_type"] = "internal"
        node_meta["status"] = "active"
        save_json(os.path.join(node_dir, "node_meta.json"), node_meta)

    return child_dirs


def process_node(context: TopicContext, node_dir: str, is_root: bool = False) -> List[str]:
    """Process a single node and return its child directories.

    Per node:
      - Ensure metadata and directories.
      - Optionally skip work if signature unchanged and status is complete.
      - Ensure we have a triage prompt and split spec.
      - Run extract_paper_features over all papers with that triage prompt.
      - Route non-triaged papers into children based on split rules.

    Triaged papers are left in this node's papers/ directory and ignored
    in routing.
    """

    meta = ensure_node_meta(context, node_dir, is_root=is_root)
    rel = meta.get("relative_path")

    print(f"[topicnode] Processing node {rel} (root={is_root})")

    # Fast-path: if nothing changed and node was previously complete, skip work.
    last_sig = meta.get("last_run_signature") or {"hash": ""}
    current_sig = meta.get("current_signature") or {"hash": ""}
    if (
        meta.get("status") == "complete"
        and last_sig.get("hash")
        and last_sig.get("hash") == current_sig.get("hash")
    ):
        print(f"[topicnode] Node {rel} unchanged; skipping")
        return list_child_nodes(node_dir)

    papers_dir = os.path.join(node_dir, "papers")
    has_papers = any(name.lower().endswith(".json") for name in os.listdir(papers_dir))
    if not has_papers:
        print(f"[topicnode] Node {rel} has no papers; marking empty")
        meta["status"] = "empty"
        meta["last_run_signature"] = current_sig
        save_json(os.path.join(node_dir, "node_meta.json"), meta)
        return []

    # Step 1: ensure triage prompt and split spec
    split_spec, triage_prompt_text = maybe_generate_triage_prompt(context, node_dir, meta)
    if not triage_prompt_text:
        # Without a triage prompt, we cannot extract features or route.
        # Treat as a leaf and mark complete.
        print(
            f"[topicnode] Node {rel} has no triage prompt; treating as leaf and marking complete"
        )
        meta["status"] = "complete"
        meta["last_run_signature"] = current_sig
        save_json(os.path.join(node_dir, "node_meta.json"), meta)
        context.stats.nodes_completed.append(rel or "")
        return []

    splits = split_spec.get("splits") or []

    # Step 2: extract features for all papers in this node
    extract_features_for_all_papers(context, node_dir, meta, triage_prompt_text)

    # Step 3: route non-triaged papers into children based on rules
    child_dirs = route_papers_to_children(context, node_dir, meta, splits)

    # Node is considered complete once routing has been applied
    meta["status"] = "complete"
    meta["last_run_signature"] = current_sig
    save_json(os.path.join(node_dir, "node_meta.json"), meta)
    context.stats.nodes_completed.append(rel or "")

    return child_dirs


# ---------------------------------------------------------------------------
# Top-level orchestration (BFS traversal)
# ---------------------------------------------------------------------------


def run_topic_node(topic_root: str, pdfs_dir: Optional[str] = None) -> RunStats:
    if pdfs_dir is None:
        pdfs_dir = os.path.join(topic_root, "pdfs")
    root_node_dir = os.path.join(topic_root, "root")

    if not os.path.exists(topic_root):
        raise RuntimeError(f"Topic root directory does not exist: {topic_root}")
    if not os.path.isdir(pdfs_dir):
        raise RuntimeError(f"PDF directory does not exist: {pdfs_dir}")
    if not os.path.isdir(root_node_dir):
        raise RuntimeError(f"Root node directory does not exist: {root_node_dir}")

    topic_path = os.path.join(topic_root, "topic.txt")
    topic_text = read_text_required(topic_path, "topic.txt")
    topic_mtime = os.path.getmtime(topic_path)

    stats = RunStats()
    context = TopicContext(
        topic_root=topic_root,
        pdfs_dir=pdfs_dir,
        topic_text=topic_text,
        topic_mtime=topic_mtime,
        stats=stats,
    )

    print(f"[topicnode] Starting run for topic root {topic_root}")

    # Ensure every PDF is represented at the root
    assign_unassigned_pdfs_to_root(
        topic_root=topic_root,
        pdfs_dir=pdfs_dir,
        root_node_dir=root_node_dir,
        stats=stats,
    )

    # BFS over nodes
    queue: deque[Tuple[str, bool]] = deque()
    queue.append((root_node_dir, True))

    while queue:
        node_dir, is_root = queue.popleft()
        child_dirs = process_node(context, node_dir, is_root=is_root)
        for child_dir in child_dirs:
            queue.append((child_dir, False))

    print(
        f"[topicnode] Run complete: {stats.papers_processed} papers processed, "
        f"{len(stats.nodes_completed)} nodes completed, "
        f"{len(stats.conflicts)} routing conflicts, {len(stats.errors)} errors"
    )

    return stats
