from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from topic_node.openai_requests import run_pdf_prompt, run_text_prompt


class LLMError(RuntimeError):
    """Raised when the underlying LLM call fails or returns invalid JSON."""


def _call_llm(prompt_template: str, source: Any) -> Dict[str, Any]:
    """Call the LLM helper and return a JSON object.

    For per-paper calls, ``source`` should be a PDF file path and the model
    is called with that file attached. For node-level calls, ``source`` is
    treated as context text (or a JSON-like object) that is appended to the
    prompt and sent as plain text.
    """
    try:
        # Per-paper: path to a PDF on disk
        if isinstance(source, str) and source.lower().endswith(".pdf"):
            data = run_pdf_prompt(source, prompt_template)
        else:
            # Node-level: treat source as JSON-like context or text
            if isinstance(source, (dict, list)):
                context = json.dumps(source, indent=2, sort_keys=True)
            else:
                context = str(source)

            full_prompt = f"{prompt_template}\n\nInput context:\n{context}"
            data = run_text_prompt(full_prompt)
    except Exception as exc:  # pragma: no cover - depends on model/runtime
        raise LLMError(f"LLM call failed: {exc}") from exc

    if not isinstance(data, dict):
        raise LLMError("LLM helper did not return a JSON object")

    return data


# ---------------------------------------------------------------------------
# Per-paper calls
# ---------------------------------------------------------------------------


def characterize_paper(
    pdf_path: str,
    topic_text: str,
    node_description: str,
) -> Dict[str, Any]:
    """Per-paper explanatory characterization used during sampling.

    This function is invoked when the system needs to build compact
    representations of papers for node-level sampling and split proposal.
    It reads the given paper (via the user's LLM helper), combines it with
    the global topic description and the current node description, and
    asks the model to explain how the paper relates to the topic. The
    output is later used to generate sampled summaries and to guide
    split-rule proposals.
    """

    prompt = f"""You are assisting with a structured literature review.

Global topic and goals:
{topic_text}

Current node description:
{node_description}

You are given the full text of a single paper as your input. Explain how
this paper relates to the topic and the node-level focus. Return a single JSON object with the following keys:

- short_explanation: a 3–5 sentence plain-language description of how this paper relates to the topic and node.
- relevance_score: a string, one of "high", "medium", or "low".
- notes: a list of short strings. Each entry should capture one specific statement or detail from the paper that is directly relevant to the topic

When possible, make each note a concise quote from the paper (in double quotes) or a close paraphrase, clearly identifying what it refers to.

Return JSON only, with no markdown and no commentary."""

    return _call_llm(prompt, pdf_path)


def extract_paper_features(
    pdf_path: str,
    topic_text: str,
    node_description: str,
    triage_prompt_text: str,
) -> Dict[str, Any]:
    """Per-paper feature extraction driven by a node-specific triage prompt.

    This function is used during triage to attach structured, machine-
    interpretable features to each paper in a node. The PDF path is
    passed directly to the user's LLM helper, together with the global
    topic text, the current node description, and a node-specific triage
    prompt that defines the feature schema (including a required boolean
    feature named "triage").

    The model is expected to return a JSON object that includes a top-
    level key "features" whose value is another object mapping feature
    names to values. These feature fields are later used for filtering,
    grouping, and evaluating split rules in the node tree.
    """

    if not triage_prompt_text:
        raise LLMError(
            "extract_paper_features was called without a triage prompt. "
            "Each node must define a triage prompt file before feature "
            "extraction can run."
        )

    triage_section = triage_prompt_text

    prompt = f"""You are assisting with a structured literature review.

Global topic and goals:
{topic_text}

Current node description:
{node_description}

Node-specific triage instructions:
{triage_section}

You are given the full text of a single paper as your input. Follow the
triage instructions above exactly.

Return a single JSON object. It must contain a top level key "features"
whose value is another JSON object containing the named feature fields
as described in the triage instructions. The features object must
include a boolean field "triage". Set triage to true if the paper is
clearly out of scope for the topic or irrelevant to this node, otherwise
set triage to false.

Return JSON only, with no markdown and no commentary."""

    return _call_llm(prompt, pdf_path)


# ---------------------------------------------------------------------------
# Node-level calls
# ---------------------------------------------------------------------------


def propose_node_splits(
    sampled_papers: List[Dict[str, Any]],
    topic_text: str,
    node_description: str,
) -> Dict[str, Any]:
    """Summarize and propose a feature schema and topic-aligned splits for a node.

    This function is called when the system needs to infer a useful set of
    features and binning rules for a node from a sample of its papers. It
    receives a list of compact per-paper representations produced from the
    per-paper JSON files (for example, including pdf_filename and an
    explanatory object from characterize_paper).

    The model is asked to:
      1. Summarize how this sampled set of papers relates to the global
         topic and the current node description.
      2. Propose a small "bag of features" (between 4 and 15) that would be
         useful for filtering and grouping all papers in this node. Each
         feature should correspond to a concrete, checkable property of a
         paper that relates to the overall topic and the current node
         description.
      3. Using only those features, propose a small number of groups (2–5)
         that will be used as bins for papers in this node.

    The returned JSON is later consumed by the engine, which will add an
    additional hard-coded boolean feature named "triage" and will construct
    the actual prompt used by extract_paper_features.
    """
    if not sampled_papers:
        raise LLMError("No sampled papers provided for split proposal")

    context = sampled_papers

    prompt = f"""You are assisting with a structured literature review that will
be organized as a tree of topic nodes.

Global topic and goals:
{topic_text}

Current node description:
{node_description}

You are given a JSON array of compact representations for a sample of
papers in this node. These already reflect the global topic and the
current node description, but there is not yet a fully defined feature
schema.

Your tasks:
1. Summarize how this sampled set of papers relates to the overall topic and node description.
2. Propose a small "bag of features" (between 4 and 15) that would be
   useful for filtering and grouping all papers in this node. Each
   feature should correspond to a concrete, checkable property of a
   paper that relates to the overall topic and the current node
   description.
3. Using only those features, propose a small number of groups (between
   2 and 5) that would be useful bins for the papers in this node.

Return a single JSON object with:
- sampled_summary: textual summary of the sampled set.
- features: a list of feature definition objects. Each feature object must have:
  - name: short identifier string (suitable as a JSON key).
  - description: human-readable explanation of what this feature captures.
  - value_type: one of "boolean", "enum", "number", or "string".
  - allowed_values: for enum features, a list of allowed string values
    (omit or use [] for non-enum types).
  - instructions: concrete instructions to the model on how to infer and
    fill in this feature from the paper.
- splits: a list of group objects describing how to bin papers using
  these features. Each group object must have:
  - group_id: short identifier string suitable as a directory name, using
    only letters, numbers, hyphen and underscore.
  - display_name: human readable group name.
  - rule: a JSON object that describes the logical condition under which
    a paper belongs to this group. The rule language is:
      * Logical connectives:
        - An object with key "all" whose value is an array of sub-rules
          means all of the sub-rules must be true (logical AND).
        - An object with key "any" whose value is an array of sub-rules
          means at least one sub-rule must be true (logical OR).
        - An object with key "not" whose value is a single sub-rule means
          the sub-rule must be false (logical NOT).
      * Comparisons:
        - An object with key "eq" whose value is a two-element array
          [feature_name, value] means the feature must be equal to value.
        - An object with key "neq" whose value is a two-element array
          [feature_name, value] means the feature must not be equal to
          value.
        - An object with key "in" whose value is a two-element array
          [feature_name, value_list] means the feature's value must be one
          of the values in value_list.
      * Feature names must match the "name" fields in the features list.
      * Values must be valid for the corresponding feature's value_type
        and allowed_values.
  - rationale: explanation string for the user.

Do not include any feature whose name is "triage"; the engine will add a
triage feature separately.

Return JSON only, with no markdown and no commentary."""
    return _call_llm(prompt, context)


def summarize_node_from_papers(
    leaf_papers: List[Dict[str, Any]],
    topic_text: str,
    node_description: str,
) -> Dict[str, Any]:
    """Summarize a leaf node using compact per-paper representations.

    This function is called once a node has been fully split (if needed)
    and the number of papers in the node is below the leaf-size
    threshold. At this point, each paper already has both explanatory
    fields (from characterize_paper) and structured features (from
    extract_paper_features). The engine therefore passes a compact
    representation of every paper in the node.

    The model is asked to integrate these representations into a
    structured summary of what this leaf node "means" in the context of
    the overall topic and the node's position in the tree. This summary
    becomes the stable, reviewable output for the node and is later used
    by internal-node summarization to build higher-level synthesis.

    The returned JSON must include:
      - summary_text: a narrative synthesis of the papers in this node.
      - key_findings: short bullet-like statements capturing important
        positive or directional conclusions supported by the papers.
      - limitations: short bullet-like statements describing important
        constraints, gaps, or limitations observed across the papers.
      - citations: a list of pdf_filename strings that support the
        summary and findings.
    """

    context = leaf_papers

    prompt = f"""You are assisting with a structured literature review organized
as a tree.

Global topic and goals:
{topic_text}

Current node description:
{node_description}

You are given a JSON array where each element describes one paper assigned
to this node with:
- pdf_filename
- short_explanation
- relevance_score
- features

Produce a JSON summary of what the papers in this node collectively say,
framed in terms of the topic. Include:
- summary_text: detailed narrative summary.
- key_findings: list of short bullet text strings.
- limitations: list of short bullet text strings describing important constraints, gaps, or unsuccessful attempts reported across these papers.
- citations: list of pdf_filename strings that should be cited for the
  findings above.

Return a single JSON object with these keys. Return JSON only, with no
markdown and no commentary."""

    return _call_llm(prompt, context)


def summarize_node_from_child_summaries(
    child_summaries: List[Dict[str, Any]],
    topic_text: str,
    node_description: str,
) -> Dict[str, Any]:
    """Summarize an internal node from its child node summaries.

    This function is used for hierarchical summarization once leaf nodes
    already have their own node-level summaries. For an internal node,
    the engine collects the existing summaries of each child node and
    passes them here as a list of JSON-like objects. Each child summary is
    expected to include:
      - node_relative_path: the child's path within the topic tree.
      - summary_text: narrative summary of that child node.
      - key_findings: list of key findings at the child level.
      - limitations: list of important limitations, gaps, or constraints
        identified at the child level.
      - citations: list of pdf_filename strings associated with the child
        node's findings.

    The model is asked to integrate these child summaries into a higher-
    level summary for the current internal node. This produces a
    structured summary that can itself be used as input when summarizing
    ancestors, enabling recursive, tree-shaped synthesis.
    """

    context = child_summaries

    prompt = f"""You are assisting with a structured literature review organized
as a tree.

Global topic and goals:
{topic_text}

Current node description:
{node_description}

You are given a JSON array where each element is a summary for a child
node of this node. Each child summary object contains:
- node_relative_path
- summary_text
- key_findings
- limitations
- citations

Produce a JSON summary for the current node that:
- Integrates and synthesizes the findings across children.
- Provides a narrative summary_text for this node.
- Provides merged lists of key_findings and limitations.
- Provides a citations list that is the union of all child citations,
  without duplicates.

Return a single JSON object with keys:
- summary_text
- key_findings
- limitations
- citations

Return JSON only, with no markdown and no commentary."""

    return _call_llm(prompt, context)