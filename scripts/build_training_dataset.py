"""
build_training_dataset.py
─────────────────────────
Construct a training dataset from sonnet-4-5 workflow search results.

Positive  : best-scoring feasible candidate per query.
Negatives : structurally diverse hard negatives drawn from the same query's
            feasible pool, filtered to avoid near-duplicates of the positive.

Diversity signal mirrors EvolutionarySearch._prune():
  bucket = int(action_id, 16) % NUM_BUCKETS

Additional diversity axes:
  • node fingerprint  – frozenset of (kind, server, tool) per node
  • edit-op signature – sorted tuple of edit operation names

Outputs (in --out_dir):
  train_sft.jsonl     – instruction-tuning pairs  (prompt / completion)
  train_dpo.jsonl     – preference pairs           (prompt / chosen / rejected)
  train_ranking.jsonl – full per-query record with all negatives + metadata

Usage:
  python scripts/build_training_dataset.py \\
      --history  results/sonnet-4-5/workflow_history.jsonl \\
      --out_dir  datasets/sonnet_workflow_v1 \\
      --max_neg  4 \\
      --min_score_gap 0.5
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

# Allow importing from the project root (mcp_router package)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mcp_router.query_planner import _build_prompt as _qp_build_prompt

# ─────────────────────────── constants ───────────────────────────────────────

NUM_BUCKETS = 16          # matches EvolutionarySearch._prune()

# Matches the system text used in QueryPlanner._call_llm for consistency
SYSTEM_PROMPT = (
    "You are a workflow planner for an LLM tool-use agent. "
    "Given a task and a catalog of MCP tools, you output the "
    "minimal ordered sequence of tool calls needed to complete "
    "the task. You respond with valid JSON only — no prose."
)

# ─────────────────────────── structural helpers ───────────────────────────────

def _node_fingerprint(workflow: Dict[str, Any]) -> FrozenSet[Tuple[str, str, str]]:
    """Canonical structural identity: frozenset of (kind, server, tool) per node."""
    return frozenset(
        (
            str(n.get("kind", "")),
            str(n.get("server", "")),
            str(n.get("tool", "")),
        )
        for n in workflow.get("nodes", [])
        if n.get("kind") != "output"   # synthetic output node carries no signal
    )


def _edit_op_signature(edit_history: List[Dict[str, Any]]) -> Tuple[str, ...]:
    """Sorted tuple of edit operation names – captures structural edit diversity."""
    return tuple(sorted(e.get("op", "") for e in edit_history))


def _action_bucket(action_id: Optional[str]) -> int:
    """Same bucketing used in EvolutionarySearch._prune()."""
    if not action_id:
        return 0
    return int(action_id, 16) % NUM_BUCKETS


# ─────────────────────────── tool catalog sampling ───────────────────────────

def _extract_workflow_tools(workflow: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Return (server, tool) pairs referenced in a workflow (excluding output nodes)."""
    found: List[Tuple[str, str]] = []
    for node in workflow.get("nodes", []):
        if node.get("kind") == "output" or node.get("node_id") == "output":
            continue
        server = node.get("server", "")
        tool = node.get("tool", "")
        tr = node.get("tool_ref")
        if not tool and isinstance(tr, list) and len(tr) >= 2:
            if not server:
                server = tr[0]
            tool = tr[1]
        if server and tool:
            found.append((server, tool))
    return found


def _sample_tool_catalog(
    required_tools: List[Tuple[str, str]],
    full_catalog: Dict[str, Dict[str, Any]],
    rng: random.Random,
    n_min: int = 10,
    n_max: int = 35,
) -> Dict[str, Dict[str, Any]]:
    """
    Build a catalog subset guaranteed to contain all *required_tools*, padded
    with randomly sampled distractors so the total tool count falls in
    [n_min, n_max].

    This produces prompts with varying numbers of tools, forcing the model to
    identify the relevant subset rather than memorising a fixed list.
    """
    required_set = set(required_tools)

    # ── seed result with required tools ──────────────────────────────────────
    result: Dict[str, Dict[str, Any]] = {}
    for server, tool in required_tools:
        if server in full_catalog and tool in full_catalog[server]:
            result.setdefault(server, {})[tool] = full_catalog[server][tool]

    n_required = sum(len(t) for t in result.values())

    # ── build distractor pool: per distractor server sample a random subset ───
    # This prevents a single large server from dominating the catalog while
    # still ensuring representation from many different servers.
    pool: List[Tuple[str, str]] = []
    for srv, tools in full_catalog.items():
        avail = [tl for tl in tools if (srv, tl) not in required_set]
        if not avail:
            continue
        # Sample 1–3 tools from this server (regardless of how many it has)
        k = rng.randint(1, min(3, len(avail)))
        pool.extend((srv, tl) for tl in rng.sample(avail, k))
    rng.shuffle(pool)

    # ── fill to a random target between n_min and n_max ──────────────────────
    target  = rng.randint(n_min, n_max)
    n_extra = min(max(0, target - n_required), len(pool))
    for server, tool in pool[:n_extra]:
        result.setdefault(server, {})[tool] = full_catalog[server][tool]

    return result


# ─────────────────────────── negative selection ───────────────────────────────

def select_diverse_negatives(
    positive: Dict[str, Any],
    pool: List[Dict[str, Any]],
    max_neg: int = 4,
    min_score_gap: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Pick up to *max_neg* hard negatives from *pool* (feasible, non-positive
    candidates for the same query).

    Selection strategy
    ──────────────────
    1. Bucket pool by action_id % NUM_BUCKETS (mirrors _prune()).
    2. Within each bucket keep best-scoring candidates first (hardest negatives).
    3. Round-robin across buckets, skipping:
         • candidates whose node fingerprint matches the positive (near-duplicate)
         • candidates already selected (avoid duplicate fingerprints)
         • candidates where score_gap < min_score_gap (too close to positive)
    4. Prefer buckets whose id differs from the positive's bucket (visited last).

    Returns annotated negative records (adds diversity_bucket, edit_op_signature).
    """
    pos_fp      = _node_fingerprint(positive["workflow"])
    pos_bucket  = _action_bucket(positive.get("action_id"))
    pos_score   = positive["score"]

    # ── bucket the pool ──────────────────────────────────────────────────────
    bins: Dict[int, List[Dict[str, Any]]] = {}
    for c in pool:
        b = _action_bucket(c.get("action_id"))
        bins.setdefault(b, []).append(c)

    # Sort each bin descending by score (hardest negative first within bucket)
    for b in bins:
        bins[b].sort(key=lambda x: x["score"], reverse=True)

    # Visit buckets in an order that de-prioritises the positive's own bucket
    bucket_order = sorted(bins.keys(), key=lambda b: (b == pos_bucket, b))

    seen_fps: set[FrozenSet] = {pos_fp}
    kept: List[Dict[str, Any]] = []

    while len(kept) < max_neg and any(bins.values()):
        progress = False
        for b in bucket_order:
            if not bins[b]:
                continue
            # Scan bucket for the first acceptable candidate
            while bins[b]:
                c = bins[b].pop(0)
                fp = _node_fingerprint(c["workflow"])
                score_gap = pos_score - c["score"]

                if fp in seen_fps:
                    continue                       # near-duplicate → skip
                if score_gap < min_score_gap:
                    continue                       # too close to positive → skip

                seen_fps.add(fp)
                kept.append(
                    {
                        **c,
                        "diversity_bucket": b,
                        "edit_op_signature": list(_edit_op_signature(
                            c.get("edit_history", [])
                        )),
                    }
                )
                progress = True
                break

            if len(kept) >= max_neg:
                break

        if not progress:
            break   # all remaining bins are exhausted or filtered

    return kept


# ─────────────────────────── prompt / completion builders ─────────────────────

def load_tool_catalog(mcpbench_root: str) -> Dict[str, Dict[str, Any]]:
    """Load the full tool catalog from mcp_servers_info.json.

    Returns {server: {tool_name: {"description": ..., "input_schema": ...}}}
    matching the shape QueryPlanner receives at inference time.
    Falls back to an empty dict when the file is not found.
    """
    info_path = Path(mcpbench_root) / "mcp_servers_info.json"
    if not info_path.exists():
        print(
            f"[warn] mcp_servers_info.json not found at {info_path}; "
            "prompts will have an empty catalog section.",
            file=sys.stderr,
        )
        return {}

    with info_path.open(encoding="utf-8") as f:
        raw = json.load(f)

    catalog: Dict[str, Dict[str, Any]] = {}
    for server, details in raw.get("servers", {}).items():
        tools = details.get("tools", {})
        if not tools:
            continue
        catalog[server] = {}
        for tool_name, spec in tools.items():
            if not isinstance(spec, dict):
                catalog[server][tool_name] = {"description": ""}
                continue
            entry: Dict[str, Any] = {"description": spec.get("description", "")}
            if spec.get("input_schema"):
                entry["input_schema"] = spec["input_schema"]
            catalog[server][tool_name] = entry
    return catalog


def _build_prompt(
    query: Dict[str, Any],
    tool_catalog: Dict[str, Dict[str, Any]],
    max_steps: int = 6,
) -> str:
    """Build the user prompt using the same format as QueryPlanner._call_llm."""
    task_text = query.get("text", "").strip()
    return _qp_build_prompt(task_text, tool_catalog, max_steps)


def _build_dpo_prompt(
    query: Dict[str, Any],
    tool_catalog: Dict[str, Dict[str, Any]],
) -> str:
    """Minimal DPO prompt: task + catalog only.

    Strips the Rules block (~330 words) and Output format example (~81 words).
    SFT already drilled syntax and format; DPO only needs per-example signal
    (what to do) and grounding (which tools exist).  Both chosen and rejected
    workflows are format-correct by construction, so repeating the spec burns
    activations across every forward pass for zero learning signal.
    """
    import textwrap as _textwrap
    task_text = query.get("text", "").strip()
    catalog_lines: list = []
    for server in sorted(tool_catalog):
        tools = tool_catalog[server]
        if not tools:
            continue
        catalog_lines.append(f"SERVER: {server}")
        for tool_name, spec in sorted(tools.items()):
            desc = spec.get("description", "") if isinstance(spec, dict) else ""
            if desc:
                desc = _textwrap.shorten(desc, width=120, placeholder="…")
                catalog_lines.append(f"  {tool_name}  —  {desc}")
            else:
                catalog_lines.append(f"  {tool_name}")
        catalog_lines.append("")
    catalog_text = "\n".join(catalog_lines).strip()
    return _textwrap.dedent(f"""
        ## Task
        {task_text}

        ## Available MCP tool servers
        {catalog_text}

        Output JSON only.
    """).strip()


def _strip_empty_description(obj: Any) -> Any:
    """Recursively remove any key whose name is 'description' and value is ''."""
    if isinstance(obj, dict):
        return {
            k: _strip_empty_description(v)
            for k, v in obj.items()
            if not (k == "description" and v == "")
        }
    if isinstance(obj, list):
        return [_strip_empty_description(i) for i in obj]
    return obj


def _normalize_node_ids(workflow: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rename auto-generated or opaque node IDs to semantic names derived from
    the tool they call, so the training data reads naturally.

    Examples
      t_add_46373  (add_tool_step artifact)  →  t_search_movies
      v89652       (insert_validator artifact) →  v_after_t1

    The "output" node and already-clean sequential IDs (t1, t2 …) are also
    normalised to tool-slug names, giving every workflow a consistent style.
    Edges and "requires" lists are rewritten to use the new IDs.
    """
    import re as _re

    nodes = workflow.get("nodes", [])
    edges = workflow.get("edges", [])

    # ── build old → new id mapping ────────────────────────────────────────────
    id_map: Dict[str, str] = {}
    slug_counts: Dict[str, int] = {}

    for node in nodes:
        old_id = node.get("node_id", "")
        kind   = node.get("kind", "tool")

        if kind == "output" or old_id == "output":
            id_map[old_id] = "output"
            continue

        if kind == "validator":
            # Derive name from the node this validator follows (first requires).
            reqs = node.get("requires", [])
            after = reqs[0] if reqs else "node"
            new_id = f"v_after_{after}"
        else:
            # Derive name from the tool name (kind == "tool" or anything else).
            tool_ref = node.get("tool_ref")
            tool_name = node.get("tool", "") or (tool_ref[1] if isinstance(tool_ref, list) and len(tool_ref) > 1 else "")
            if tool_name:
                slug = _re.sub(r"[^a-z0-9]+", "_", tool_name.lower()).strip("_")
            else:
                slug = "step"
            new_id = f"t_{slug}"

        # Deduplicate: t_search, t_search_2, t_search_3, …
        if new_id in slug_counts:
            slug_counts[new_id] += 1
            new_id = f"{new_id}_{slug_counts[new_id]}"
        else:
            slug_counts[new_id] = 1

        id_map[old_id] = new_id

    # Validator nodes reference their predecessor by old ID in requires; now that
    # all tool-node IDs are mapped we can fix up validator names that embedded the
    # old predecessor ID.
    for node in nodes:
        if node.get("kind") == "validator" or node.get("node_id", "").startswith("v"):
            old_id = node.get("node_id", "")
            if old_id in id_map:
                reqs     = node.get("requires", [])
                after_old = reqs[0] if reqs else None
                if after_old and after_old in id_map:
                    id_map[old_id] = f"v_after_{id_map[after_old]}"

    # ── rewrite nodes ─────────────────────────────────────────────────────────
    new_nodes = []
    for node in nodes:
        old_id   = node.get("node_id", "")
        new_node = dict(node)
        new_node["node_id"]  = id_map.get(old_id, old_id)
        new_node["requires"] = [id_map.get(r, r) for r in node.get("requires", [])]
        new_nodes.append(new_node)

    # ── rewrite edges ─────────────────────────────────────────────────────────
    new_edges = [[id_map.get(s, s), id_map.get(d, d)] for s, d in edges]

    result = dict(workflow)
    result["nodes"] = new_nodes
    result["edges"] = new_edges
    return result


def _workflow_to_steps(workflow: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert internal nodes+edges workflow to the steps-array format that matches
    the prompt schema the model is trained to produce.

    Input:  {"nodes": [...], "edges": [[src, dst], ...]}
    Output: {"steps": [{"node_id": ..., "server": ..., "tool": ...,
                         "params": ..., "produces": ..., "requires": [...]}, ...]}

    Nodes are topologically sorted by their dependency order.
    """
    nodes = workflow.get("nodes", [])
    edges = workflow.get("edges", [])

    # Build requires from edges as fallback
    edge_requires: Dict[str, List[str]] = defaultdict(list)
    for edge in edges:
        if len(edge) >= 2:
            edge_requires[edge[1]].append(edge[0])

    node_map: Dict[str, Dict[str, Any]] = {}
    for node in nodes:
        nid = node.get("node_id", "")
        kind = node.get("kind", "tool")
        node_requires = node.get("requires")
        requires = node_requires if node_requires is not None else edge_requires.get(nid, [])

        if kind == "output" or nid == "output":
            step: Dict[str, Any] = {
                "node_id": nid,
                "kind": "output",
                "params": node.get("params", {}),
                "produces": node.get("produces", {}),
                "requires": requires,
            }
        else:
            step = {"node_id": nid}
            server = node.get("server", "")
            tool = node.get("tool", "")
            # Handle tool_ref: ["server", "tool"]
            if not tool:
                tr = node.get("tool_ref")
                if isinstance(tr, list) and len(tr) >= 2:
                    if not server:
                        server = tr[0]
                    tool = tr[1]
            if server:
                step["server"] = server
            if tool:
                step["tool"] = tool
            step["params"] = node.get("params", {})
            step["produces"] = node.get("produces", {})
            step["requires"] = requires

        node_map[nid] = step

    # Topological sort (Kahn's algorithm) to preserve dependency order
    in_degree: Dict[str, int] = {nid: 0 for nid in node_map}
    children: Dict[str, List[str]] = {nid: [] for nid in node_map}
    for nid, step in node_map.items():
        for dep in step["requires"]:
            if dep in node_map:
                in_degree[nid] += 1
                children[dep].append(nid)

    queue = sorted(nid for nid, deg in in_degree.items() if deg == 0)
    ordered: List[str] = []
    while queue:
        nid = queue.pop(0)
        ordered.append(nid)
        for child in sorted(children[nid]):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
                queue.sort()

    # Append any remaining nodes (e.g. cycles)
    seen = set(ordered)
    ordered.extend(nid for nid in node_map if nid not in seen)

    return {"steps": [node_map[nid] for nid in ordered if nid in node_map]}


def _workflow_completion(record: Dict[str, Any]) -> str:
    wf = _normalize_node_ids(record.get("workflow", {}))
    wf = _strip_empty_description(wf)
    wf = _workflow_to_steps(wf)
    return json.dumps(wf, separators=(",", ":"), ensure_ascii=False)


# ─────────────────────────── dataset writers ─────────────────────────────────

def _write_sft(
    examples: List[Dict[str, Any]],
    out_path: Path,
) -> int:
    """
    SFT format: chat-style messages with system / user / assistant turns.
    Compatible with HuggingFace TRL SFTTrainer and most fine-tuning frameworks.

    The user prompt is rebuilt from the per-example sampled_catalog so that
    the catalog visible to the model matches the distractor-sampled subset
    (required workflow tools + random distractors, 20–50 tools total).
    """
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            # Use the pre-sampled catalog when available; fall back to the
            # stored prompt (e.g. legacy examples without sampled_catalog).
            if ex.get("sampled_catalog") is not None:
                user_content = _build_prompt(ex["query_dict"], ex["sampled_catalog"])
            else:
                user_content = ex["prompt"]

            record = {
                "query_id": ex["query_id"],
                "messages": [
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": user_content},
                    {"role": "assistant", "content": ex["positive_completion"]},
                ],
                "positive_score": ex["positive_score"],
                "positive_action_id": ex["positive_action_id"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count



def _shuffle_catalog(
    catalog: Dict[str, Dict[str, Any]],
    rng: random.Random,
) -> Dict[str, Dict[str, Any]]:
    """Return a copy of *catalog* with servers and tools in a random order."""
    servers = list(catalog.keys())
    rng.shuffle(servers)
    result: Dict[str, Dict[str, Any]] = {}
    for srv in servers:
        tools = list(catalog[srv].keys())
        rng.shuffle(tools)
        result[srv] = {t: catalog[srv][t] for t in tools}
    return result


def _write_dpo(
    examples: List[Dict[str, Any]],
    out_path: Path,
    tool_catalog: Optional[Dict[str, Dict[str, Any]]] = None,
    num_dpo_shuffles: int = 3,
    seed: int = 42,
) -> int:
    """
    DPO format: one (prompt, chosen, rejected) triplet per positive-negative pair.
    Compatible with HuggingFace TRL DPOTrainer.

    Catalog variants are collapsed to one record per base query — distractor
    variation adds negligible preference signal and triples data volume for free.
    The prompt catalog contains only the exact tools referenced in the positive
    and this specific negative workflow (no distractors, no cross-negative leakage).

    For each (positive, negative) pair, *num_dpo_shuffles* records are written,
    each with the server and tool order independently shuffled.  This multiplies
    DPO pairs without adding any distractors or changing the preference signal.
    """
    rng = random.Random(seed)
    count = 0
    seen_base_ids: set = set()
    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            # Deduplicate catalog variants — one record per base query is enough.
            base_qid = ex["query_id"].rsplit("_v", 1)[0]
            if base_qid in seen_base_ids:
                continue
            seen_base_ids.add(base_qid)

            pos_wf = ex.get("positive_workflow", {})
            pos_tools = set(_extract_workflow_tools(pos_wf))
            chosen_completion = ex["positive_completion"]

            for neg in ex["negatives"]:
                # Build catalog from only this pair's tools — positive + this
                # specific negative.  Union across all negatives would leak tools
                # from unrelated pairs and inflate the prompt.
                pair_tools = pos_tools | set(_extract_workflow_tools(neg.get("workflow", {})))
                if tool_catalog:
                    exact_catalog: Dict[str, Dict[str, Any]] = {}
                    for server, tool in pair_tools:
                        if server in tool_catalog and tool in tool_catalog[server]:
                            exact_catalog.setdefault(server, {})[tool] = tool_catalog[server][tool]
                else:
                    exact_catalog = {}

                rejected_completion = _workflow_completion(neg)

                for shuffle_idx in range(num_dpo_shuffles):
                    shuffled = _shuffle_catalog(exact_catalog, rng)

                    record = {
                        "query_id":            f"{base_qid}_s{shuffle_idx}",
                        "prompt":              SYSTEM_PROMPT + "\n\n" + _build_dpo_prompt(ex["query_dict"], shuffled),
                        "chosen":              chosen_completion,
                        "rejected":            rejected_completion,
                        "chosen_score":        ex["positive_score"],
                        "rejected_score":      neg["score"],
                        "score_gap":           round(ex["positive_score"] - neg["score"], 4),
                        "diversity_bucket":    neg["diversity_bucket"],
                        "edit_op_signature":   neg.get("edit_op_signature", []),
                        "chosen_action_id":    ex["positive_action_id"],
                        "rejected_action_id":  neg.get("action_id", ""),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
    return count


def _write_ranking(
    examples: List[Dict[str, Any]],
    out_path: Path,
) -> int:
    """
    Ranking format: one record per query with all negatives and full metadata.
    Suitable for listwise ranking losses or custom training loops.
    """
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            record = {
                "query_id":   ex["query_id"],
                "prompt":     ex["prompt"],
                "positive": {
                    "action_id":       ex["positive_action_id"],
                    "candidate_id":    ex["positive_candidate_id"],
                    "score":           ex["positive_score"],
                    "score_detail":    ex["positive_score_detail"],
                    "workflow":        _workflow_to_steps(_normalize_node_ids(ex["positive_workflow"])),
                    "edit_history":    ex["positive_edit_history"],
                    "edit_op_signature": list(_edit_op_signature(
                        ex.get("positive_edit_history", [])
                    )),
                    "diversity_bucket": _action_bucket(ex.get("positive_action_id")),
                    "completion":      ex["positive_completion"],
                },
                "negatives": [
                    {
                        "action_id":       neg.get("action_id", ""),
                        "candidate_id":    neg.get("candidate_id", ""),
                        "score":           neg["score"],
                        "score_detail":    neg.get("score_detail", {}),
                        "workflow":        _workflow_to_steps(_normalize_node_ids(neg["workflow"])),
                        "edit_history":    neg.get("edit_history", []),
                        "edit_op_signature": neg.get("edit_op_signature", []),
                        "diversity_bucket":  neg["diversity_bucket"],
                        "score_gap":       round(ex["positive_score"] - neg["score"], 4),
                        "completion":      _workflow_completion(neg),
                    }
                    for neg in ex["negatives"]
                ],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


# ─────────────────────────── main pipeline ───────────────────────────────────

def load_history(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load workflow_history.jsonl grouped by query_id."""
    by_query: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_query[rec["query_id"]].append(rec)
    return dict(by_query)


def build_examples(
    by_query: Dict[str, List[Dict[str, Any]]],
    max_neg: int,
    min_score_gap: float,
    tool_catalog: Optional[Dict[str, Dict[str, Any]]] = None,
    num_catalog_variants: int = 3,
    seed: int = 42,
    catalog_n_min: int = 10,
    catalog_n_max: int = 25,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Build per-query training examples.  Returns (examples, stats).

    For each query we generate *num_catalog_variants* copies of the example,
    each with a different randomly sampled tool catalog (always containing the
    tools referenced in the positive + negative workflows, plus random
    distractors to reach a total tool count of 20–50).  This multiplies the
    number of training datapoints while teaching the model to handle varying
    and partially irrelevant tool lists.
    """
    rng = random.Random(seed)

    examples: List[Dict[str, Any]] = []
    stats = {
        "queries_total":    len(by_query),
        "queries_used":     0,
        "queries_skipped":  0,
        "total_negatives":  0,
        "neg_per_query":    defaultdict(int),
    }

    for qid, records in by_query.items():
        feasible = [r for r in records if r.get("feasible", False)
                    and r.get("workflow") is not None]

        if len(feasible) < 2:
            stats["queries_skipped"] += 1
            continue

        # ── pick positive ────────────────────────────────────────────────────
        best = max(feasible, key=lambda r: r["score"])

        # ── pool = everything except the positive ────────────────────────────
        pool = [r for r in feasible if r["candidate_id"] != best["candidate_id"]]

        # ── select diverse hard negatives ────────────────────────────────────
        negatives = select_diverse_negatives(
            positive=best,
            pool=pool,
            max_neg=max_neg,
            min_score_gap=min_score_gap,
        )

        if not negatives:
            stats["queries_skipped"] += 1
            continue

        query_dict = best.get("action", {}).get("q", {})
        positive_completion = _workflow_completion(best)

        # ── collect tools that MUST appear in every catalog variant ──────────
        # (all tools referenced across the positive + every negative workflow)
        required_tools: List[Tuple[str, str]] = list({
            (srv, tl)
            for wf in [best["workflow"]] + [n["workflow"] for n in negatives]
            for srv, tl in _extract_workflow_tools(wf)
        })

        # ── generate catalog variants ─────────────────────────────────────────
        for variant_idx in range(num_catalog_variants):
            if tool_catalog:
                sampled_catalog = _sample_tool_catalog(
                    required_tools, tool_catalog, rng,
                    n_min=catalog_n_min, n_max=catalog_n_max,
                )
            else:
                sampled_catalog = {}

            prompt = _build_prompt(query_dict, sampled_catalog)

            examples.append(
                {
                    "query_id":               f"{qid}_v{variant_idx}",
                    "prompt":                 prompt,
                    "query_dict":             query_dict,
                    "sampled_catalog":        sampled_catalog,
                    "positive_action_id":     best.get("action_id", ""),
                    "positive_candidate_id":  best.get("candidate_id", ""),
                    "positive_score":         best["score"],
                    "positive_score_detail":  best.get("score_detail", {}),
                    "positive_workflow":      best["workflow"],
                    "positive_edit_history":  best.get("edit_history", []),
                    "positive_completion":    positive_completion,
                    "negatives":              negatives,
                }
            )

        stats["queries_used"]    += 1
        stats["total_negatives"] += len(negatives)
        stats["neg_per_query"][len(negatives)] += 1

    return examples, stats


def print_stats(stats: Dict[str, Any], n_sft: int, n_dpo: int, n_rank: int) -> None:
    neg_dist = stats["neg_per_query"]
    print(f"\n{'─'*55}")
    print(f"  Queries total          : {stats['queries_total']}")
    print(f"  Queries used           : {stats['queries_used']}")
    print(f"  Queries skipped        : {stats['queries_skipped']}")
    print(f"  Total negatives chosen : {stats['total_negatives']}")
    print(f"  Negatives distribution : {dict(sorted(neg_dist.items()))}")
    print(f"  SFT examples written   : {n_sft}")
    print(f"  DPO pairs written      : {n_dpo}")
    print(f"  Ranking examples       : {n_rank}")
    print(f"{'─'*55}\n")


# ─────────────────────────── CLI entry point ─────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build SFT / DPO / Ranking training datasets from sonnet workflow results."
    )
    p.add_argument(
        "--history",
        default="results/sonnet-4-5/workflow_history.jsonl",
        help="Path to workflow_history.jsonl (default: results/sonnet-4-5/workflow_history.jsonl)",
    )
    p.add_argument(
        "--out_dir",
        default="datasets/sonnet_workflow_v1",
        help="Output directory (default: datasets/sonnet_workflow_v1)",
    )
    p.add_argument(
        "--max_neg",
        type=int,
        default=4,
        help="Maximum hard negatives per query (default: 4)",
    )
    p.add_argument(
        "--min_score_gap",
        type=float,
        default=1.0,
        help="Minimum score gap between positive and negative (default: 1.0)",
    )
    p.add_argument(
        "--formats",
        nargs="+",
        choices=["sft", "dpo", "ranking", "all"],
        default=["all"],
        help="Output formats to write (default: all)",
    )
    p.add_argument(
        "--mcpbench_root",
        default="mcp-bench",
        help="Path to MCPBench root dir containing mcp_servers_info.json (default: mcp-bench)",
    )
    p.add_argument(
        "--num_catalog_variants",
        type=int,
        default=3,
        help=(
            "Number of catalog-sampled variants to generate per query. "
            "Each variant has the same workflows but a different random subset "
            "of tools (required tools + distractors, 20–50 tools total). "
            "Multiplies total datapoints by this factor. (default: 3)"
        ),
    )
    p.add_argument(
        "--num_dpo_shuffles",
        type=int,
        default=3,
        help=(
            "Number of server/tool-order shuffles per (positive, negative) DPO pair. "
            "Each shuffle produces a distinct prompt with identical content but different "
            "catalog ordering, multiplying DPO pairs without adding distractors. (default: 3)"
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for catalog sampling and DPO shuffles (default: 42)",
    )
    p.add_argument(
        "--catalog_n_min",
        type=int,
        default=10,
        help="Minimum total tools in each sampled catalog (default: 10)",
    )
    p.add_argument(
        "--catalog_n_max",
        type=int,
        default=25,
        help="Maximum total tools in each sampled catalog (default: 35)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    history_path = Path(args.history)
    if not history_path.exists():
        print(f"[error] History file not found: {history_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    want = set(args.formats)
    write_all = "all" in want

    print(f"[load]  Reading {history_path} …")
    by_query = load_history(history_path)
    print(f"        {sum(len(v) for v in by_query.values())} records, "
          f"{len(by_query)} queries")

    tool_catalog = load_tool_catalog(args.mcpbench_root)
    print(f"[catalog] {sum(len(v) for v in tool_catalog.values())} tools across "
          f"{len(tool_catalog)} servers loaded from {args.mcpbench_root}")

    print(
        f"[build] max_neg={args.max_neg}  min_score_gap={args.min_score_gap}  "
        f"num_catalog_variants={args.num_catalog_variants}  "
        f"num_dpo_shuffles={args.num_dpo_shuffles}  seed={args.seed}  "
        f"catalog_tools=[{args.catalog_n_min}, {args.catalog_n_max}]"
    )
    examples, stats = build_examples(
        by_query,
        args.max_neg,
        args.min_score_gap,
        tool_catalog,
        num_catalog_variants=args.num_catalog_variants,
        seed=args.seed,
        catalog_n_min=args.catalog_n_min,
        catalog_n_max=args.catalog_n_max,
    )

    n_sft = n_dpo = n_rank = 0

    if write_all or "sft" in want:
        n_sft = _write_sft(examples, out_dir / "train_sft.jsonl")
        print(f"[write] train_sft.jsonl     → {n_sft} examples")

    if write_all or "dpo" in want:
        n_dpo = _write_dpo(
            examples, out_dir / "train_dpo.jsonl",
            tool_catalog=tool_catalog,
            num_dpo_shuffles=args.num_dpo_shuffles,
            seed=args.seed,
        )
        print(f"[write] train_dpo.jsonl     → {n_dpo} pairs")

    if write_all or "ranking" in want:
        n_rank = _write_ranking(examples, out_dir / "train_ranking.jsonl")
        print(f"[write] train_ranking.jsonl → {n_rank} examples")

    print_stats(stats, n_sft, n_dpo, n_rank)
    print(f"[done]  Dataset written to {out_dir}/")


if __name__ == "__main__":
    main()



'''
python scripts/build_training_dataset.py \
    --history  ./results/sonnet-4-5/workflow_history.jsonl \
    --out_dir  datasets/sonnet_workflow_v1 \
    --mcpbench_root mcp-bench \
    --max_neg 4 --min_score_gap 1.0

'''