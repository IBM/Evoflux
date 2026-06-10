from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Literal
import hashlib
import json

NodeType = Literal["tool", "validator", "router_hint", "cache", "output"]


# ── Registry ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ToolSpec:
    server: str
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any] | None = None
    metadata: Dict[str, Any][str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolRegistry:
    tools: Dict[Tuple[str, str], ToolSpec]

    def get(self, server: str, tool: str) -> ToolSpec:
        return self.tools[(server, tool)]

    def exists(self, server: str, tool: str) -> bool:
        return (server, tool) in self.tools


# ── Graph nodes ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Node:
    node_id: str
    kind: NodeType
    tool_ref: Optional[Tuple[str, str]] = None      # (server, tool_name)
    params: Dict[str, Any] = field(default_factory=dict)
    requires: List[str] = field(default_factory=list)   # node_ids that must run first
    produces: Dict[str, str] = field(default_factory=dict)


# ── Query-specific workflow graph ─────────────────────────────────────────────

@dataclass(frozen=True)
class GraphTemplate:
    """
    A fully-specified, executable workflow graph for ONE query.

    This is NOT a reusable skeleton.  The Compiler builds one per query by
    asking the LLM planner: "given this task, what tools are needed and in
    what order?"  The result covers every action required to complete that
    specific task.

    The Executor walks `nodes` in topological order (already sorted by the
    Compiler), calling each tool in turn, and accumulates results in a
    context dict that later nodes can read from.
    """
    template_id: str                        # stable hash of (query, plan)
    nodes: List[Node]                       # topological order, enforced by Compiler
    edges: List[Tuple[str, str]]            # (src_node_id, dst_node_id)
    description: str = ""                   # LLM's plain-English summary of the plan
    original_text: str = ""                 # original text of LLM after parsing
    token_cost: Dict[str, Any] = None       # this contains how much token cost has been spent for this Graph
    heuristic: bool = False                 # True when planned by heuristic fallback (no LLM)


    def complexity(self) -> int:
        return len(self.nodes) + len(self.edges)


# ── Execution policy ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PromptParams:
    retry_count: int = 1
    backoff_ms: int = 200
    strict_schema: bool = True
    verifier_strict: bool = True
    max_steps: int = 8


# ── Structural edits ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TypedEdit:
    """
    A targeted structural change to a query-specific workflow graph.

    The evolutionary search applies these to discover better workflow variants
    for a given query.  Edits are always relative to the base plan the LLM
    produced for that query.

    Supported operations
    --------------------
    swap_tool
        Replace the tool on an existing node with a different (server, tool).
        args: {node_id, new_server, new_tool}

    insert_validator
        Insert a schema-check node immediately after an existing node.
        args: {after_node_id, validator_id}

    set_param
        Change a node-level execution parameter (e.g. strict_schema=False).
        args: {node_id, key, value}

    add_tool_step
        Append a new tool-call node, optionally after a specified node.
        args: {node_id, server, tool, after_node_id (optional)}

    remove_tool_step
        Delete a node and all edges that reference it.
        args: {node_id}

    reorder_step
        Change the dependency list of a node (its requires field), effectively
        moving it earlier or later in the execution sequence.
        args: {node_id, new_requires: List[str]}
    """
    op: str
    args: Dict[str, Any]


# ── Action ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Action:
    """
    An Action is a (query, edits, policy) triple.

    The Compiler takes an Action and:
      1. Calls the LLM planner to produce a base GraphTemplate for `q`.
      2. Applies the TypedEdits in `e` on top of that base graph.
      3. Returns a CompiledWorkflow the Executor can run directly.

    Two Actions for the same query but different edits represent different
    workflow variants competing in the evolutionary population.  An empty
    edit list means "run the LLM's plan exactly as proposed".
    """
    q: Dict[str, Any]                       # the query this action is for
    e: Tuple[TypedEdit, ...] = ()           # structural edits on top of base plan
    p: PromptParams = field(default_factory=PromptParams)
    graph: GraphTemplate   = None           # graph template of the action
    token_cost: Dict[str, Any]    = None    # token spent to create this action
    error_context: Optional[str]  = None   # last known failure reason; used by mutation LLM
    def stable_id(self) -> str:
        """Deterministic hash over (query, edits, policy, base plan)."""
        payload = {
            "q": self.q,
            "e": [{"op": x.op, "args": x.args} for x in self.e],
            "p": self.p.__dict__,
            "g": self.graph.template_id if self.graph is not None else None,
        }
        s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]