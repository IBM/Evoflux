"""
mcp_router/adapters/toy.py
--------------------------
Zero-dependency toy adapter.  No API keys, no Docker, no network.
Used for CI, unit tests, and offline development.

The ToyAdapter demonstrates the new query-specific planner model:
  - build_query_planner() returns a ToyQueryPlanner that builds a graph
    for each query from the small (srvA, srvB) toy registry.
  - No static templates are needed.
"""
from __future__ import annotations
import random
from typing import Any, Dict, List, Optional, Tuple

from ..benchmark_adapter import BenchmarkAdapter
from ..dsl import ToolSpec, ToolRegistry, GraphTemplate, Node, Action, PromptParams
from ..executor import ExecResult


class ToyQueryPlanner:
    """
    Deterministic toy planner — no LLM needed.

    For every query it builds the same three-step linear graph:
        srvA.tool1 → srvA.tool2 → srvB.tool3

    This keeps CI fast and deterministic while still exercising the full
    Compiler → Executor → Scorer pipeline.  Real adapters replace this
    with a QueryPlanner that calls an LLM.
    """

    def plan(self, query: Dict[str, Any], registry) -> GraphTemplate:
        import hashlib, json
        nodes = [
            Node(node_id="t1", kind="tool", tool_ref=("srvA", "tool1")),
            Node(node_id="t2", kind="tool", tool_ref=("srvA", "tool2"), requires=["t1"]),
            Node(node_id="t3", kind="tool", tool_ref=("srvB", "tool3"), requires=["t2"]),
        ]
        edges = [("t1", "t2"), ("t2", "t3")]
        qhash = hashlib.sha256(
            json.dumps(query, sort_keys=True, separators=(",",":")).encode()
        ).hexdigest()[:8]
        return GraphTemplate(
            template_id = f"toy_{qhash}",
            query       = query,
            nodes       = nodes,
            edges       = edges,
            description = "toy linear plan: tool1 → tool2 → tool3",
        )


class ToyAdapter(BenchmarkAdapter):
    """Zero-dependency adapter. No API keys, no Docker, no network."""

    def build_registry(self) -> ToolRegistry:
        def obj():
            return {"type": "object", "properties": {"query": {"type": "object"}},
                    "required": ["query"]}
        tools = {}
        for srv in ["srvA", "srvB"]:
            for tool in ["tool1", "tool2", "tool3"]:
                tools[(srv, tool)] = ToolSpec(
                    server=srv, name=tool,
                    input_schema=obj(), output_schema=obj(),
                )
        return ToolRegistry(tools=tools)

    def build_query_planner(self, registry: ToolRegistry):
        # The toy planner is deterministic — no LLM required.
        return ToyQueryPlanner()

    def load_datasets(self):
        queries = [{"id": i, "text": f"task {i}"} for i in range(300)]
        random.Random(0).shuffle(queries)
        n = len(queries)
        return queries[:180], queries[180:240], queries[240:]

    def call_tool(self, server, tool, args):
        return True, None, {"server": server, "tool": tool, "echo": args}

    def build_init_actions(self, D_fb: List[Dict[str, Any]]) -> List[Action]:
        # One base Action (no edits) per query in D_fb.
        # The Compiler will call ToyQueryPlanner.plan(query) on first use.
        rng = random.Random(42)
        actions = []
        for query in D_fb:
            actions.append(Action(
                q=query, e=(),
                p=PromptParams(
                    max_steps=rng.choice([4, 6, 8]),
                    strict_schema=rng.choice([True, False]),
                    verifier_strict=rng.choice([True, False]),
                ),
            ))
        return actions