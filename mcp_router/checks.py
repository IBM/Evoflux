from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
from jsonschema import Draft202012Validator

from .dsl import ToolRegistry
from .compiler import CompiledWorkflow


@dataclass(frozen=True)
class ConstraintLimits:
    max_nodes: int = 20
    max_edges: int = 30
    max_steps: int = 12  # tool-call budget


@dataclass(frozen=True)
class FeasibilityReport:
    feasible: bool
    reasons: List[str]


class StaticChecker:
    def __init__(self, registry: ToolRegistry, limits: ConstraintLimits):
        self.registry = registry
        self.limits = limits

    def check(self, wf: CompiledWorkflow) -> FeasibilityReport:
        reasons: List[str] = []

        '''if len(wf.nodes) > self.limits.max_nodes:
            reasons.append("too_many_nodes")
        if len(wf.edges) > self.limits.max_edges:
            reasons.append("too_many_edges")
        if wf.policy.max_steps > self.limits.max_steps:
            reasons.append("policy_max_steps_exceeds_limit")'''

        # Tool existence + schema sanity
        for n in wf.nodes:
            if n.kind == "tool":
                if n.tool_ref is None:
                    reasons.append(f"tool_node_missing_ref:{n.node_id}")
                    continue
                s, t = n.tool_ref
                if not self.registry.exists(s, t):
                    reasons.append(f"unknown_tool:{s}.{t}")
                    continue
                spec = self.registry.get(s, t)
                # Validate that params (if any) is at least compatible with schema keys (weak check)
                # Strong check happens at runtime with actual filled args.
                try:
                    Draft202012Validator.check_schema(spec.input_schema)
                    #Draft202012Validator.check_schema(spec.output_schema)
                except Exception:
                    reasons.append(f"bad_schema:{s}.{t}")
        return FeasibilityReport(feasible=(len(reasons) == 0), reasons=reasons)