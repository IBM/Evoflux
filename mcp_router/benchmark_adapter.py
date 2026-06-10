"""
mcp_router/benchmark_adapter.py
--------------------------------
Abstract BenchmarkAdapter interface.

Architecture change from v3
----------------------------
Templates are now query-specific, not static skeletons.
The adapter no longer implements build_templates().  Instead it
implements build_query_planner() which returns a QueryPlanner — the
object the Compiler uses to generate a bespoke graph for each query.

Concrete adapters live in:
    mcp_router/adapters/assetopsbench.py
    mcp_router/adapters/mcpbench.py
    mcp_router/adapters/toy.py
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .dsl import ToolRegistry, Action
from .executor import ExecResult, AdaptedExecutor, ScoringFn, MetricBreakdown
from .compiler import CompiledWorkflow
#from .metrics import compute_metrics, scalarize

# ── What every adapter must supply ────────────────────────────────────────────

class BenchmarkAdapter(ABC):

    # ------------------------------------------------------------------
    # 1. Tool inventory
    # ------------------------------------------------------------------
    @abstractmethod
    def build_registry(self) -> ToolRegistry:
        """Return a ToolRegistry of all tools this benchmark exposes."""

    # ------------------------------------------------------------------
    # 2. Query planner (replaces build_templates)
    # ------------------------------------------------------------------
    @abstractmethod
    def build_query_planner(self, registry: ToolRegistry):
        """
        Return a QueryPlanner for this benchmark.

        The QueryPlanner is called by the Compiler once per unique query.
        It takes (query, registry) and returns a GraphTemplate — a fully
        specified, executable graph of the tool calls needed to complete
        that specific task.

        The simplest implementation:
            from mcp_router.query_planner import QueryPlanner
            return QueryPlanner(
                registry=registry,
                tool_catalog=self._build_tool_catalog(),
                openrouter_api_key=self.openrouter_api_key,
            )
        """

    # ------------------------------------------------------------------
    # 3. Dataset
    # ------------------------------------------------------------------
    @abstractmethod
    def load_datasets(self) -> Tuple[
        List[Dict[str, Any]],   # D_fb   — minibatch feedback set
        List[Dict[str, Any]],   # D_val  — held-out validation set
    ]:
        """
        Load and split task/scenario dicts.
        Each dict is passed verbatim as `query` to MCPExecutor.run().
        """

    # ------------------------------------------------------------------
    # 4. Tool execution
    # ------------------------------------------------------------------
    @abstractmethod
    def call_tool(
        self,
        server: str,
        tool: str,
        args: Dict[str, Any],
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """Execute one tool call. Return (ok, error_or_None, output_or_None)."""

    # ------------------------------------------------------------------
    # 5. (Optional) Scoring override
    # ------------------------------------------------------------------
    def score_result(self, res: ExecResult, wf: CompiledWorkflow) -> Optional[MetricBreakdown]:
        """Override with a real LLM / rule-based judge. None → structural proxy."""
        return None

    # ------------------------------------------------------------------
    # 6. (Optional) Initial actions seed
    # ------------------------------------------------------------------
    def build_init_actions(
        self,
        D_fb: List[Dict[str, Any]],
    ) -> List[Action]:
        """
        Seed the evolutionary population.

        Default: one base Action (no edits) per query in D_fb.
        Each Action carries its query — the Compiler will plan a graph for
        it on first use.  Override to seed in a domain-aware way.
        """
        from .dsl import PromptParams
        return [
            Action(q=query, e=(), p=PromptParams(max_steps=8))
            for query in D_fb
        ]

    # ------------------------------------------------------------------
    # Internal — builds ScoringFn for AdaptedExecutor
    # ------------------------------------------------------------------
    def _make_score_fn(self) -> ScoringFn:
        adapter = self

        def _score(res: ExecResult, wf: CompiledWorkflow) -> MetricBreakdown:
            override = adapter.score_result(res, wf)
            if override is not None:
                return override
            #Manually scoring the result
            raise "Score_result() is not defined within the adapter. Ensure evaluation method exists."
            return scalarize(compute_metrics(res))

        return _score


# ── Runner ────────────────────────────────────────────────────────────────────

@dataclass
class RunnerConfig:
    B: int = 40
    K: int = 12
    #rank_repeats: int = 2
    max_steps: int = 15
    alpha: float = 0.05
    seed: int = 0
    timeout_s: float = 30.0
    output_dir: str = "results"
    plan_retries: int = 5  # max LLM re-plan attempts per query on execution failure


class BenchmarkAdapterRunner:
    """
    Wires a BenchmarkAdapter into the full mcp_router pipeline.

    Construction sequence
    ---------------------
    1. build_registry()        → ToolRegistry
    2. build_query_planner()   → QueryPlanner
    3. Compiler(registry, planner)  — no static templates
    4. build_init_actions(D_fb)     — one Action per query, no template_id
    """

    def __init__(self, adapter: BenchmarkAdapter, cfg: RunnerConfig = RunnerConfig()):
        self.adapter = adapter
        self.cfg     = cfg
        self.registry = adapter.build_registry()
        self.planner       = adapter.build_query_planner(self.registry)

        self._score_fn: ScoringFn = adapter._make_score_fn()
        self._executor_factory    = self._make_executor_factory(perturbation_scale=1.0)

        from .compiler import Compiler
        from .checks import StaticChecker, ConstraintLimits

        self.compiler = Compiler(registry=self.registry)
        self.checker  = StaticChecker(
            registry=self.registry,
            limits=ConstraintLimits(max_steps=cfg.max_steps + 4),
        )

    # ------------------------------------------------------------------

    def _make_executor_factory(
        self,
        perturbation_scale: float = 1.0,
    ) -> Callable[[int], AdaptedExecutor]:
        from .perturbations import Perturbation, PerturbationSampler

        registry       = self.registry
        timeout        = self.cfg.timeout_s
        tool_fn        = self.adapter.call_tool
        score_fn       = self._score_fn
        synthesizer_fn = getattr(self.planner, "synthesize", None)
        base_pert = Perturbation(
            timeout_inject_p    = 0.02 * perturbation_scale,
            tool_error_inject_p = 0.02 * perturbation_scale,
            latency_jitter_ms   = 200,
        )

        def factory(seed: int) -> AdaptedExecutor:
            sampler = PerturbationSampler(base_pert, seed=seed)
            return AdaptedExecutor(
                registry       = registry,
                sampler        = sampler,
                timeout_s      = timeout,
                tool_fn        = tool_fn,
                score_fn       = score_fn,
                synthesizer_fn = synthesizer_fn,
            )

        return factory

    # ------------------------------------------------------------------
    # Stage 1: search
    # ------------------------------------------------------------------

    def run_search(self, on_record=None, on_query_done=None, skip_candidates=None, existing_pop=None, D_override=None) -> dict:
        from .evolution import EvolutionarySearch
        from .config import Budgets, RobustConfig

        D_fb,D_val = self.adapter.load_datasets()
        self._D_fb   = D_fb
        self._D_val  = D_val

        # Allow caller to substitute a different query set (e.g. D_val for eval-time search)
        D_run = D_override if D_override is not None else D_fb

        search = EvolutionarySearch(
            adapter          = self.adapter,
            compiler         = self.compiler,
            planner         = self.planner,
            checker          = self.checker,
            executor_factory = self._executor_factory,
            budgets          = Budgets(
                B=self.cfg.B, K=self.cfg.K,
                max_steps=self.cfg.max_steps,
                plan_retries=self.cfg.plan_retries,
            ),
            robust_cfg = RobustConfig(alpha=self.cfg.alpha),
            rng_seed   = self.cfg.seed,
            on_record  = on_record,
        )

        self._search = search
        self._pop = search.run(D_fb=D_run, skip_candidates=skip_candidates, on_query_done=on_query_done, existing_pop=existing_pop)
        return self._pop

    # ------------------------------------------------------------------
    # Stage 2: validate
    # ------------------------------------------------------------------

    def validate(self, D_val=None, on_record=None) -> list:
        """
        For each query in D_val: plan a workflow, compile, execute, and score it.
        Calls on_record(qid, candidate) after each query so callers can persist
        the result in the same format as the search stage.
        Returns a list of per-query result dicts sorted by score descending.
        """
        import numpy as np
        from .dsl import Action, PromptParams
        from .evolution import Candidate

        if D_val is None:
            D_val = getattr(self, "_D_val", None)
        if D_val is None:
            _, D_val = self.adapter.load_datasets()

        val_factory = self._make_executor_factory(perturbation_scale=0.0)

        plan_retries = self.cfg.plan_retries

        results = []
        for i, query in enumerate(D_val):
            qid = str(query.get("id", i))

            error_context: Optional[str] = None
            cand: Optional[Any] = None

            for attempt in range(plan_retries):
                is_final = (attempt == plan_retries - 1)

                # ── Plan ──────────────────────────────────────────────
                try:
                    graph = self.planner.plan(query, error_context=error_context)
                except Exception as exc:
                    print(f"[validate] {qid}: attempt {attempt+1} planning failed — {exc}", flush=True)
                    error_context = f"Planning exception: {exc}"
                    continue

                action = Action(
                    q=query,
                    graph=graph,
                    token_cost=getattr(graph, "token_cost", None),
                    e=(),
                    p=PromptParams(max_steps=self.cfg.max_steps),
                    error_context=error_context,
                )

                # ── Compile ───────────────────────────────────────────
                try:
                    wf = self.compiler.compile(action)
                except Exception as exc:
                    print(f"[validate] {qid}: attempt {attempt+1} compile failed — {exc}", flush=True)
                    cand = Candidate(
                        action=action, wf=None, feasible=False, score_detail=None,
                        score=0.0, execution_result=None,
                        feasibility_reasons=[f"compile_error:{exc}"],
                    )
                    error_context = f"Compilation/validation errors: {exc}"
                    continue

                # ── Static check ──────────────────────────────────────
                rep = self.checker.check(wf)
                if not rep.feasible:
                    cand = Candidate(
                        action=action, wf=wf, feasible=False, score_detail=None,
                        score=0.0, execution_result=None,
                        feasibility_reasons=rep.reasons,
                    )
                    error_context = "Compilation/validation errors: " + "; ".join(rep.reasons[:5])
                    print(f"[validate] {qid}: attempt {attempt+1} static check failed — {error_context}", flush=True)
                    continue

                # ── Execute ───────────────────────────────────────────
                execu = val_factory(i)
                res   = execu.run(wf, crn_seed=i)

                exec_ok = res.ok
                print(
                    f"[validate] {qid}: attempt {attempt+1}  ok={exec_ok}",
                    flush=True,
                )

                if not exec_ok and not is_final:
                    # Extract error context for next attempt — skip scoring
                    parts: List[str] = []
                    replan_errors = getattr(res, "replan_errors", None)
                    if replan_errors:
                        parts.append(
                            "Parallel execution failures: "
                            + "; ".join(
                                f"node={e.get('node_id')} tool={e.get('tool')} "
                                f"error={e.get('error_type')} args={e.get('input_args')}"
                                for e in replan_errors[:5]
                            )
                        )
                    elif res.logs:
                        failed = [lg for lg in res.logs if not lg.ok]
                        if failed:
                            parts.append(
                                "Execution failures: "
                                + "; ".join(
                                    f"node={lg.node_id} tool={lg.tool} error={lg.error_type}"
                                    for lg in failed[:3]
                                )
                            )
                    error_context = "\n".join(parts) if parts else "Unknown error — please revise the plan."
                    print(f"[validate] {qid}: retrying with error context: {error_context[:120]}", flush=True)
                    continue

                # ── Score (only on success or final attempt) ──────────
                score_detail = execu.score(res, wf)
                score_vals = [
                    score_detail.task_fulfillment,
                    score_detail.grounding,
                    score_detail.tool_appropriateness,
                    score_detail.parameter_accuracy,
                    score_detail.dependency_awareness,
                    score_detail.parallelism_and_efficiency,
                ]
                score = float(np.mean([v for v in score_vals if v is not None] or [0.0]))

                cand = Candidate(
                    action=action, wf=wf, feasible=True,
                    score_detail=score_detail,
                    score=score,
                    execution_result=res,
                    feasibility_reasons=[],
                )
                print(
                    f"[validate] {qid}: attempt {attempt+1}  ok={exec_ok}  score={score:.3f}",
                    flush=True,
                )
                break

            if cand is None:
                continue

            cand.plan_attempts = attempt + 1

            if on_record:
                on_record(qid, cand)

            if cand.wf is not None:
                results.append({
                    "query_id":  qid,
                    "action_id": cand.wf.action_id,
                    "ok":        cand.execution_result.ok if cand.execution_result else False,
                    "score":     cand.score,
                    "violation": getattr(cand.execution_result, "violation", None)
                                 if cand.execution_result else None,
                })

        results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        return results
    