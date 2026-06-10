from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Callable, Optional, Tuple
import math
import random
import numpy as np
import hashlib
import json

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):  # type: ignore
        desc = kwargs.get("desc", "")
        items = list(it)
        total = len(items)
        for i, x in enumerate(items):
            if i % max(1, total // 10) == 0:
                print(f"  [{desc}] {i}/{total}")
            yield x

from .benchmark_adapter import BenchmarkAdapter
from .dsl import Action, TypedEdit, PromptParams,GraphTemplate
from .compiler import Compiler, CompiledWorkflow
from .checks import StaticChecker
from .executor import MCPExecutor,ExecResult,MetricBreakdown
from .config import Budgets, RobustConfig, AdaEvolveConfig
from .query_planner import QueryPlanner


def _make_iter_candidate_id(qid: str, phase: str, iteration: int) -> str:
    """Deterministic candidate ID based on (query_id, phase, iteration index).

    - init candidates:       phase="init",  iteration=0
    - evolution candidates:  phase="evolve", iteration=<loop idx>
    - meta-guidance:         phase="meta",  iteration=<loop idx>
    """
    s = f"{qid}:{phase}:{iteration}"
    return hashlib.sha256(s.encode()).hexdigest()[:16]


@dataclass
class Candidate:
    action: Action
    wf: Optional[CompiledWorkflow]
    score_detail: MetricBreakdown
    score: MetricBreakdown
    feasible: bool
    feasibility_reasons: List[str]   #sore_lcb: float
    execution_result: ExecResult
    #score_mean: float
    plan_attempts: int = 1   # number of LLM planning/mutation attempts needed
    candidate_id: Optional[str] = None  # iteration-based stable ID; set in run()



ExecutorFactory = Callable[[int], MCPExecutor]


class EvolutionarySearch:
    """
    Implements the Evolutionary Teacher Search loop from the algorithm diagram.
    Per-query evolutionary search.
    Key correctness properties:
    Population structure
      - CRN (Common Random Numbers): fair_compare forks a fresh executor per
    --------------------
        (seed, query, repeat) so parent and child see *identical* perturbations.
    pop : Dict[query_id, List[Candidate]]
      - Stable scoring: _build_candidate derives its D_rank subsample seed
        deterministically from the action's stable_id so scores are comparable
    Each query maintains its own independent candidate list.  The evolutionary
        across generations.
    loop iterates over every query in D_fb, selects the best parent for that
      - Evidence-guided mutation: _collect_evidence returns per-node error rates
    query, collects evidence from the parent's stored execution result, asks
        from D_fb minibatch runs; _mutate biases toward failing nodes.
    the LLM to propose a targeted TypedEdit, builds and scores the child, then
      - Deduplication: Compiler._dedupe_edits prevents redundant edits accumulating
    accepts it if it beats the parent on the same query.
        across multiple mutation generations.
    LLM-guided mutation
    -------------------
    _mutate() calls QueryPlanner.propose_edit(query, failing_nodes, graph) which
    sends the query text, the list of failing node ids, and the current graph
    structure to the LLM and asks: "what single change would most improve this
    workflow?"  The LLM response is parsed into a TypedEdit and appended to the
    action's edit list.
    Random-edit fallback
    --------------------
    If the LLM call fails (no credentials, parse error, etc.) _mutate() falls
    back to the original random-op selection.
    """

    def __init__(
        self,
        adapter: BenchmarkAdapter,
        compiler: Compiler,
        planner: QueryPlanner,
        checker: StaticChecker,
        executor_factory: ExecutorFactory,
        budgets: Budgets,
        robust_cfg: RobustConfig,
        rng_seed: int = 123,
        adaevolve_cfg: Optional[AdaEvolveConfig] = None,
        on_record: Optional[Callable] = None,
    ):
        self.adapter = adapter
        self.compiler = compiler
        self.planner = planner
        self.checker = checker
        self.executor_factory = executor_factory
        self.budgets = budgets
        self.robust_cfg = robust_cfg
        self.adaevolve_cfg = adaevolve_cfg or AdaEvolveConfig()
        self.rng = random.Random(rng_seed)
        self._plan_cache: Dict[str, GraphTemplate] = {}   # query_hash → base plan
        # AdaEvolve per-query state
        self._growth_signal: Dict[str, float] = {}        # G_t per query
        self._local_best: Dict[str, float] = {}           # f_k* per query
        # Dataset accumulator — every recorded candidate across all generations
        # Each entry is (qid, generation, phase, candidate)
        self._history: List[Tuple[str, int, str, "Candidate"]] = []
        # Optional streaming export callback — called immediately on each record
        self._on_record = on_record
        # Task 2: per-query node snapshot for differential context
        self._prev_nodes: Dict[str, List[Dict]] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        D_fb: List[Dict[str, Any]],
        skip_candidates: Optional[set] = None,
        on_query_done: Optional[Callable] = None,
        existing_pop: Optional[Dict[str, List["Candidate"]]] = None,
    ) -> Dict[str, List["Candidate"]]:
        """
        Run evolutionary search.
        First initializes action as an candidate and uses it to evolution

        Args:
            D_fb:             feedback dataset  — used for evidence collection + fair compare
            skip_candidates:  set of (query_id, candidate_id) tuples already present in workflow_history.jsonl; per-candidate skip — if the init
                              candidate's ID is already known the LLM/executor is not called and the existing candidate from existing_pop is used instead.
            on_query_done:    called as on_query_done(qid, candidates) immediately after each query's evolution finishes so results are persisted
                              before the next query begins.
            existing_pop:     pre-loaded Dict[qid, List[Candidate]] from a previous run; used to seed pop when the init candidate is already in history so the LLM planner and executor are not called again.

        Returns:
            Final population (new queries only) sorted by score descending.
        """
        # Build a fast per-qid lookup: qid → set of known candidate_ids
        _skip_cids: Dict[str, set] = {}
        if skip_candidates:
            for qid_s, cid_s in skip_candidates:
                _skip_cids.setdefault(qid_s, set()).add(cid_s)

        # Seed: one Action per query in D_fb, no edits
        # ── Step 2: build initial feasible population ──────────────────
        pop : Dict[str,List[Candidate]] = {}
        for query in D_fb:
            qid  = str(query["id"])
            known_cids = _skip_cids.get(qid, set())

            # Compute init candidate_id before any LLM/executor work so we can
            # skip the expensive call entirely when this iteration is already done.
            init_cid = _make_iter_candidate_id(qid, "init", 0)
            if init_cid in known_cids:
                # Load existing candidates from the pre-populated pool so
                # evolution can use them as parents without re-running init.
                if existing_pop and qid in existing_pop:
                    for ec in existing_pop[qid]:
                        pop.setdefault(qid, []).append(ec)
                    print(
                        f"[skip] {qid}: init candidate (iter=0) already in history"
                        f" — loaded {len(existing_pop[qid])} existing candidate(s)",
                        flush=True,
                    )
                else:
                    print(
                        f"[skip] {qid}: init candidate (iter=0) already in history"
                        f" — no existing_pop provided, skipping query",
                        flush=True,
                    )
                continue

            error_context: Optional[str] = None
            cand: Optional[Candidate] = None
            for plan_attempt in range(self.budgets.plan_retries):
                is_final = (plan_attempt == self.budgets.plan_retries - 1)
                graph = self._get_base_plan(query, error_context=error_context)
                cand = self._build_candidate(
                    Action(q=query, graph=graph, token_cost=graph.token_cost,
                           e=(), p=PromptParams(max_steps=15),
                           error_context=error_context),
                    is_final_attempt=is_final,
                )
                # Success: plan compiled, passed static checks, and executed OK
                exec_ok = (
                    cand.execution_result is None
                    or cand.execution_result.ok
                )
                if cand.feasible and exec_ok:
                    print(
                        f"[initialization] {qid}: score={cand.score} "
                        f" (attempt {plan_attempt + 1})",
                        f"| detail_score={cand.score_detail} ",
                        flush=True,
                    )
                    break

                # Failure: extract error context and retry with LLM feedback
                error_context = self._extract_plan_error(cand)
                print(
                    f"[initialization] {qid}: attempt {plan_attempt + 1} failed"
                    f" — {error_context}",
                    flush=True,
                )
            if cand is not None:
                cand.plan_attempts = plan_attempt + 1
                cand.candidate_id = init_cid
                self._record_candidate(qid, generation=-1, phase="init", cand=cand)
            pop.setdefault(qid, []).append(cand)

        if not pop:
            if _skip_cids:
                print("[search] All queries already in candidate store — nothing new to run.", flush=True)
                return pop
            raise RuntimeError("No feasible candidates in initial population.")

        # ── Initialise AdaEvolve per-query state ────────────────────────
        for qid_key, cands in pop.items():
            scores = [c.score for c in cands if isinstance(c.score, (int, float))]
            self._local_best[qid_key]   = max(scores) if scores else 0.0
            self._growth_signal[qid_key] = 0.0

        # ── Step 4: evolutionary loop (AdaEvolve-guided) ────────────────
        for query in D_fb:
            qid = str(query["id"])
            if qid not in pop:
                continue   # was skipped in init loop
            idx = -1
            meta_cooldown = 0   # iterations until next meta-guidance is allowed
            print(f"[evolutionary loop] {qid}:", flush=True)

            try:
                for _ in tqdm(range(self.budgets.B), desc=f"evolve q={qid}"):
                    idx += 1

                    # ── AdaEvolve: compute adaptive intensity ────────────────
                    intensity = self._compute_intensity(qid)

                    # ── AdaEvolve L3: meta-guidance on persistent stagnation ─
                    if (self._growth_signal.get(qid, 0.0) < self.adaevolve_cfg.tau_m
                            and meta_cooldown == 0):
                        # Check meta candidate_id before any LLM work
                        meta_cid = _make_iter_candidate_id(qid, "meta", idx)
                        if meta_cid in _skip_cids.get(qid, set()):
                            print(f"[skip] {qid} iter={idx}: meta candidate already in history, skipping", flush=True)
                        else:
                            new_graph = self._meta_guidance(query, pop[qid])
                            if new_graph is not None:
                                meta_action = Action(
                                    q=query, graph=new_graph,
                                    token_cost=new_graph.token_cost,
                                    e=(), p=PromptParams(max_steps=15),
                                )
                                meta_cand = self._build_candidate(meta_action)
                                if meta_cand.feasible:
                                    meta_cand.candidate_id = meta_cid
                                    self._update_growth_signal(qid, meta_cand.score)
                                    self._record_candidate(qid, generation=idx, phase="meta", cand=meta_cand,
                                                           G=self._growth_signal.get(qid), I=self._compute_intensity(qid),
                                                           local_best=self._local_best.get(qid))
                                    pop[qid].append(meta_cand)
                                    pop[qid] = self._prune(pop[qid])
                                    print(
                                        f"\t{idx}: [meta-guidance] score={meta_cand.score:.3f}"
                                        f" G={self._growth_signal[qid]:.4f}",
                                        flush=True,
                                    )
                        cooldown_iters = max(1, int(self.budgets.B
                                                    * self.adaevolve_cfg.meta_cooldown_frac))
                        meta_cooldown = cooldown_iters

                    if meta_cooldown > 0:
                        meta_cooldown -= 1

                    # ── Skip evolution iteration if already in history ────────
                    child_cid = _make_iter_candidate_id(qid, "evolve", idx)
                    if child_cid in _skip_cids.get(qid, set()):
                        print(f"[skip] {qid} iter={idx}: evolution candidate already in history, skipping", flush=True)
                        continue

                    # ── Adaptive parent selection ────────────────────────────
                    parent = self._select_parent(pop[qid], intensity=intensity)

                    # Collect per-node error rates from parent's stored execution
                    evidence = self._collect_evidence(parent)

                    # Exploration mode coin-flip (AdaEvolve: prob = intensity)
                    exploration_mode = self.rng.random() < intensity

                    child_cand: Optional[Candidate] = None
                    mut_attempt = 0

                    # ── Standard mutation with retry ──────────────────────────
                    if child_cand is None:
                        mut_error_context = parent.action.error_context
                        for mut_attempt in range(self.budgets.plan_retries):
                            is_final = (mut_attempt == self.budgets.plan_retries - 1)
                            child_action = self._mutate(
                                parent.action, evidence=evidence,
                                error_context=mut_error_context,
                                exploration_mode=exploration_mode,
                            )
                            child_cand = self._build_candidate(child_action, is_final_attempt=is_final)
                            exec_ok = (
                                child_cand.execution_result is None
                                or child_cand.execution_result.ok
                            )
                            if child_cand.feasible and exec_ok:
                                break
                            mut_error_context = self._extract_plan_error(child_cand)
                            print(
                                f"[evolution] {qid} iter={idx}: mutation attempt {mut_attempt + 1}"
                                f" failed — {mut_error_context}",
                                flush=True,
                            )
                    if child_cand is None or not child_cand.feasible:
                        continue

                    child_cand.plan_attempts = mut_attempt + 1
                    child_cand.candidate_id = child_cid

                    # ── AdaEvolve: update growth signal ──────────────────────
                    self._update_growth_signal(qid, child_cand.score)

                    _phase = "evolve"
                    self._record_candidate(qid, generation=idx, phase=_phase, cand=child_cand,
                                           G=self._growth_signal.get(qid), I=intensity,
                                           exploration_mode=exploration_mode,
                                           parent_score=parent.score,
                                           local_best=self._local_best.get(qid))

                    pop[qid].append(child_cand)
                    pop[qid] = self._prune(pop[qid])
                    print(
                        f"\t{idx}: Parent={parent.score:.3f} Child={child_cand.score:.3f}"
                        f" I={intensity:.3f} G={self._growth_signal[qid]:.4f}"
                        f" mode={'explore' if exploration_mode else 'exploit'}"
                        f" fair={self._fair_compare(parent, child_cand)}",
                        flush=True,
                    )

            except Exception as exc:
                exc_str = str(exc)
                is_token_expired = (
                    "ExpiredTokenException" in exc_str
                    or "ExpiredToken" in type(exc).__name__
                )
                if is_token_expired and pop.get(qid):
                    print(
                        f"[evolutionary loop] {qid}: token expired — flushing {len(pop[qid])} "
                        f"partial candidate(s) before exit",
                        flush=True,
                    )
                    pop[qid].sort(key=lambda c: c.score, reverse=True)
                    if on_query_done is not None:
                        try:
                            on_query_done(qid, pop[qid],
                                          G=self._growth_signal.get(qid),
                                          I=self._compute_intensity(qid))
                        except Exception as flush_exc:
                            print(f"[query_done] flush on expiry failed ({flush_exc})", flush=True)
                raise

            # ── Persist this query's final population immediately ────────
            pop[qid].sort(key=lambda c: c.score, reverse=True)
            if on_query_done is not None:
                try:
                    on_query_done(qid, pop[qid],
                                  G=self._growth_signal.get(qid),
                                  I=self._compute_intensity(qid))
                except Exception as exc:
                    print(f"[query_done] persist callback failed ({exc}), continuing", flush=True)

        for qid in pop:
            pop[qid].sort(key=lambda c: c.score, reverse=True)
        return pop

    # ------------------------------------------------------------------
    # Dataset recording
    # ------------------------------------------------------------------

    def _record_candidate(
        self,
        qid:              str,
        generation:       int,
        phase:            str,      # "init" | "evolve" | "meta"
        cand:             "Candidate",
        G:                Optional[float] = None,
        I:                Optional[float] = None,
        exploration_mode: Optional[bool]  = None,
        parent_score:     Optional[float] = None,
        local_best:       Optional[float] = None,
    ) -> None:
        """
        Append a (qid, generation, phase, candidate) tuple to the history log.
        Serialization is deferred to run_pipeline._save_history so there is
        one canonical serializer for the whole codebase.

        If on_record was provided at construction, it is called immediately so
        the candidate is persisted to disk before the next iteration begins —
        ensuring data survives even if the loop crashes partway through.
        """
        self._history.append((qid, generation, phase, cand))
        if self._on_record is not None:
            try:
                self._on_record(qid, generation, phase, cand, G=G, I=I,
                                exploration_mode=exploration_mode,
                                parent_score=parent_score,
                                local_best=local_best)
            except Exception as exc:
                print(f"[record] export callback failed ({exc}), continuing", flush=True)

    def get_history(self) -> List[Tuple[str, int, str, "Candidate"]]:
        """Return the full list of (qid, generation, phase, candidate) tuples."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Candidate construction
    # ------------------------------------------------------------------

    def _build_candidate(self, a: Action, is_final_attempt: bool = True) -> Candidate:
        """
        Compile action, static-check, execute on the action's own query,
        and score the result.

        When is_final_attempt=False, scoring is skipped if the execution
        fails so that the caller can retry without paying the scoring cost.
        Scoring always runs on success or on the final attempt.
        """
        try:
            wf = self.compiler.compile(a)
        except Exception as exc:
            return Candidate(
                action=a, wf=None, feasible=False,score_detail=None, score=0.0,
                execution_result=None,
                feasibility_reasons=[f"compile_error:{exc}"],
            )

        rep = self.checker.check(wf)
        if not rep.feasible:
            return Candidate(
                action=a, wf=wf, feasible=False, score_detail=None, score=0.0,
                execution_result=None,
                feasibility_reasons=rep.reasons,
            )
        
        # Deterministic subsample: seed = low bits of action hash
        action_seed = int(a.stable_id(), 16) % (2 ** 31)
        execu       = self.executor_factory(action_seed)
        res         = execu.run(wf, crn_seed=action_seed)

        # Skip scoring when execution errored and this is not the final attempt.
        # The caller will retry with a revised plan; scoring a broken execution
        # wastes an LLM call and produces a misleading score.
        if not res.ok and not is_final_attempt:
            return Candidate(
                action=a, wf=wf, feasible=True,
                score_detail=None,
                score=0.0,
                execution_result=res,
                feasibility_reasons=[],
            )

        evaluate_run = execu.score(res, wf)
        score = [evaluate_run.task_fulfillment,
                 evaluate_run.grounding,
                 evaluate_run.tool_appropriateness,
                 evaluate_run.parameter_accuracy,
                 evaluate_run.dependency_awareness,
                 evaluate_run.parallelism_and_efficiency]
        return Candidate(
            action=a, wf=wf, feasible=True,
            score_detail=evaluate_run,
            score=np.mean(score),
            execution_result=res,
            feasibility_reasons=[],
        )
    
    # ------------------------------------------------------------------
    # Base plan: LLM planner with cache
    # ------------------------------------------------------------------

    def _get_base_plan(
        self,
        query: Dict[str, Any],
        error_context: Optional[str] = None,
    ) -> GraphTemplate:
        """
        Return (and cache) the LLM-generated base plan for this query.

        The cache key is a hash of the query dict so identical queries
        never call the LLM more than once during a run.

        When error_context is provided (retry path) the cache is bypassed
        so the LLM receives the error feedback and produces a fresh plan.
        """
        qkey = hashlib.sha256(
            json.dumps(query, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]

        if qkey not in self._plan_cache or error_context is not None:
            self._plan_cache[qkey] = self.planner.plan(query, error_context=error_context)

        return self._plan_cache[qkey]

    # ------------------------------------------------------------------
    # Error extraction for plan-retry feedback
    # ------------------------------------------------------------------

    def _extract_plan_error(self, cand: "Candidate") -> str:
        """
        Build a concise error description from a failed Candidate so it can
        be fed back to the LLM planner on the next attempt.
        """
        parts: List[str] = []

        if cand.feasibility_reasons:
            parts.append(
                "Compilation/validation errors: "
                + "; ".join(cand.feasibility_reasons[:5])
            )

        if cand.execution_result:
            # Prefer the structured replan_errors collected from parallel
            # execution — they include all failures in a wave, not just the
            # first one, and carry input_args for better LLM diagnosis.
            replan_errors = getattr(cand.execution_result, "replan_errors", None)
            if replan_errors:
                parts.append(
                    "Parallel execution failures: "
                    + "; ".join(
                        f"node={e.get('node_id')} tool={e.get('tool')} "
                        f"error={e.get('error_type')} args={e.get('input_args')}"
                        for e in replan_errors[:5]
                    )
                )
            elif cand.execution_result.logs:
                failed = [lg for lg in cand.execution_result.logs if not lg.ok]
                if failed:
                    parts.append(
                        "Execution failures: "
                        + "; ".join(
                            f"node={lg.node_id} tool={lg.tool} error={lg.error_type}"
                            for lg in failed[:3]
                        )
                    )
        return "\n".join(parts) if parts else "Unknown error — please revise the plan."

    def invalidate_cache(self, query: Optional[Dict[str, Any]] = None) -> None:
        """
        Clear the plan cache.  Pass a query to invalidate just that entry,
        or call with no args to flush everything (useful between runs).
        """
        if query is None:
            self._plan_cache.clear()
        else:
            qkey = hashlib.sha256(
                json.dumps(query, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()[:16]
            self._plan_cache.pop(qkey, None)


    # ------------------------------------------------------------------
    # AdaEvolve: growth signal + exploration intensity
    # ------------------------------------------------------------------

    def _update_growth_signal(self, qid: str, child_score: float) -> None:
        """
        Update the exponential moving average growth signal G_t for a query.

        delta_t  = max( (child_score - local_best) / local_best, 0 )
        G_t      = rho * G_{t-1} + (1-rho) * delta_t^2

        When the child improves the local best, local_best is also updated.
        """
        cfg        = self.adaevolve_cfg
        local_best = self._local_best.get(qid, 0.0)
        denom      = abs(local_best) if abs(local_best) > cfg.eps else cfg.eps
        delta      = max((child_score - local_best) / denom, 0.0)
        G_prev     = self._growth_signal.get(qid, 0.0)
        self._growth_signal[qid] = cfg.rho * G_prev + (1.0 - cfg.rho) * delta ** 2
        if child_score > local_best:
            self._local_best[qid] = child_score

    def _compute_intensity(self, qid: str) -> float:
        """
        Compute the adaptive exploration intensity I_t for a query.

        I_t = I_min + (I_max - I_min) / (1 + sqrt(G_t + eps))

        High G (active improvement)  → I_t → I_min  (exploit the gradient)
        Low  G (stagnation)          → I_t → I_max  (explore new regions)
        """
        cfg = self.adaevolve_cfg
        G   = self._growth_signal.get(qid, 0.0)
        return cfg.I_min + (cfg.I_max - cfg.I_min) / (1.0 + math.sqrt(G + cfg.eps))

    # ------------------------------------------------------------------
    # AdaEvolve L3: meta-guidance on persistent stagnation
    # ------------------------------------------------------------------

    def _meta_guidance(
        self,
        query: Dict[str, Any],
        pop:   List[Candidate],
    ) -> Optional[GraphTemplate]:
        """
        Invoke the LLM planner's meta_guide to escape a local optimum.

        Triggered when G_t < tau_m, meaning the search has not improved the
        local best for several consecutive iterations.  Passes the current
        best workflow structure and a digest of recent low-scoring variants
        to the planner so it can propose a qualitatively different plan.
        """
        sorted_pop = sorted(pop, key=lambda c: c.score, reverse=True)
        best       = sorted_pop[0]

        best_nodes: List[Dict[str, Any]] = []
        if best.wf is not None:
            best_nodes = [
                {
                    "node_id": n.node_id,
                    "server":  n.tool_ref[0] if n.tool_ref else None,
                    "tool":    n.tool_ref[1] if n.tool_ref else None,
                    "requires": n.requires,
                }
                for n in best.wf.nodes
                if n.kind == "tool"
            ]

        # Summarise the worst recent candidates as "failed attempts"
        worst_n = sorted_pop[max(0, len(sorted_pop) - 3):]
        failed_summaries = [
            f"score={c.score:.3f} ops=[{', '.join(e.op for e in c.action.e) or 'base'}]"
            for c in worst_n
        ]

        print(
            f"[meta-guidance] G below tau_m={self.adaevolve_cfg.tau_m:.2f}"
            f" for qid={query.get('id')} — requesting strategy redesign",
            flush=True,
        )
        try:
            return self.planner.meta_guide(query, best_nodes, failed_summaries)
        except Exception as exc:
            exc_str = str(exc)
            if "ExpiredTokenException" in exc_str or "ExpiredToken" in type(exc).__name__:
                raise
            print(f"[meta-guidance] failed ({exc}), skipping", flush=True)
            return None

    # ------------------------------------------------------------------
    # Evidence collection
    # ------------------------------------------------------------------

    def _collect_evidence(self, parent: Candidate) -> Dict[str, float]:
        """
        Extract per-node error rates from the parent's stored execution result.

        Returns {node_id: error_rate} where error_rate ∈ [0, 1].
        The rates reflect which nodes structurally failed (tool errors, timeouts,
        validator failures) — not the overall task reward.
        """
        if parent.execution_result is None:
            return {}

        error_counts: Dict[str, int] = {}
        for log in parent.execution_result.logs:
            if not log.ok:
                error_counts[log.node_id] = error_counts.get(log.node_id, 0) + 1

        total = sum(error_counts.values()) or 1
        return {k: v / total for k, v in error_counts.items()}

    # ------------------------------------------------------------------
    # Task 4: Fail-fast feasibility check
    # ------------------------------------------------------------------

    def _feasible_edit_ops(
        self,
        _wf: Optional["CompiledWorkflow"],
        all_node_ids: List[str],
    ) -> List[str]:
        """Return the subset of edit operations that are structurally valid."""
        all_tools = list(self.compiler.registry.tools.keys())
        ops = ["insert_validator", "set_param"]
        if all_tools:
            ops.extend(["swap_tool", "add_tool_step"])
        if len(all_node_ids) > 1:
            ops.extend(["remove_tool_step", "reorder_step"])
        return ops

    # ------------------------------------------------------------------
    # Task 1: Compound random mutate (exploration path — zero LLM calls)
    # ------------------------------------------------------------------

    def _compound_random_mutate(
        self,
        a: Action,
        evidence: Optional[Dict[str, float]],
        n_edits: int = 3,
    ) -> Action:
        """
        Apply n_edits sequential random edits without any LLM calls.
        Recomputes feasible ops after each edit since the graph changes.
        """
        edits = list(a.e)
        current_action = a

        for _ in range(n_edits):
            try:
                wf = self.compiler.compile(current_action)
                tool_nodes = [n for n in wf.nodes if n.kind == "tool"]
                all_node_ids = [n.node_id for n in tool_nodes]
            except Exception:
                # Graph is broken — stop here rather than stacking edits
                # against phantom node IDs that would fail at _build_candidate.
                break
            if not all_node_ids:
                break

            feasible_ops = self._feasible_edit_ops(wf, all_node_ids)
            all_tools = list(self.compiler.registry.tools.keys())

            # Evidence-weighted node targeting
            if evidence:
                hot = [n for n in sorted(evidence, key=lambda k: evidence[k], reverse=True)
                       if n in all_node_ids]
                nid = hot[0] if hot and self.rng.random() < 0.70 else self.rng.choice(all_node_ids)
            else:
                nid = self.rng.choice(all_node_ids)

            op = self.rng.choice(feasible_ops)

            if op == "swap_tool" and all_tools:
                ns, nt = self.rng.choice(all_tools)
                edits.append(TypedEdit(op="swap_tool", args={
                    "node_id": nid, "new_server": ns, "new_tool": nt,
                }))
            elif op == "insert_validator":
                vid = f"v{self.rng.randint(1, 99999)}"
                edits.append(TypedEdit(op="insert_validator", args={
                    "after_node_id": nid, "validator_id": vid,
                }))
            elif op == "set_param":
                key = self.rng.choice(["strict_schema", "verifier_strict"])
                val = self.rng.choice([True, False])
                edits.append(TypedEdit(op="set_param", args={
                    "node_id": nid, "key": key, "value": val,
                }))
            elif op == "add_tool_step" and all_tools:
                ns, nt = self.rng.choice(all_tools)
                new_nid = f"t_add_{self.rng.randint(1, 99999)}"
                edits.append(TypedEdit(op="add_tool_step", args={
                    "node_id": new_nid, "server": ns, "tool": nt,
                    "after_node_id": nid,
                }))
            elif op == "remove_tool_step" and len(all_node_ids) > 1:
                edits.append(TypedEdit(op="remove_tool_step", args={"node_id": nid}))
            elif op == "reorder_step" and len(all_node_ids) > 1:
                other_ids = [i for i in all_node_ids if i != nid]
                if other_ids:
                    new_req = self.rng.choice(other_ids)
                    edits.append(TypedEdit(op="reorder_step", args={
                        "node_id": nid, "new_requires": [new_req],
                    }))

            current_action = Action(q=a.q, graph=a.graph, e=tuple(edits),
                                    token_cost=None, p=a.p)

        return Action(q=a.q, graph=a.graph, e=tuple(edits), token_cost=None,
                      p=a.p, error_context=a.error_context)

    # ------------------------------------------------------------------
    # Task 2: Differential context compression
    # ------------------------------------------------------------------

    def _compress_node_context(
        self,
        qid: str,
        current_nodes: List[Dict],
        evidence: Dict[str, float],
        max_context_nodes: int = 8,
    ) -> List[Dict]:
        """
        Return a compressed node list for exploit-mode propose_edit calls.

        Focus set = nodes with errors + their require-neighbours + diff from
        previous snapshot. Capped at max_context_nodes. Everything outside
        the focus set is replaced with a single summary entry.

        First call per query (no prior snapshot) returns the full list.
        """
        prev = self._prev_nodes.get(qid)
        if prev is None:
            return current_nodes

        current_by_id = {n["node_id"]: n for n in current_nodes}

        # Seed focus: nodes with non-zero error rate
        focus_ids: set = set()
        for node_id, err_rate in evidence.items():
            if err_rate > 0 and node_id in current_by_id:
                focus_ids.add(node_id)

        # Expand to require-neighbours
        seed_ids = set(focus_ids)
        for node_id in seed_ids:
            node = current_by_id[node_id]
            for n in current_nodes:
                if node_id in (n.get("requires") or []):
                    focus_ids.add(n["node_id"])
            for req in (node.get("requires") or []):
                if req in current_by_id:
                    focus_ids.add(req)

        # Add nodes added/removed since last snapshot
        prev_ids = {n["node_id"] for n in prev}
        curr_ids = set(current_by_id.keys())
        focus_ids.update((prev_ids ^ curr_ids) & curr_ids)

        # Sort by error rate descending and cap
        focus_list = sorted(
            [n for n in current_nodes if n["node_id"] in focus_ids],
            key=lambda n: evidence.get(n["node_id"], 0.0),
            reverse=True,
        )[:max_context_nodes]

        focus_set_ids = {n["node_id"] for n in focus_list}
        excluded = [n for n in current_nodes if n["node_id"] not in focus_set_ids]

        result = list(focus_list)
        if excluded:
            result.append({
                "summary": f"{len(excluded)} other node(s) unchanged",
                "node_ids": [n["node_id"] for n in excluded],
            })
        return result

    # ------------------------------------------------------------------
    # Mutation (evidence-guided)
    # ------------------------------------------------------------------

    def _mutate(
        self,
        a:                Action,
        evidence:         Optional[Dict[str, float]] = None,
        error_context:    Optional[str] = None,
        exploration_mode: bool = False,
    ) -> Action:
        """
        Produce a child action by appending one or more TypedEdits.

        Exploration path (Task 1):
            Calls _compound_random_mutate — applies n_edits random edits with
            zero LLM calls.  Evidence-weighted node targeting is preserved.

        Exploitation path:
            Calls QueryPlanner.propose_edit with a compressed node context
            (Task 2) and feasible-ops constraint (Task 4).

        Fallback — random edit:
            Used when the planner has no credentials or the LLM call fails.
            Uses _feasible_edit_ops so impossible ops are never attempted.
        """
        # ── Task 1: Exploration bypass — no LLM call ─────────────────
        if exploration_mode:
            return self._compound_random_mutate(a, evidence or {})

        # ── Compile parent to derive graph context ────────────────────
        edits = list(a.e)
        try:
            wf           = self.compiler.compile(a)
            tool_nodes   = [n for n in wf.nodes if n.kind == "tool"]
            all_node_ids = [n.node_id for n in tool_nodes]
        except Exception:
            wf, tool_nodes, all_node_ids = None, [], []
        if not all_node_ids:
            all_node_ids = ["t1", "t2", "t3"]

        # Task 4: pre-filter to structurally valid operations
        feasible_ops = self._feasible_edit_ops(wf, all_node_ids)

        # Identify failing nodes (highest error rate first)
        failing_nodes = sorted(evidence or {}, key=lambda k: (evidence or {})[k], reverse=True)

        # Task 2: compress node context using per-query snapshot
        qid = str(a.q.get("id", ""))
        full_nodes = [
            {"node_id": n.node_id,
             "server":  n.tool_ref[0] if n.tool_ref else None,
             "tool":    n.tool_ref[1] if n.tool_ref else None,
             "requires": n.requires}
            for n in (wf.nodes if tool_nodes else [])
            if n.kind == "tool"
        ]
        compressed_nodes = self._compress_node_context(qid, full_nodes, evidence or {})

        # ── Primary: ask the LLM to propose an edit ───────────────────
        llm_edit = None
        try:
            llm_edit, new_token_cost = self.planner.propose_edit(
                query            = a.q,
                failing_nodes    = failing_nodes,
                current_nodes    = compressed_nodes,
                registry         = self.compiler.registry,
                error_context    = error_context,
                exploration_mode = False,
                allowed_ops      = feasible_ops,
            )
        except Exception as exc:
            exc_str = str(exc)
            if "ExpiredTokenException" in exc_str or "ExpiredToken" in type(exc).__name__:
                raise
            print(f"[evolution] propose_edit failed ({exc}), using random fallback")
            llm_edit = None

        if llm_edit is not None:
            edits.append(llm_edit)
            # Task 2: update snapshot after successful call
            if qid:
                self._prev_nodes[qid] = full_nodes
            token_cost = self.merge_token_usage(a.token_cost, new_token_cost)
            return Action(q=a.q, graph=a.graph, e=tuple(edits), token_cost=token_cost,
                          p=a.p, error_context=error_context)

        # ── Fallback: random edit (Task 4: use feasible_ops) ──────────
        if evidence:
            hot = [n for n in sorted(evidence, key=lambda k: evidence[k], reverse=True)
                   if n in all_node_ids]
            nid = hot[0] if hot and self.rng.random() < 0.70 else self.rng.choice(all_node_ids)
        else:
            nid = self.rng.choice(all_node_ids)

        all_tools = list(self.compiler.registry.tools.keys())
        op = self.rng.choice(feasible_ops)

        if op == "swap_tool" and all_tools:
            ns, nt = self.rng.choice(all_tools)
            edits.append(TypedEdit(op="swap_tool", args={
                "node_id": nid, "new_server": ns, "new_tool": nt,
            }))

        elif op == "insert_validator":
            vid = f"v{self.rng.randint(1, 99999)}"
            edits.append(TypedEdit(op="insert_validator", args={
                "after_node_id": nid, "validator_id": vid,
            }))

        elif op == "set_param":
            key = self.rng.choice(["strict_schema", "verifier_strict"])
            val = self.rng.choice([True, False])
            edits.append(TypedEdit(op="set_param", args={
                "node_id": nid, "key": key, "value": val,
            }))

        elif op == "add_tool_step" and all_tools:
            ns, nt  = self.rng.choice(all_tools)
            new_nid = f"t_add_{self.rng.randint(1, 99999)}"
            edits.append(TypedEdit(op="add_tool_step", args={
                "node_id": new_nid, "server": ns, "tool": nt,
                "after_node_id": nid,
            }))

        elif op == "remove_tool_step" and len(all_node_ids) > 1:
            edits.append(TypedEdit(op="remove_tool_step", args={"node_id": nid}))

        elif op == "reorder_step" and len(all_node_ids) > 1:
            other_ids = [i for i in all_node_ids if i != nid]
            if other_ids:
                new_req = self.rng.choice(other_ids)
                edits.append(TypedEdit(op="reorder_step", args={
                    "node_id": nid, "new_requires": [new_req],
                }))

        # Task 2: keep snapshot fresh even on the fallback path so the
        # next iteration's diff doesn't accumulate stale "changed" nodes.
        if qid and full_nodes:
            self._prev_nodes[qid] = full_nodes

        return Action(q=a.q, graph=a.graph, e=tuple(edits), token_cost=None,
                      p=a.p, error_context=error_context)

    # ------------------------------------------------------------------
    # Fair compare  (same query, both candidates already executed)
    # ------------------------------------------------------------------

    def _fair_compare(self, parent: Candidate, child: Candidate) -> bool:
        """
        Accept child if its score exceeds the parent's score on the same query.

        Both candidates have already been executed against their own query in
        _build_candidate, so no re-execution is needed.
        """
        if not child.feasible:
            return False
        return child.score > parent.score

    # ------------------------------------------------------------------
    # Selection  (per-query list)
    # ------------------------------------------------------------------

    def _select_parent(self, candidates: List[Candidate], intensity: float = 0.4) -> Candidate:
        """
        Adaptive parent selection (AdaEvolve).

        Exploration mode  (prob = intensity):  sample uniformly from the full
            population, encouraging variety in the mutation starting point.
        Exploitation mode (prob = 1-intensity): tournament within the top
            quartile, biasing toward refining the current best candidates.
        """
        if self.rng.random() < intensity:
            # Exploration: uniform sample from whole population
            return self.rng.choice(candidates)

        # Exploitation: tournament from top quartile
        sorted_cands = sorted(candidates, key=lambda c: c.score, reverse=True)
        top_k        = max(1, len(sorted_cands) // 4)
        pool         = sorted_cands[:top_k]
        k            = min(5, len(pool))
        contestants  = self.rng.sample(pool, k=k)
        return max(contestants, key=lambda c: c.score)

    # ------------------------------------------------------------------
    # Pruning  (per-query list)
    # ------------------------------------------------------------------

    def _prune(self, candidates: List[Candidate]) -> List[Candidate]:
        """
        Keep the best-K candidates for one query by score, with light
        diversity binning (16 action-hash buckets, round-robin fill).
        """
        sorted_cands = sorted(candidates, key=lambda c: c.score, reverse=True)

        bins: Dict[int, List[Candidate]] = {}
        for c in sorted_cands:
            b = int(c.wf.action_id, 16) % 16 if c.wf else 0
            bins.setdefault(b, []).append(c)

        kept: List[Candidate] = []
        while len(kept) < self.budgets.K and any(bins.values()):
            for b in list(bins.keys()):
                if bins[b]:
                    kept.append(bins[b].pop(0))
                    if len(kept) >= self.budgets.K:
                        break
        return kept

    # ------------------------------------------------------------------
    # Merge two token usage dicts, accumulating totals and call history
    # ------------------------------------------------------------------
    @staticmethod
    def merge_token_usage(a: dict | None, b: dict | None) -> dict:
        """
        Merge two token-cost dicts produced by individual LLM calls.

        Output shape
        ------------
        {
            "prompt_tokens":     <int total>,
            "completion_tokens": <int total>,
            "total_tokens":      <int total>,
            "calls": [
                {"prompt_tokens": 120, "completion_tokens": 40, ...},
                {"prompt_tokens":  80, "completion_tokens": 30, ...},
            ]
        }

        ``calls`` records every individual call so per-call cost is always
        recoverable; the top-level numeric keys are running sums for cheap
        budget checks.

        Both *a* and *b* may be a raw single-call dict (no "calls" key) or an
        already-merged dict (has "calls") — both shapes are handled.
        """
        NUMERIC_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")

        def _to_calls(d: dict) -> list:
            """Normalise either shape to a list of raw call dicts."""
            if not d:
                return []
            if "calls" in d:
                return list(d["calls"])
            # Raw single-call dict — strip the meta keys before storing
            return [{k: d[k] for k in NUMERIC_KEYS if k in d}]

        a = a or {}
        b = b or {}
        calls = _to_calls(a) + _to_calls(b)

        totals: dict = {}
        for key in NUMERIC_KEYS:
            total = sum(
                c.get(key) or 0
                for c in calls
                if isinstance(c.get(key), (int, float))
            )
            totals[key] = total

        return {**totals, "calls": calls}






