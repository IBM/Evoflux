"""
scripts/run_pipeline.py
-----------------------
Unified pipeline runner.  Select a benchmark with --benchmark.
 
Usage:
    # Built-in toy (no external deps, for testing)
    PYTHONPATH=. python3 -m scripts.run_pipeline --benchmark toy
 
    # IBM AssetOpsBench  (needs Docker stack running)
    PYTHONPATH=. python3 -m scripts.run_pipeline --benchmark assetopsbench
 
    # Accenture MCP-Bench  (needs mcp-bench cloned + servers installed)
    PYTHONPATH=. python3 -m scripts.run_pipeline \\
        --benchmark mcpbench \\
        --mcpbench-root /path/to/mcp-bench \\
        --complexity single
 
    # Full pipeline stages can be selected individually:
    PYTHONPATH=. python3 -m scripts.run_pipeline --benchmark toy --stages search,export,validate
 
Outputs (all written to --output-dir, default "results/"):
    workflow_history.jsonl  — all candidates (intermediates + finals); final candidates have _meta["is_final"]=True
    validation_report.json  — per-candidate score table
 
Population file format
----------------------
Each line is one JSON object:
{
    "query_id":  str,
    "action_id": str,
    "score":     float,
    "feasible":  bool,
    "feasibility_reasons": [...],
    "action": {
        "q": {...},                         # full query dict
        "e": [{"op": ..., "args": {...}}],  # typed edits
        "p": {...}                          # PromptParams fields
    },
    "execution_result": {
        "ok":           bool,
        "violation":    str | null,
        "final_output": {...} | null,
        "logs": [
            {
                "node_id":    str,
                "kind":       str,
                "tool":       str | null,
                "ok":         bool,
                "error_type": str | null,
                "latency_ms": int,
                "output":     {...} | null
            },
            ...
        ]
    }
}
"""
from __future__ import annotations
import argparse
import json
import os
import sys
#sys.path.append("..")
# ── Adapter registry ──────────────────────────────────────────────────────────
# Add new adapters here; nothing else needs to change.
def _build_adapter(args):
    name = args.benchmark.lower()

    if name == "toy":
        from mcp_router.adapters.toy import ToyAdapter
        return ToyAdapter()
    if name == "assetopsbench":
        from mcp_router.adapters.assetopsbench import AssetOpsBenchAdapter
        return AssetOpsBenchAdapter(
            endpoints={
                "iot":  f"http://localhost:{args.iot_port}",
                "fmsa": f"http://localhost:{args.fmsa_port}",
                "tsfm": f"http://localhost:{args.tsfm_port}",
                "wo":   f"http://localhost:{args.wo_port}",
            },
            judge_url=f"http://localhost:{args.judge_port}/judge",
            timeout_s=args.timeout,
        )

    if name == "mcpbench":
        from mcp_router.adapters.mcpbench import MCPBenchAdapter

        _BACKEND_BASE_URLS = {
            "ollama":   "http://localhost:11434/v1",
            "vllm":     "http://localhost:8000/v1",
            "lmstudio": "http://localhost:1234/v1",
        }
        _BACKEND_API_KEYS = {
            "ollama":   "ollama",
            "vllm":     "EMPTY",
            "lmstudio": "lm-studio",
        }

        # --local-model is a shortcut: override model + provider + base_url
        provider       = args.provider
        model          = args.model
        local_base_url = None
        local_api_key  = None
        judge_local_base_url = None
        judge_local_api_key  = None
        judge_provider = args.judge_provider
        judge_model    = args.judge_model

        if args.local_model:
            model      = args.local_model
            provider   = "base_url"
            local_base_url = (
                args.local_base_url
                or _BACKEND_BASE_URLS.get(args.local_backend, _BACKEND_BASE_URLS["vllm"])
            )
            local_api_key = (
                args.local_api_key
                or _BACKEND_API_KEYS.get(args.local_backend, "local")
            )

            # Judge: separate local model/server if specified, else reuse planner's
            if args.local_judge_model:
                judge_model          = args.local_judge_model
                judge_provider       = "base_url"
                judge_local_base_url = args.local_judge_base_url or "http://localhost:8001/v1"
                judge_local_api_key  = args.local_judge_api_key or local_api_key
            else:
                judge_model          = model
                judge_provider       = "base_url"
                judge_local_base_url = local_base_url
                judge_local_api_key  = local_api_key

        return MCPBenchAdapter(
            mcpbench_root=args.mcpbench_root,
            provider=provider,
            template_model=model,
            template_base_url=local_base_url,
            template_api_key=local_api_key or os.getenv("OPENROUTER_API_KEY"),
            judge_provider=judge_provider,
            judge_model=judge_model,
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            local_base_url=judge_local_base_url,
            local_api_key=judge_local_api_key,
            task_complexity=args.complexity,
            enable_judge_stability=args.judge_stability,
            timeout_s=args.timeout,
        )

    raise ValueError(
        f"Unknown benchmark '{name}'. "
        f"Add it to _build_adapter() in scripts/run_pipeline.py."
    )


# ── Argument parsing ──────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="mcp_router unified pipeline runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Benchmark selection ──────────────────────────────────────────
    p.add_argument("--benchmark", default="toy",
                   choices=["toy", "assetopsbench", "mcpbench"],
                   help="Which benchmark adapter to use")

    # ── Pipeline stage control ───────────────────────────────────────
    p.add_argument("--stages", default="search,validate",
                   help="Comma-separated list of stages to run: search,validate,react")

    # ── Evolution hyperparams ────────────────────────────────────────
    p.add_argument("--B",            type=int,   default=15,   help="Evolution budget")#15#5
    p.add_argument("--K",            type=int,   default=15,   help="Population cap")#15#10
    p.add_argument("--fb-repeats",   type=int,   default=4,    help="Minibatch repeats")#4#2
    p.add_argument("--alpha",        type=float, default=0.05, help="LCB alpha")
    p.add_argument("--max-steps",    type=int,   default=25,    help="Max tool steps")#25#15
    p.add_argument("--plan-retries", type=int,   default=10,    help="Max LLM re-plan attempts per query on execution failure")#10#3
    p.add_argument("--seed",         type=int,   default=0,    help="RNG seed")
    p.add_argument("--timeout",      type=float, default=30.0, help="Tool call timeout (s)")#30#15

    # ── Output ───────────────────────────────────────────────────────
    p.add_argument("--output-dir", default="results", help="Output directory")

    # ── AssetOpsBench ports ───────────────────────────────────────────
    p.add_argument("--iot-port",   type=int, default=8001)
    p.add_argument("--fmsa-port",  type=int, default=8002)
    p.add_argument("--tsfm-port",  type=int, default=8003)
    p.add_argument("--wo-port",    type=int, default=8004)
    p.add_argument("--judge-port", type=int, default=8005)

    # ── MCP-Bench options ─────────────────────────────────────────────
    p.add_argument("--mcpbench-root", default="./mcp-bench",
                   help="Path to cloned mcp-bench repo")
    p.add_argument("--complexity", default="all",
                   choices=["all", "single", "2server", "3server"],
                   help="Task complexity filter for MCP-Bench")

    p.add_argument("--model", default="qwen/qwen-2.5-72b-instruct",
                   help="Query-planner model")
    p.add_argument("--provider", default="openrouter",
                   choices=["aws", "bedrock", "openrouter", "azure", "base_url"],
                   help="LLM provider for the query planner")

    # ── Local / open-source model options ────────────────────────────────────
    p.add_argument("--local-model", default=None,
                   help="Local model name (e.g. 'meta-llama/Llama-3.2-3B-Instruct'). Sets --provider=base_url automatically.")
    p.add_argument("--local-backend", default="vllm",
                   choices=["ollama", "vllm", "lmstudio"],
                   help="Local serving backend (default: vllm)")
    p.add_argument("--local-base-url", default=None,
                   help="Override base URL for the local backend "
                        "(defaults: vllm=http://localhost:8000/v1, "
                        "ollama=http://localhost:11434/v1, lmstudio=http://localhost:1234/v1)")
    p.add_argument("--local-api-key", default=None,
                   help="API key for the local backend (usually not needed)")

    # ── Local judge model (separate vLLM server) ──────────────────────────────
    p.add_argument("--local-judge-model", default=None,
                   help="Local model for the judge (e.g. 'Qwen/Qwen2.5-7B-Instruct'). "
                        "Runs on a separate vLLM port. Defaults to --local-model if omitted.")
    p.add_argument("--local-judge-base-url", default=None,
                   help="Base URL for the judge's local server (default: http://localhost:8001/v1)")
    p.add_argument("--local-judge-api-key", default=None,
                   help="API key for the judge's local backend (usually not needed)")

    p.add_argument("--judge-model", default="o4-mini",
                   help="LLM model used by the MCP-Bench judge (defaults to --model when omitted)")
    p.add_argument("--judge-provider", default=None,
                   choices=["aws", "bedrock", "openrouter", "azure", "base_url"],
                   help="LLM provider for the judge (defaults to --provider when omitted)")

    p.add_argument("--no-judge-stability", dest="judge_stability", action="store_false",
                   help="Disable judge stability (enabled by default)")
    p.set_defaults(judge_stability=True)

    p.add_argument("--evolve-on-val", dest="evolve_on_val", action="store_true", default=False,
                   help="Run evolutionary search on the validation set instead of a plain single-plan validate. "
                        "Results are stored in val_workflow_history_<model>.jsonl.")

    p.add_argument("--react-max-steps", type=int, default=25, #25#15
                   help="Maximum ReAct Thought/Action/Observation rounds per query (react stage only).")
    p.add_argument("--react-tool-retries", type=int, default=10,
                   help="Consecutive tool-execution failures tolerated before the ReAct loop "
                        "aborts a query. Each failure is fed back as an Observation so the LLM "
                        "can self-correct before the next attempt. (default: 10)")
    p.add_argument("--react-parse-retries", type=int, default=3,
                   help="Consecutive response-parse failures tolerated before aborting a query. "
                        "Each failure injects a format-error Observation so the model can self-correct. (default: 3)")
    p.add_argument("--react-max-tokens", type=int, default=3096,
                   help="Max tokens per LLM call in the ReAct loop. Increase for reasoning models "
                        "that emit long <think> blocks (e.g. DeepSeek-R1). (default: 3096)")

    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = _parse_args()
    stages = {s.strip() for s in args.stages.split(",")}

    print("=" * 64)
    print(f"  mcp_router pipeline  |  benchmark: {args.benchmark}")
    print(f"  stages: {', '.join(sorted(stages))}")
    print("=" * 64)

    from mcp_router.benchmark_adapter import BenchmarkAdapterRunner, RunnerConfig

    adapter = _build_adapter(args)
    cfg = RunnerConfig(
        B=args.B,
        K=args.K,
        max_steps=args.max_steps,
        plan_retries=args.plan_retries,
        alpha=args.alpha,
        seed=args.seed,
        timeout_s=args.timeout,
        output_dir=args.output_dir,
    )
    runner = BenchmarkAdapterRunner(adapter=adapter, cfg=cfg)

    os.makedirs(args.output_dir, exist_ok=True)
    pop = None

    _model_tag = (args.local_model or args.model).split("/")[-1]

    # ── Stage 1: search ───────────────────────────────────────────────
    if "search" in stages:
        print("\n[1/2] Evolutionary search...")

        hist_path = os.path.join(args.output_dir, f"workflow_history_{_model_tag}.jsonl")

        # ── Resume detection ─────────────────────────────────────────────────
        #
        # Collect every (query_id, candidate_id) pair already in history.
        # Any record with both fields non-null counts — no is_final required,
        # so existing files produced by earlier runs are respected.
        # The history file is never rewritten here; new records are appended.
        # Per-candidate skip: a specific candidate is skipped if its (qid, cid)
        # pair is already present, but evolution continues for remaining budget.

        done_pairs: set = set()   # (query_id, candidate_id) already in history
        if os.path.exists(hist_path):
            with open(hist_path) as _hf:
                for _line in _hf:
                    try:
                        rec = json.loads(_line)
                        qid_val = rec.get("query_id")
                        cid_val = rec.get("candidate_id")
                        if qid_val and cid_val:
                            done_pairs.add((qid_val, cid_val))
                    except (json.JSONDecodeError, AttributeError):
                        pass  # corrupt line — skip

        skip_candidates = done_pairs
        if skip_candidates:
            done_qids = {q for q, _ in done_pairs}
            print(f"  Resuming: {len(done_qids)} query_id(s) with "
                  f"{len(skip_candidates)} known candidate(s) — skipping them.")

        # ── Open history file in append mode ──────────────────────────
        _hist_count = [0]
        _hist_file  = open(hist_path, "a")

        def _stream_record(qid, generation, phase, cand, G=None, I=None,
                           exploration_mode=None, parent_score=None, local_best=None):
            """Append one intermediate candidate to workflow_history.jsonl."""
            record = _candidate_to_dict(cand, qid)
            record["_meta"] = {"qid": qid, "generation": generation, "phase": phase,
                               "G": G, "I": I, "exploration_mode": exploration_mode,
                               "parent_score": parent_score, "local_best": local_best,
                               "is_final": False}
            _hist_file.write(json.dumps(record) + "\n")
            _hist_file.flush()
            _hist_count[0] += 1

        def _persist_query_pop(qid, candidates, G=None, I=None):
            """Append the final candidates for one query to workflow_history.jsonl."""
            for c in candidates:
                rec = _candidate_to_dict(c, qid)
                rec["_meta"] = {"G": G, "I": I, "is_final": True}
                _hist_file.write(json.dumps(rec) + "\n")
            _hist_file.flush()

        # Pre-load all existing candidates (including non-final/streamed records)
        # so that skipped-init queries can seed pop without re-running the LLM.
        existing_pop = _load_all_records_pop(hist_path, runner.compiler) if skip_candidates else {}

        try:
            new_pop = runner.run_search(
                on_record      = _stream_record,
                on_query_done  = _persist_query_pop,
                skip_candidates = skip_candidates,
                existing_pop   = existing_pop,
            )
        finally:
            _hist_file.close()

        # Reload the full merged population (skipped + newly run) for downstream stages
        pop = _load_pop(args.output_dir, runner.compiler, os.path.basename(hist_path))
        n_queries = len(pop)
        total     = sum(len(v) for v in pop.values())
        print(f"  {total} candidates across {n_queries} queries ({len(new_pop)} newly processed) → {hist_path}")
        _print_pop_table(pop, n=5)
        print(f"  {_hist_count[0]} new history records → {hist_path}")

    # ── Stage 2: validate ─────────────────────────────────────────────
    if "validate" in stages:
        print("\n[2/2] Validation...")

        val_hist_path = os.path.join(args.output_dir, f"val_workflow_history_{_model_tag}.jsonl")

        if args.evolve_on_val:
            # ── Evolutionary search on the validation set ─────────────
            print("  (evolve-on-val: running evolutionary search on D_val)")

            _, D_val = runner.adapter.load_datasets()

            # Resume detection — same logic as search stage
            done_pairs: set = set()
            if os.path.exists(val_hist_path):
                with open(val_hist_path) as _hf:
                    for _line in _hf:
                        try:
                            rec = json.loads(_line)
                            qid_val = rec.get("query_id")
                            cid_val = rec.get("candidate_id")
                            if qid_val and cid_val:
                                done_pairs.add((qid_val, cid_val))
                        except (json.JSONDecodeError, AttributeError):
                            pass

            existing_val_pop = _load_all_records_pop(val_hist_path, runner.compiler) if done_pairs else {}

            _val_hist_count = [0]
            _val_file = open(val_hist_path, "a")

            def _val_stream_record(qid, generation, phase, cand, G=None, I=None,
                                   exploration_mode=None, parent_score=None, local_best=None):
                record = _candidate_to_dict(cand, qid)
                record["_meta"] = {"qid": qid, "generation": generation, "phase": phase,
                                   "G": G, "I": I, "exploration_mode": exploration_mode,
                                   "parent_score": parent_score, "local_best": local_best,
                                   "is_final": False}
                _val_file.write(json.dumps(record) + "\n")
                _val_file.flush()
                _val_hist_count[0] += 1

            def _val_persist_query_pop(qid, candidates, G=None, I=None):
                for c in candidates:
                    rec = _candidate_to_dict(c, qid)
                    rec["_meta"] = {"G": G, "I": I, "is_final": True, "phase": "validate"}
                    _val_file.write(json.dumps(rec) + "\n")
                _val_file.flush()

            try:
                runner.run_search(
                    on_record       = _val_stream_record,
                    on_query_done   = _val_persist_query_pop,
                    skip_candidates = done_pairs,
                    existing_pop    = existing_val_pop,
                    D_override      = D_val,
                )
            finally:
                _val_file.close()

            val_pop = _load_pop(args.output_dir, runner.compiler, os.path.basename(val_hist_path))
            print(f"  {_val_hist_count[0]} new history records → {val_hist_path}")
            _print_pop_table(val_pop, n=5)

        else:
            # ── Standard single-plan validation ───────────────────────
            _val_file = open(val_hist_path, "a")

            def _val_record(qid, cand):
                record = _candidate_to_dict(cand, qid)
                record["_meta"] = {"is_final": True, "phase": "validate"}
                _val_file.write(json.dumps(record) + "\n")
                _val_file.flush()

            try:
                results = runner.validate(on_record=_val_record)
            finally:
                _val_file.close()

            report_path = os.path.join(args.output_dir, "validation_report.json")
            with open(report_path, "w") as f:
                json.dump({"benchmark": args.benchmark, "results": results}, f, indent=2)
            print(f"  Validation report    → {report_path}")
            _print_val_table(results, n=10)

        print(f"  Val workflow history → {val_hist_path}")

    # ── Stage 3: react ────────────────────────────────────────────────
    if "react" in stages:
        print("\n[3/3] ReAct validation...")

        from mcp_router.react_validate import ReactValidationRunner

        react_hist_path   = os.path.join(args.output_dir, f"react_val_history_{_model_tag}.jsonl")
        react_report_path = os.path.join(args.output_dir, f"react_validation_report_{_model_tag}.json")

        # ── Load D_val ────────────────────────────────────────────────
        _, D_val_react = runner.adapter.load_datasets()
        print(f"  ReAct queries: {len(D_val_react)}   max_steps: {args.react_max_steps}   tool_retries: {args.react_tool_retries}")

        # ── Streaming JSONL writer (same pattern as validate stage) ───
        _react_file = open(react_hist_path, "a")

        def _react_record(_, rec: dict) -> None:
            """Persist one finished ReAct trajectory immediately."""
            rec["_meta"] = {"is_final": True, "phase": "react"}
            _react_file.write(json.dumps(rec) + "\n")
            _react_file.flush()

        # ── Run ReAct ─────────────────────────────────────────────────
        # Uses the same adapter (and therefore the same LLMJudge wiring)
        # as the validate stage — scores are directly comparable.
        react_runner = ReactValidationRunner(
            adapter        = runner.adapter,
            planner        = runner.planner,
            max_steps      = args.react_max_steps,
            tool_retries   = args.react_tool_retries,
            parse_retries  = args.react_parse_retries,
            max_tokens     = args.react_max_tokens,
            on_record      = _react_record,
        )

        try:
            react_results = react_runner.run(D_val=D_val_react)
        finally:
            _react_file.close()

        # ── Report ────────────────────────────────────────────────────
        mean_score = (sum(r["score"] for r in react_results) / len(react_results)
                      if react_results else 0.0)
        ok_rate    = (sum(1 for r in react_results if r.get("ok")) / len(react_results)
                      if react_results else 0.0)

        with open(react_report_path, "w") as _rf:
            json.dump({
                "benchmark":  args.benchmark,
                "model":      _model_tag,
                "max_steps":  args.react_max_steps,
                "mean_score": mean_score,
                "ok_rate":    ok_rate,
                "results":    react_results,
            }, _rf, indent=2)

        _print_react_table(react_results, n=10)
        print(f"  Mean score: {mean_score:.4f}   OK rate: {ok_rate:.1%}")
        print(f"  React history → {react_hist_path}")
        print(f"  React report  → {react_report_path}")

    print("\nDone.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_pop_table(pop):
    print(f"\n  {'Rank':<5} {'action_id':<18} {'LCB':>7} {'Mean':>7}  template")
    print("  " + "-" * 55)
    for i, c in enumerate(pop, 1):
        qtext = str(c.action.q.get("text", c.wf.action_id))[:30]
        print(f"  {i:<5} {c.wf.action_id:<18} "
              f"{c.score_lcb:>7.3f} {c.score_mean:>7.3f}  {qtext}")


def _print_val_table(results):
    print(f"\n  {'Rank':<5} {'action_id':<18} {'LCB':>7} {'Success':>8} {'Violation':>10}")
    print("  " + "-" * 55)
    for i, r in enumerate(results, 1):
        print(f"  {i:<5} {r['action_id']:<18} "
              f"{r['score_lcb']:>7.3f} {r['success_rate']:>8.1%} "
              f"{r['violation_rate']:>10.1%}")


def _load_pop(output_dir, compiler):
    """Reload population from disk for export/validate when search was skipped."""
    from mcp_router.dsl import Action, PromptParams, TypedEdit
    from mcp_router.evolution import Candidate

    pop_path = os.path.join(output_dir, "population.jsonl")
    if not os.path.exists(pop_path):
        print(f"[error] {pop_path} not found. Run with --stages search first.")
        sys.exit(1)

    pop = []
    with open(pop_path) as f:
        for line in f:
            rec = json.loads(line)
            a = Action(
                q=rec["action"]["q"],
                e=tuple(TypedEdit(op=e["op"], args=e["args"]) for e in rec["action"]["e"]),
                p=PromptParams(**rec["action"]["p"]),
            )
            try:
                wf = compiler.compile(a)
            except Exception:
                continue
            pop.append(Candidate(
                action=a, wf=wf,
                score_lcb=rec["score_lcb"], score_mean=rec["score_mean"],
                feasible=True, reasons=[],
            ))
    return pop

# ── Serialisation helpers ─────────────────────────────────────────────────────
 
def _serialize_graph(g) -> dict:
    """
    Serialise a GraphTemplate or CompiledWorkflow to a JSON-safe dict.
    Captures description, edges, and full per-node detail
    (server, tool, params, produces, requires).
    Returns None if g is None.
    """
    if g is None:
        return None
    return {
        "description": getattr(g, "description", ""),
        "heuristic": getattr(g, "heuristic", False),
        "nodes": [
            {
                "node_id":  n.node_id,
                "kind":     n.kind,
                "server":   n.tool_ref[0] if n.tool_ref else None,
                "tool":     n.tool_ref[1] if n.tool_ref else None,
                "params":   _safe_json(dict(n.params)   if n.params   else {}),
                "produces": _safe_json(dict(n.produces) if n.produces else {}),
                "requires": list(n.requires) if n.requires else [],
            }
            for n in g.nodes
        ],
        "edges": [list(e) for e in g.edges],
    }


def _exec_result_to_dict(res) -> dict:
    """
    Convert an ExecResult to a JSON-serialisable dict.
    Captures ok, violation, final_output, and per-step logs.
    input_args is intentionally omitted — it can be large and is
    reconstructable from the query + node params.
    """
    if res is None:
        return None
    return {
        "ok":           res.ok,
        "violation":    getattr(res, "violation", None),
        "final_output": _safe_json(res.final_output),
        "logs": [
            {
                "node_id":    log.node_id,
                "kind":       log.kind,
                "tool":       log.tool,
                "ok":         log.ok,
                "error_type": log.error_type,
                "latency_ms": log.latency_ms,
                "output":     _safe_json(log.output),
            }
            for log in (res.logs or [])
        ],
    }
 
 
def _safe_json(obj):
    """Best-effort conversion of an arbitrary object to a JSON-safe value."""
    if obj is None:
        return None
    if isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(v) for v in obj]
    return str(obj)
 
 
def _metric_breakdown_to_dict(mb):
    """Serialise a MetricBreakdown to a JSON-safe dict. Returns None if mb is None."""
    if mb is None:
        return None
    # Use __dict__ if available (dataclass), otherwise fall back field-by-field
    try:
        return {k: _safe_json(v) for k, v in mb.__dict__.items()}
    except AttributeError:
        pass
    # Fallback: known fields from the extended MetricBreakdown
    fields = [
        "task_fulfillment", "grounding",
        "tool_appropriateness", "parameter_accuracy",
        "dependency_awareness", "parallelism_and_efficiency",
        "task_completion_score", "tool_selection_score",
        "planning_effectiveness_and_efficiency_score",
        "task_fulfillment_reasoning", "grounding_reasoning",
        "tool_appropriateness_reasoning", "parameter_accuracy_reasoning",
        "dependency_awareness_reasoning", "parallelism_and_efficiency_reasoning",
    ]
    return {f: _safe_json(getattr(mb, f, None)) for f in fields}
 
 
def _candidate_to_dict(c, query_id: str) -> dict:
    """Serialise one Candidate to a JSON-serialisable dict.

    Includes:
      - candidate_id             — stable hash of (query, edits, policy)
      - All MetricBreakdown sub-scores and reasoning strings (score_detail)
      - Scalar mean score
      - workflow                 — compiled graph (nodes with server/tool/params/produces)
      - base_plan                — original LLM GraphTemplate before any edits
      - edit_history             — list of TypedEdits applied from base_plan → workflow
      - error_context            — last known failure reason used by mutation LLM
      - action.graph.token_cost  — LLM token usage for the base plan
      - action.token_cost        — accumulated token usage across all edits
    """
    return {
        "candidate_id":        getattr(c, "candidate_id", None) or c.action.stable_id(),
        "query_id":            query_id,
        "action_id":           c.wf.action_id if c.wf else None,
        "score":               c.score,
        "feasible":            c.feasible,
        "feasibility_reasons": c.feasibility_reasons,
        "plan_attempts":       getattr(c, "plan_attempts", 1),
        "score_detail":        _metric_breakdown_to_dict(getattr(c, "score_detail", None)),
        "token_cost":          _safe_json(getattr(c.action, "token_cost", None)),
        "workflow":            _serialize_graph(c.wf),
        "base_plan":           _serialize_graph(getattr(c.action, "graph", None)),
        "edit_history":        [{"op": e.op, "args": e.args} for e in c.action.e],
        "error_context":       getattr(c.action, "error_context", None),
        "action": {
            "q": c.action.q,
            "e": [{"op": e.op, "args": e.args} for e in c.action.e],
            "p": c.action.p.__dict__,
        },
        "execution_result":    _exec_result_to_dict(c.execution_result),
    }
 
 
# ── Save / load population + history ─────────────────────────────────────────

def _load_pop(output_dir: str, compiler, hist_filename: str = "workflow_history.jsonl") -> dict:
    """
    Reload Dict[query_id, List[Candidate]] from workflow_history.jsonl.
    Only records with _meta["is_final"]=True are loaded.
    Candidates whose workflow cannot be recompiled are skipped.
    """
    from mcp_router.dsl import Action, PromptParams, TypedEdit
    from mcp_router.evolution import Candidate

    pop_path = os.path.join(output_dir, hist_filename)
    if not os.path.exists(pop_path):
        print(f"[error] {pop_path} not found. Run with --stages search first.")
        sys.exit(1)
 
    # Pass 1 — collect records, deduplicate by (query_id, candidate_id).
    # is_final=True records take priority; for old files with no is_final flag
    # all unique (qid, cid) records are used.
    seen: dict = {}        # (qid, cid) -> rec
    any_is_final = False
    with open(pop_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = rec.get("query_id")
            cid = rec.get("candidate_id")
            if not (qid and cid):
                continue
            is_final = rec.get("_meta", {}).get("is_final")
            if is_final:
                any_is_final = True
                seen[(qid, cid)] = rec          # always overwrite with final
            elif (qid, cid) not in seen:
                seen[(qid, cid)] = rec          # keep first non-final as fallback

    # If the file has any is_final records, load only those (new format).
    # Otherwise load all unique records (old format — intermediates are the only signal).
    records_to_load = [
        rec for rec in seen.values()
        if (not any_is_final) or rec.get("_meta", {}).get("is_final")
    ]

    pop: dict = {}
    for rec in records_to_load:
        qid = rec["query_id"]
        a   = Action(
            q=rec["action"]["q"],
            e=tuple(
                TypedEdit(op=e["op"], args=e["args"])
                for e in rec["action"]["e"]
            ),
            p=PromptParams(**rec["action"]["p"]),
        )
        try:
            wf = compiler.compile(a)
        except Exception:
            wf = None

        # Reconstruct a lightweight ExecResult from stored logs so
        # evidence collection works after reload, without re-running.
        exec_res = _dict_to_exec_result(rec.get("execution_result"))

        cand = Candidate(
            action=a,
            wf=wf,
            score=rec.get("score", 0.0),
            score_detail=_dict_to_metric_breakdown(rec.get("score_detail")),
            feasible=rec.get("feasible", wf is not None),
            feasibility_reasons=rec.get("feasibility_reasons", []),
            execution_result=exec_res,
        )
        pop.setdefault(qid, []).append(cand)

    return pop


def _load_all_records_pop(hist_path: str, compiler) -> dict:
    """
    Load ALL unique (query_id, candidate_id) records from workflow_history.jsonl
    regardless of is_final flag.  Used to seed existing_pop before a resumed run
    so that queries whose init candidate is already in history can skip the LLM
    planner and executor entirely.
    """
    from mcp_router.dsl import Action, PromptParams, TypedEdit
    from mcp_router.evolution import Candidate

    if not os.path.exists(hist_path):
        return {}

    seen: dict = {}   # (qid, cid) -> rec; is_final records overwrite non-final
    with open(hist_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = rec.get("query_id")
            cid = rec.get("candidate_id")
            if not (qid and cid):
                continue
            is_final = rec.get("_meta", {}).get("is_final")
            if is_final or (qid, cid) not in seen:
                seen[(qid, cid)] = rec

    pop: dict = {}
    for rec in seen.values():
        qid = rec["query_id"]
        cid = rec.get("candidate_id")
        a = Action(
            q=rec["action"]["q"],
            e=tuple(TypedEdit(op=e["op"], args=e["args"]) for e in rec["action"]["e"]),
            p=PromptParams(**rec["action"]["p"]),
        )
        try:
            wf = compiler.compile(a)
        except Exception:
            wf = None
        exec_res = _dict_to_exec_result(rec.get("execution_result"))
        cand = Candidate(
            action=a,
            wf=wf,
            score=rec.get("score", 0.0),
            score_detail=_dict_to_metric_breakdown(rec.get("score_detail")),
            feasible=rec.get("feasible", wf is not None),
            feasibility_reasons=rec.get("feasibility_reasons", []),
            execution_result=exec_res,
            candidate_id=cid,
        )
        pop.setdefault(qid, []).append(cand)

    return pop

 
def _dict_to_exec_result(d: dict):
    """
    Reconstruct a minimal ExecResult from a serialised dict so that
    _collect_evidence() can read log.ok / log.node_id after a reload.
    Returns None if d is None or malformed.
    """
    if not d:
        return None
    try:
        from mcp_router.executor import ExecResult, StepLog
        logs = [
            StepLog(
                node_id    = lg["node_id"],
                kind       = lg["kind"],
                tool       = lg.get("tool"),
                ok         = lg["ok"],
                error_type = lg.get("error_type"),
                latency_ms = lg.get("latency_ms", 0),
                output     = lg.get("output"),
            )
            for lg in d.get("logs", [])
        ]
        return ExecResult(
            ok           = d["ok"],
            logs         = logs,
            final_output = d.get("final_output"),
            violation    = d.get("violation"),
        )
    except Exception:
        return None
 
def _dict_to_metric_breakdown(d: dict):
    """
    Reconstruct a MetricBreakdown from a serialised dict.
    Returns None if d is None or MetricBreakdown is not importable.
    All unknown fields are silently ignored so old population files load cleanly.
    """
    if not d:
        return None
    try:
        from mcp_router.executor import MetricBreakdown
        import inspect
        # Only pass fields that the current MetricBreakdown dataclass accepts
        valid = {f for f in inspect.signature(MetricBreakdown).parameters}
        kwargs = {k: v for k, v in d.items() if k in valid}
        return MetricBreakdown(**kwargs)
    except Exception:
        return None
    
# ── Display helpers ───────────────────────────────────────────────────────────
 
def _print_pop_table(pop: dict, n: int = 5):
    """Print the top-n candidates across all queries."""
    all_cands = [
        (qid, c) for qid, cands in pop.items() for c in cands
    ]
    all_cands.sort(key=lambda x: x[1].score, reverse=True)
 
    print(f"\n  {'Rank':<5} {'query_id':<14} {'action_id':<18} {'Score':>7}  task_snippet")
    print("  " + "-" * 70)
    for i, (qid, c) in enumerate(all_cands[:n], 1):
        aid   = c.wf.action_id[:16] if c.wf else "(no wf)"
        qtext = str(c.action.q.get("text", qid))[:30]
        print(f"  {i:<5} {str(qid):<14} {aid:<18} {c.score:>7.3f}  {qtext}")
 
 
def _print_val_table(results, n: int = 10):
    print(f"\n  {'Rank':<5} {'query_id':<18} {'action_id':<18} {'Score':>7} {'OK':>5} {'Violation':>12}")
    print("  " + "-" * 70)
    for i, r in enumerate(results[:n], 1):
        print(
            f"  {i:<5} {str(r.get('query_id', '')):<18} "
            f"{str(r.get('action_id', '')):<18} "
            f"{r.get('score', 0.0):>7.3f} "
            f"{str(r.get('ok', '?')):>5} "
            f"{str(r.get('violation') or ''):>12}"
        )


def _print_react_table(results, n: int = 10):
    print(f"\n  {'Rank':<5} {'query_id':<18} {'Score':>7} {'Steps':>6} {'OK':>5} {'Violation':>12}")
    print("  " + "-" * 60)
    for i, r in enumerate(results[:n], 1):
        print(
            f"  {i:<5} {str(r.get('query_id', '')):<18} "
            f"{r.get('score', 0.0):>7.3f} "
            f"{r.get('n_steps', 0):>6} "
            f"{str(r.get('ok', '?')):>5} "
            f"{str(r.get('violation') or ''):>12}"
        )


if __name__ == "__main__":
    main()
# PYTHONPATH=. python3 -m scripts.run_pipeline --benchmark mcpbench > test.txt