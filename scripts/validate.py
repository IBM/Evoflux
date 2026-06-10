"""
scripts/validate.py
-------------------
Evaluate the final population on D_val under perturbations.
Reports per-candidate LCB, mean reward, success rate, violation rate,
latency, and per-node error counts.

NOTE: The preferred way to run this stage is via run_pipeline.py:
    PYTHONPATH=. python3 -m scripts.run_pipeline --stages validate

This standalone script lets you re-run validation after search has
already completed, optionally with a different adapter or noise level.

Usage:
    PYTHONPATH=. python3 -m scripts.validate [--benchmark toy]

Reads:  results/workflow_history.jsonl  (records with _meta["is_final"]=True)
Output: results/validation_report.json
"""
from __future__ import annotations
import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

from mcp_router.benchmark_adapter import BenchmarkAdapterRunner, RunnerConfig


def _load_adapter(benchmark: str):
    if benchmark == "toy":
        from mcp_router.adapters.toy import ToyAdapter
        return ToyAdapter()
    if benchmark == "assetopsbench":
        from mcp_router.adapters.assetopsbench import AssetOpsBenchAdapter
        return AssetOpsBenchAdapter()
    if benchmark == "mcpbench":
        from mcp_router.adapters.mcpbench import MCPBenchAdapter
        return MCPBenchAdapter()
    raise ValueError(f"Unknown benchmark: {benchmark}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="toy",
                   choices=["toy", "assetopsbench", "mcpbench"])
    p.add_argument("--pop-path",    default="results/workflow_history.jsonl")
    p.add_argument("--output-dir",  default="results")
    p.add_argument("--alpha",       type=float, default=0.05)
    p.add_argument("--max-steps",   type=int,   default=8)
    args = p.parse_args()

    print("=" * 60)
    print("MCP Router — Validation")
    print("=" * 60)

    if not Path(args.pop_path).exists():
        print(f"[error] {args.pop_path} not found.")
        print("  Run the search stage first:")
        print("    PYTHONPATH=. python3 -m scripts.run_pipeline --stages search")
        return

    adapter = _load_adapter(args.benchmark)
    cfg = RunnerConfig(
        alpha=args.alpha,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
    )
    runner = BenchmarkAdapterRunner(adapter=adapter, cfg=cfg)

    # Reload D_val from adapter
    _, _, D_val = adapter.load_datasets()
    runner._D_val = D_val

    # Reconstruct population from disk
    from mcp_router.dsl import Action, PromptParams, TypedEdit
    from mcp_router.evolution import Candidate

    pop = []
    with open(args.pop_path) as f:
        for line in f:
            rec = json.loads(line)
            if not rec.get("_meta", {}).get("is_final"):
                continue
            a = Action(
                g=rec["action"]["g"],
                e=tuple(TypedEdit(op=e["op"], args=e["args"]) for e in rec["action"]["e"]),
                p=PromptParams(**rec["action"]["p"]),
            )
            try:
                wf = runner.compiler.compile(a)
            except Exception as exc:
                print(f"[warn] skipping {rec['action_id']}: {exc}")
                continue
            pop.append(Candidate(
                action=a, wf=wf,
                score_lcb=rec["score_lcb"], score_mean=rec["score_mean"],
                feasible=True, reasons=[],
            ))

    print(f"[validate] {len(pop)} candidates, {len(D_val)} val queries, repeats=3")

    # runner.validate() uses execu.score() → adapter judge or structural proxy
    results = runner.validate(pop=pop, D_val=D_val)

    # ── Report ──────────────────────────────────────────────────────────────
    print(f"\n{'Rank':<5} {'action_id':<18} {'LCB':>7} {'Mean':>7} "
          f"{'Success':>8} {'Violation':>10} {'Latency':>10}")
    print("-" * 70)
    for rank, r in enumerate(results, 1):
        print(
            f"{rank:<5} {r['action_id']:<18} "
            f"{r['score_lcb']:>7.3f} {r['score_mean']:>7.3f} "
            f"{r['success_rate']:>8.1%} {r['violation_rate']:>10.1%} "
            f"{r['mean_latency_ms']:>8.0f}ms"
        )

    # ── Step-level error table for the best candidate ───────────────────────
    if results:
        best = results[0]
        print(f"\nBest: {best['action_id']}  LCB={best['score_lcb']:.4f}")
        errs = best.get("step_error_counts", {})
        if errs:
            print("  Step-level errors:")
            for node_id, cnt in sorted(errs.items(), key=lambda x: -x[1]):
                print(f"    {node_id:<12}  {cnt} errors")
        else:
            print("  No step-level errors recorded.")

    # ── Save ────────────────────────────────────────────────────────────────
    out_path = Path(args.output_dir) / "validation_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({"benchmark": args.benchmark, "results": results}, f, indent=2)
    print(f"\n[validate] Report → {out_path}")


if __name__ == "__main__":
    main()
