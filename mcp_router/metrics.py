from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np

from .executor import ExecResult, StepLog, MetricBreakdown
from .config import ScalarWeights



# ---------------------------------------------------------------------------
# Structural proxy — used when no adapter judge is available
# ---------------------------------------------------------------------------

def compute_metrics(res: ExecResult) -> MetricBreakdown:
    tool_logs = [l for l in res.logs if l.kind == "tool"]
    val_logs  = [l for l in res.logs if l.kind == "validator"]
    #latency_ms = float(np.sum([l.latency_ms for l in res.logs])) if res.logs else 0.0

    return MetricBreakdown(
        task_completion = 10.0 if res.ok else 1.0,
        tool_selection  = 10.0 if all(l.ok for l in tool_logs) else 1.0,
        planning        = 10.0 if all(l.ok for l in val_logs)  else 1.0,
        parallelism     = 5.5,
        #latency_ms      = latency_ms,
    )


def compute_step_metrics(logs: List[StepLog]) -> Dict[str, Dict[str, Any]]:
    """
    Per-step breakdown for richer reporting in validate().
    Returns a dict keyed by node_id with ok, error_type, latency_ms.
    """
    return {
        l.node_id: {
            "ok":         l.ok,
            "error_type": l.error_type,
            "latency_ms": l.latency_ms,
            "tool":       l.tool,
        }
        for l in logs
    }


# ---------------------------------------------------------------------------
# Scalarization
# ---------------------------------------------------------------------------

def scalarize(m: MetricBreakdown, w: ScalarWeights = ScalarWeights()) -> float:
    """
    Linear combination of MetricBreakdown dimensions → scalar reward J.

    Weights are configurable via ScalarWeights in config.py.
    Used by _default_scoring_fn in executor.py as the built-in fallback.
    Adapter judges that return a MetricBreakdown are also scalarized here,
    so weight tuning applies uniformly regardless of scoring source.
    """
    return (
        w.task_completion * m.task_completion
        + w.tool_selection * m.tool_selection
        + w.planning       * m.planning
        + w.parallelism    * m.parallelism
        + w.latency_per_ms * m.latency_ms
    )