from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Budgets:
    B: int            # total evolution iterations
    K: int            # population size cap
    max_steps: int    # max tool calls per workflow execution
    plan_retries: int = 3  # max LLM re-plan attempts per query on execution failure


@dataclass(frozen=True)
class RobustConfig:
    alpha: float        # LCB confidence level (e.g., 0.05)
    method: str = "hoeffding"


@dataclass(frozen=True)
class ScalarWeights:
    task_completion: float = 1.0
    tool_selection: float = 1.0
    planning: float = 1.0
    # Negative value: violation_rate=1.0 subtracts this magnitude from the score.
    # Keep negative. Rename from 'violation' to 'violation_penalty' makes
    # the sign intent explicit at every call site.
    parallelism: float = 1.0
    #latency_per_ms: float = 1.0


@dataclass(frozen=True)
class AdaEvolveConfig:
    """
    Hyperparameters for the AdaEvolve adaptive exploration/exploitation mechanism.

    Exploration intensity I_t per query is computed as:
        I_t = I_min + (I_max - I_min) / (1 + sqrt(G_t + eps))
    where G_t is an exponential moving average of squared relative improvement:
        G_t = rho * G_{t-1} + (1 - rho) * delta_t^2
        delta_t = max((child_score - local_best) / local_best, 0)

    High G (active progress)  → I_t → I_min  (exploit)
    Low  G (stagnation)       → I_t → I_max  (explore)

    When G_t < tau_m and no improvement has occurred for meta_cooldown_frac of
    the budget, a high-level meta-guidance LLM call proposes a qualitatively
    different strategy to escape the local optimum.
    """
    rho: float = 0.9              # EMA decay factor for growth signal
    I_min: float = 0.1            # minimum exploration probability
    I_max: float = 0.7            # maximum exploration probability
    tau_m: float = 0.12           # meta-guidance stagnation threshold on G_t
    meta_cooldown_frac: float = 0.2  # fraction of B between meta-guidance calls
    eps: float = 1e-8             # numerical stability in intensity formula


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 123
    timeout_s: float = 30.0
    cache_enabled: bool = True