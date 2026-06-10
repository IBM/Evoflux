from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import random


@dataclass(frozen=True)
class Perturbation:
    timeout_inject_p: float = 0.02
    tool_error_inject_p: float = 0.02
    latency_jitter_ms: int = 200


class PerturbationSampler:
    def __init__(self, base: Perturbation, seed: int):
        self.base = base
        self.rng = random.Random(seed)

    def sample(self) -> Perturbation:
        # Here perturbations are stationary; that can be expand later with non-stationary regimes
        # Sample could vary probabilities slightly if desired
        return self.base

    def should_timeout(self, p: float) -> bool:
        return self.rng.random() < p

    def should_error(self, p: float) -> bool:
        return self.rng.random() < p

    def jitter_ms(self, max_jitter: int) -> int:
        return self.rng.randint(0, max_jitter)