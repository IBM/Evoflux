from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
import math


@dataclass(frozen=True)
class RobustEstimate:
    mean: float
    lcb: float
    n: int


def hoeffding_lcb(samples: List[float], alpha: float, min_val: float = -10.0, max_val: float = 10.0) -> RobustEstimate:
    x = np.array(samples, dtype=float)
    if min_val is None: min_val = float(x.min())
    if max_val is None: max_val = float(x.max())
    if max_val == min_val: max_val = min_val + 1.0  # guard

    n = len(samples)
    if n == 0:
        return RobustEstimate(mean=float("-inf"), lcb=float("-inf"), n=0)
    mean = float(x.mean())
    # Hoeffding: P(|mean - E| >= eps) <= 2 exp(-2 n eps^2 / (b-a)^2)
    # One-sided LCB uses exp(-2 n eps^2 / (b-a)^2) = alpha => eps = (b-a) * sqrt(log(1/alpha)/(2n))
    rng = (max_val - min_val)
    eps = rng * math.sqrt(math.log(1.0 / max(alpha, 1e-12)) / (2.0 * n))
    return RobustEstimate(mean=mean, lcb=mean - eps, n=n)