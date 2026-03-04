from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class MetricSummary:
    mean: float
    std: float
    ci_low: float
    ci_high: float


def bootstrap_ci(values: list[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 7) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    means = []
    n = len(arr)
    for _ in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        means.append(float(np.mean(sample)))
    means = np.array(means, dtype=float)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def summarize_metric(values: list[float]) -> MetricSummary:
    if not values:
        return MetricSummary(0.0, 0.0, 0.0, 0.0)
    mean = float(np.mean(values))
    std = float(np.std(values))
    ci_low, ci_high = bootstrap_ci(values)
    return MetricSummary(mean=mean, std=std, ci_low=ci_low, ci_high=ci_high)


def paired_permutation_pvalue(a: list[float], b: list[float], trials: int = 4000, seed: int = 11) -> float:
    if not a or not b or len(a) != len(b):
        return 1.0
    rng = np.random.default_rng(seed)
    diff = np.array(a, dtype=float) - np.array(b, dtype=float)
    observed = abs(float(np.mean(diff)))
    count = 0
    for _ in range(trials):
        signs = rng.choice([-1, 1], size=len(diff))
        perm = diff * signs
        stat = abs(float(np.mean(perm)))
        if stat >= observed:
            count += 1
    return (count + 1) / (trials + 1)


def cohen_d(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    am = float(np.mean(a))
    bm = float(np.mean(b))
    av = float(np.var(a))
    bv = float(np.var(b))
    pooled = math.sqrt((av + bv) / 2.0) if (av + bv) > 0 else 1.0
    return (am - bm) / pooled


def holm_bonferroni_adjust(pvals: dict[str, float]) -> dict[str, float]:
    """
    Return Holm-Bonferroni adjusted p-values by metric name.
    """
    if not pvals:
        return {}
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for i, (name, p) in enumerate(items, start=1):
        adj = min(1.0, (m - i + 1) * p)
        running_max = max(running_max, adj)
        adjusted[name] = running_max
    return adjusted
