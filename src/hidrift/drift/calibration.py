from __future__ import annotations


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = min(max(int(round((len(vals) - 1) * p)), 0), len(vals) - 1)
    return vals[idx]


def calibrate_threshold(no_drift_scores: list[float], quantile: float = 0.95) -> float:
    return percentile(no_drift_scores, quantile)

