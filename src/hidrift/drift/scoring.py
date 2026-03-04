from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DriftWeights:
    alpha: float = 0.45
    beta: float = 0.30
    gamma: float = 0.25


class DriftScorer:
    def __init__(self, weights: DriftWeights | None = None) -> None:
        self.weights = weights or DriftWeights()

    def score(self, behavioral_shift: float, task_shift: float, performance_drop: float) -> float:
        return (
            self.weights.alpha * behavioral_shift
            + self.weights.beta * task_shift
            + self.weights.gamma * performance_drop
        )

