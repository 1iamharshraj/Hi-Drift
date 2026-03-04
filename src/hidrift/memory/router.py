from __future__ import annotations

from dataclasses import dataclass

from hidrift.utils import sigmoid


@dataclass
class ImportanceWeights:
    recency: float = 0.25
    usage: float = 0.25
    reward: float = 0.30
    stability: float = 0.20


class MemoryRouter:
    def __init__(self, weights: ImportanceWeights | None = None) -> None:
        self.weights = weights or ImportanceWeights()

    def importance_score(
        self,
        recency: float,
        usage: float,
        reward: float,
        stability: float,
    ) -> float:
        return (
            self.weights.recency * recency
            + self.weights.usage * usage
            + self.weights.reward * reward
            + self.weights.stability * stability
        )

    def keep_probability(self, importance: float, drift_penalty: float) -> float:
        return sigmoid(importance - drift_penalty)

