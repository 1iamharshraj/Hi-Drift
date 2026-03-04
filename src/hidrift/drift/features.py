from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from hidrift.utils import kl_divergence, l2_distance


@dataclass
class OnlineDriftState:
    embedding_mean: list[float] | None = None
    n_embeddings: int = 0
    task_hist: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    perf_window: list[float] = field(default_factory=list)
    perf_window_size: int = 20

    def update_embedding(self, emb: list[float]) -> float:
        if self.embedding_mean is None:
            self.embedding_mean = emb[:]
            self.n_embeddings = 1
            return 0.0
        distance = l2_distance(emb, self.embedding_mean)
        self.n_embeddings += 1
        rate = 1.0 / self.n_embeddings
        self.embedding_mean = [(1 - rate) * m + rate * e for m, e in zip(self.embedding_mean, emb)]
        return distance

    def update_task_distribution(self, task_label: str | None) -> tuple[dict[str, float], dict[str, float], float]:
        label = task_label or "unknown"
        prior = self.normalized_hist()
        self.task_hist[label] += 1.0
        post = self.normalized_hist()
        kl = kl_divergence(post, prior) if prior else 0.0
        return post, prior, kl

    def normalized_hist(self) -> dict[str, float]:
        total = sum(self.task_hist.values())
        if total <= 0:
            return {}
        return {k: v / total for k, v in self.task_hist.items()}

    def update_performance(self, reward: float | None) -> float:
        value = reward if reward is not None else 0.0
        self.perf_window.append(value)
        if len(self.perf_window) > self.perf_window_size:
            self.perf_window.pop(0)
        if len(self.perf_window) < 2:
            return 0.0
        midpoint = max(1, len(self.perf_window) // 2)
        prev = self.perf_window[:midpoint]
        recent = self.perf_window[midpoint:]
        prev_avg = sum(prev) / len(prev)
        recent_avg = sum(recent) / len(recent)
        return max(prev_avg - recent_avg, 0.0)

