from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalMetrics:
    task_success_rate: float
    retrieval_precision_at_k: float
    retrieval_recall_at_k: float
    hallucination_rate: float
    memory_bloat: float
    adaptation_latency: float
    stability_score: float


def compute_metrics(records: list[dict]) -> EvalMetrics:
    if not records:
        return EvalMetrics(0, 0, 0, 1, 0, 0, 0)
    successes = sum(1 for r in records if r["success"])
    precision = sum(r["precision"] for r in records) / len(records)
    recall = sum(r["recall"] for r in records) / len(records)
    hallucinations = sum(1 for r in records if r["hallucinated"])
    bloat = records[-1]["memory_items"]
    latency_vals = [r["latency"] for r in records if r["latency"] is not None]
    adaptation_latency = sum(latency_vals) / len(latency_vals) if latency_vals else 0.0
    mean_success = successes / len(records)
    variance = sum((((1.0 if r["success"] else 0.0) - mean_success) ** 2) for r in records) / len(records)
    stability = 1.0 - variance
    return EvalMetrics(
        task_success_rate=mean_success,
        retrieval_precision_at_k=precision,
        retrieval_recall_at_k=recall,
        hallucination_rate=hallucinations / len(records),
        memory_bloat=float(bloat),
        adaptation_latency=adaptation_latency,
        stability_score=stability,
    )
