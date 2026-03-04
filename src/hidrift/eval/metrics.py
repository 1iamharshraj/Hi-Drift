from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalMetrics:
    task_success_rate: float
    retrieval_precision_at_k: float
    retrieval_recall_at_k: float
    hallucination_rate: float
    constraint_violation_rate: float
    memory_bloat: float
    adaptation_latency: float
    stability_score: float
    avg_turn_latency_ms: float
    consolidation_events_per_100_turns: float


def compute_metrics(records: list[dict]) -> EvalMetrics:
    if not records:
        return EvalMetrics(0, 0, 0, 1, 1, 0, 0, 0, 0, 0)
    successes = sum(1 for r in records if r["success"])
    precision = sum(r["precision"] for r in records) / len(records)
    recall = sum(r["recall"] for r in records) / len(records)
    hallucinations = sum(1 for r in records if r["hallucinated"])
    constraint_violations = sum(1 for r in records if r.get("constraint_violated", False))
    bloat = records[-1]["memory_items"]
    latency_vals = [r["latency"] for r in records if r["latency"] is not None]
    if latency_vals:
        adaptation_latency = sum(latency_vals) / len(latency_vals)
    else:
        # If the system never recovers after drift, assign max-penalty latency.
        adaptation_latency = float(len(records))
    turn_latency_vals = [float(r.get("turn_latency_ms", 0.0)) for r in records]
    consolidation_events = sum(1 for r in records if r.get("consolidation_event", False))
    mean_success = successes / len(records)
    variance = sum((((1.0 if r["success"] else 0.0) - mean_success) ** 2) for r in records) / len(records)
    stability = 1.0 - variance
    return EvalMetrics(
        task_success_rate=mean_success,
        retrieval_precision_at_k=precision,
        retrieval_recall_at_k=recall,
        hallucination_rate=hallucinations / len(records),
        constraint_violation_rate=constraint_violations / len(records),
        memory_bloat=float(bloat),
        adaptation_latency=adaptation_latency,
        stability_score=stability,
        avg_turn_latency_ms=(sum(turn_latency_vals) / len(turn_latency_vals)) if turn_latency_vals else 0.0,
        consolidation_events_per_100_turns=(100.0 * consolidation_events / len(records)) if records else 0.0,
    )
