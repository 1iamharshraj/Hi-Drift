from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import asdict
from pathlib import Path

from hidrift.eval.baselines import build_baseline
from hidrift.eval.metrics import EvalMetrics, compute_metrics
from hidrift.eval.simulator import build_personal_assistant_scenario


async def _run_single(system_name: str, seed: int, n_turns: int) -> EvalMetrics:
    runtime, cfg = build_baseline(system_name)
    scenario = build_personal_assistant_scenario(seed=seed, n_turns=n_turns)
    records: list[dict] = []
    last_drift_turn = None
    recovered_turn = None

    for i, turn in enumerate(scenario.turns):
        result = await runtime.handle_turn(
            session_id=f"s-{seed}",
            user_id="u-1",
            user_input=turn.user_input,
            agent_output=None,
            reward=None,
            task_label=turn.task_label,
        )
        model_reply = result["event"]["agent_output"]
        reward = 1.0 if turn.expected_style in model_reply.lower() else 0.4
        if cfg.fixed_consolidation_interval and (i + 1) % cfg.fixed_consolidation_interval == 0:
            await runtime.consolidation.run_once()

        retrieval = runtime.memory.retrieve(turn.oracle_fact)
        semantic_hit = any(turn.expected_style in s["statement"] for s in retrieval.semantic)
        episodic_hit = any(turn.task_label == e.goal for e in retrieval.episodic)
        precision = 1.0 if (semantic_hit or episodic_hit) else 0.0
        recall = precision
        hallucinated = not (turn.task_label in result["event"]["user_input"])
        success = reward >= 0.75
        if i in scenario.drift_turns:
            last_drift_turn = i
            recovered_turn = None
        if last_drift_turn is not None and recovered_turn is None and success:
            recovered_turn = i
        latency = (recovered_turn - last_drift_turn) if (last_drift_turn is not None and recovered_turn is not None) else None
        records.append(
            {
                "success": success,
                "precision": precision,
                "recall": recall,
                "hallucinated": hallucinated,
                "memory_items": len(runtime.memory.episodic) + len(runtime.memory.semantic),
                "latency": latency,
            }
        )
    return compute_metrics(records)


def run_experiment(
    systems: list[str] | None = None,
    seeds: list[int] | None = None,
    n_turns: int = 120,
    output_dir: str = "artifacts",
) -> dict:
    systems = systems or ["RAG-only", "HierMemory-noDrift", "HiDrift-full"]
    seeds = seeds or [11, 22, 33, 44, 55]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid.uuid4())
    report: dict = {"run_id": run_id, "systems": {}}
    for system in systems:
        metrics = []
        for seed in seeds:
            m = asyncio.run(_run_single(system, seed, n_turns))
            metrics.append(asdict(m))
        keys = metrics[0].keys()
        aggregate = {k: sum(m[k] for m in metrics) / len(metrics) for k in keys}
        report["systems"][system] = {
            "per_seed": metrics,
            "aggregate_mean": aggregate,
            "n_seeds": len(seeds),
        }
    (out / f"eval_{run_id}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
