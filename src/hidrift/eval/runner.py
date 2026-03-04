from __future__ import annotations

import asyncio
import json
import uuid
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from hidrift.eval.baselines import BaselineConfig, build_baseline
from hidrift.eval.metrics import EvalMetrics, compute_metrics
from hidrift.eval.simulator import SimScenario, build_benchmark_suite
from hidrift.eval.stats import cohen_d, paired_permutation_pvalue, summarize_metric


def _style_reward(expected_style: str, model_reply: str) -> float:
    return 1.0 if expected_style in model_reply.lower() else 0.3


def _measure_retrieval_hits(cfg: BaselineConfig, retrieval, turn) -> tuple[float, float, bool]:
    episodic_hit = any(turn.task_label == e.goal for e in retrieval.episodic)
    vector_hit = any(turn.expected_style in s["statement"].lower() for s in retrieval.semantic)
    graph_hit = any(turn.task_label in f.get("subject", "") or turn.expected_style in f.get("object", "") for f in retrieval.hard_constraints)
    if not cfg.use_vector_semantic:
        vector_hit = False
    if not cfg.use_graph_semantic:
        graph_hit = False
    hit = episodic_hit or vector_hit or graph_hit
    precision = 1.0 if hit else 0.0
    recall = precision
    # Hallucination proxy: semantic constraint contradicts expected style.
    contradiction = any(
        (turn.expected_style not in f.get("statement", "").lower()) and (turn.task_label in f.get("statement", "").lower())
        for f in retrieval.supporting_context
    )
    hallucinated = contradiction
    return precision, recall, hallucinated


async def _run_single_scenario(system_name: str, cfg: BaselineConfig, seed: int, scenario: SimScenario) -> tuple[EvalMetrics, list[dict]]:
    runtime, _ = build_baseline(system_name)
    records: list[dict] = []
    traces: list[dict] = []
    last_drift_turn = None
    recovered_turn = None
    consolidation_count = 0

    for i, turn in enumerate(scenario.turns):
        result = await runtime.handle_turn(
            session_id=f"{scenario.name}-{seed}",
            user_id="u-1",
            user_input=turn.user_input,
            agent_output=None,
            reward=None,
            task_label=turn.task_label,
        )
        if cfg.fixed_consolidation_interval and (i + 1) % cfg.fixed_consolidation_interval == 0:
            await runtime.consolidation.run_once()
            consolidation_count += 1
        if result.get("consolidation"):
            consolidation_count += 1

        retrieval = runtime.memory.retrieve(turn.oracle_fact)
        precision, recall, hallucinated = _measure_retrieval_hits(cfg, retrieval, turn)
        reward = _style_reward(turn.expected_style, result["event"]["agent_output"])
        success = reward >= 0.75 and precision >= 0.5
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
                "memory_items": len(runtime.memory.episodic) + len(runtime.memory.semantic) + len(runtime.memory.semantic.all_facts()),
                "latency": latency,
            }
        )
        traces.append(
            {
                "scenario": scenario.name,
                "system": system_name,
                "seed": seed,
                "turn": i,
                "task_label": turn.task_label,
                "expected_style": turn.expected_style,
                "success": success,
                "precision": precision,
                "hallucinated": hallucinated,
                "drift_score": result["drift_signal"]["total_score"],
                "drift_triggered": result["drift_signal"]["triggered"],
                "memory_items": records[-1]["memory_items"],
                "consolidations": consolidation_count,
            }
        )
    return compute_metrics(records), traces


def _aggregate_metrics(metric_dicts: list[dict]) -> dict:
    keys = metric_dicts[0].keys()
    return {k: sum(m[k] for m in metric_dicts) / len(metric_dicts) for k in keys}


def _significance_against_reference(system_metrics: dict[str, list[dict]], reference: str = "HiDrift-full") -> dict:
    if reference not in system_metrics:
        return {}
    ref = system_metrics[reference]
    output: dict = {}
    metric_names = list(ref[0].keys()) if ref else []
    for system, vals in system_metrics.items():
        if system == reference:
            continue
        output[system] = {}
        for metric in metric_names:
            a = [v[metric] for v in ref]
            b = [v[metric] for v in vals]
            p = paired_permutation_pvalue(a, b)
            d = cohen_d(a, b)
            summary_a = summarize_metric(a)
            summary_b = summarize_metric(b)
            output[system][metric] = {
                "p_value": p,
                "effect_size_cohen_d": d,
                "reference_mean": summary_a.mean,
                "system_mean": summary_b.mean,
                "reference_ci95": [summary_a.ci_low, summary_a.ci_high],
                "system_ci95": [summary_b.ci_low, summary_b.ci_high],
            }
    return output


def run_experiment(
    systems: list[str] | None = None,
    seeds: list[int] | None = None,
    n_turns: int = 120,
    output_dir: str = "artifacts",
) -> dict:
    systems = systems or [
        "RAG-only",
        "HierMemory-noDrift",
        "VectorOnly-noGraph",
        "GraphOnly-noVector",
        "HiDrift-noConflict",
        "HiDrift-noDriftSignal",
        "HiDrift-full",
    ]
    seeds = seeds or [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid.uuid4())
    report: dict = {"run_id": run_id, "systems": {}, "scenario_reports": {}, "traces": []}
    scenario_reports: dict[str, dict] = defaultdict(dict)

    for system in systems:
        _, cfg = build_baseline(system)
        per_seed_overall = []
        per_seed_by_scenario: dict[str, list[dict]] = defaultdict(list)
        for seed in seeds:
            scenarios = build_benchmark_suite(seed=seed, n_turns=n_turns)
            scenario_metric_rows = []
            for scenario in scenarios:
                if not scenario.turns:
                    continue
                m, traces = asyncio.run(_run_single_scenario(system, cfg, seed, scenario))
                row = asdict(m)
                row["scenario"] = scenario.name
                scenario_metric_rows.append(row)
                per_seed_by_scenario[scenario.name].append(row)
                report["traces"].extend(traces)
            if not scenario_metric_rows:
                continue
            # Average across scenarios for this seed/system.
            merged = _aggregate_metrics([{k: v for k, v in r.items() if k != "scenario"} for r in scenario_metric_rows])
            per_seed_overall.append(merged)
        if not per_seed_overall:
            continue
        aggregate = _aggregate_metrics(per_seed_overall)
        report["systems"][system] = {
            "per_seed": per_seed_overall,
            "aggregate_mean": aggregate,
            "n_seeds": len(per_seed_overall),
            "ablation_config": asdict(cfg),
        }
        for scenario_name, rows in per_seed_by_scenario.items():
            scenario_aggregate = _aggregate_metrics([{k: v for k, v in r.items() if k != "scenario"} for r in rows])
            scenario_reports[scenario_name][system] = {
                "per_seed": rows,
                "aggregate_mean": scenario_aggregate,
                "n_seeds": len(rows),
            }
    report["scenario_reports"] = dict(scenario_reports)
    system_seed_metrics = {s: payload["per_seed"] for s, payload in report["systems"].items()}
    report["significance_vs_hidrift_full"] = _significance_against_reference(system_seed_metrics, reference="HiDrift-full")
    path = out / f"eval_{run_id}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

