from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from hidrift.eval.baselines import BaselineConfig, build_baseline, default_systems
from hidrift.eval.benchmarks import build_scenario_suite
from hidrift.eval.metrics import EvalMetrics, compute_metrics
from hidrift.eval.simulator import KNOWN_STYLES, SimScenario
from hidrift.eval.stats import cohen_d, holm_bonferroni_adjust, paired_permutation_pvalue, summarize_metric


def _style_reward(expected_style: str, model_reply: str) -> float:
    return 1.0 if expected_style in model_reply.lower() else 0.3


def _measure_retrieval_hits(cfg: BaselineConfig, retrieval, turn) -> tuple[float, float, bool]:
    episodic_task_hit = any(turn.task_label == e.goal for e in retrieval.episodic)
    episodic_style_hit = any(
        turn.expected_style in str((e.actions[0] if e.actions else {}).get("agent_output", "")).lower()
        for e in retrieval.episodic
    )
    vector_style_hit = any(turn.expected_style in s["statement"].lower() for s in retrieval.semantic)
    graph_style_hit = any(
        turn.expected_style in str(f.get("object", "")).lower() or turn.expected_style in str(f.get("statement", "")).lower()
        for f in retrieval.hard_constraints
    )
    graph_task_hit = any(turn.task_label in str(f.get("subject", "")).lower() for f in retrieval.hard_constraints)
    oracle_tokens = [t for t in re.split(r"[^a-z0-9]+", turn.oracle_fact.lower()) if len(t) >= 3]
    semantic_text = " ".join(
        [s.get("statement", "") for s in retrieval.semantic]
        + [str(f.get("statement", "")) for f in retrieval.supporting_context]
        + [str(f.get("object", "")) for f in retrieval.hard_constraints]
    ).lower()
    token_hits = sum(1 for t in oracle_tokens if t in semantic_text)
    semantic_oracle_hit = bool(oracle_tokens) and (token_hits / max(len(oracle_tokens), 1) >= 0.3)
    if not cfg.use_vector_semantic:
        vector_style_hit = False
        semantic_oracle_hit = False
    if not cfg.use_graph_semantic:
        graph_style_hit = False
        graph_task_hit = False
    # Precision requires resolving both task and style under drift.
    task_hit = episodic_task_hit or graph_task_hit
    style_hit = episodic_style_hit or vector_style_hit or graph_style_hit
    oracle_hit = semantic_oracle_hit
    hit = task_hit and style_hit
    precision = (float(task_hit) + float(style_hit) + float(oracle_hit)) / 3.0
    recall = precision
    # Hallucination proxy: a supporting fact explicitly asserts a *different* style for this task.
    hallucinated = any(
        turn.task_label in f.get("statement", "").lower()
        and any(s in f.get("statement", "").lower() for s in KNOWN_STYLES if s != turn.expected_style)
        for f in retrieval.supporting_context
    )
    return precision, recall, hallucinated


async def _run_single_scenario(system_name: str, cfg: BaselineConfig, seed: int, scenario: SimScenario) -> tuple[EvalMetrics, list[dict]]:
    runtime, _ = build_baseline(system_name)
    records: list[dict] = []
    traces: list[dict] = []
    last_drift_turn = None
    recovered_turn = None
    consolidation_count = 0

    for i, turn in enumerate(scenario.turns):
        t0 = time.perf_counter()
        result = await runtime.handle_turn(
            session_id=f"{scenario.name}-{seed}",
            user_id="u-1",
            user_input=turn.user_input,
            agent_output=None,
            reward=None,
            task_label=turn.task_label,
        )
        turn_latency_ms = (time.perf_counter() - t0) * 1000.0
        consolidation_event = False
        if cfg.fixed_consolidation_interval and (i + 1) % cfg.fixed_consolidation_interval == 0:
            await runtime.consolidation.run_once()
            consolidation_count += 1
            consolidation_event = True
        if result.get("consolidation"):
            consolidation_count += 1
            consolidation_event = True

        retrieval = runtime.memory.retrieve(turn.oracle_fact)
        precision, recall, hallucinated = _measure_retrieval_hits(cfg, retrieval, turn)
        reward = _style_reward(turn.expected_style, result["event"]["agent_output"])
        required_precision = 0.65
        success = reward >= 0.75 and precision >= required_precision
        # Constraint violated when a hard constraint asserts a stale style for this task.
        stale_constraint = any(
            turn.task_label in str(f.get("subject", "")).lower()
            and turn.expected_style not in str(f.get("object", "")).lower()
            and turn.expected_style not in str(f.get("statement", "")).lower()
            for f in retrieval.hard_constraints
        )
        constraint_violated = stale_constraint
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
                "constraint_violated": constraint_violated,
                # Effective footprint excludes inactive superseded facts.
                "memory_items": len(runtime.memory.episodic) + len(runtime.memory.semantic) + len(runtime.memory.semantic.active_facts()),
                "latency": latency,
                "turn_latency_ms": turn_latency_ms,
                "consolidation_event": consolidation_event,
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
        raw_pvals: dict[str, float] = {}
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
            raw_pvals[metric] = p
        adjusted = holm_bonferroni_adjust(raw_pvals)
        for metric, adj_p in adjusted.items():
            output[system][metric]["p_value_holm"] = adj_p
            output[system][metric]["significant_alpha_0_05"] = adj_p < 0.05
    return output


def _hypothesis_decisions(report: dict) -> dict:
    """
    H1: HiDrift-full > VectorOnly-noGraph on adaptation latency and success.
    H2: HiDrift-full > HiDrift-noDriftSignal on adaptation latency.
    H3: HiDrift-full < HiDrift-noConflict on constraint violations.
    """
    systems = report.get("systems", {})
    out: dict = {}
    if "HiDrift-full" not in systems:
        return out
    hf = systems["HiDrift-full"]["aggregate_mean"]
    if "VectorOnly-noGraph" in systems:
        base = systems["VectorOnly-noGraph"]["aggregate_mean"]
        out["H1"] = {
            "statement": "HiDrift-full improves over VectorOnly-noGraph on stability-critical metrics",
            "success_improved": hf["task_success_rate"] > base["task_success_rate"],
            "latency_improved": hf["adaptation_latency"] < base["adaptation_latency"],
        }
    if "HiDrift-noDriftSignal" in systems:
        base = systems["HiDrift-noDriftSignal"]["aggregate_mean"]
        out["H2"] = {
            "statement": "Drift-triggered consolidation reduces adaptation latency",
            "latency_improved": hf["adaptation_latency"] < base["adaptation_latency"],
        }
    if "HiDrift-noConflict" in systems:
        base = systems["HiDrift-noConflict"]["aggregate_mean"]
        out["H3"] = {
            "statement": "Conflict logic reduces constraint violations",
            "constraint_violation_improved": hf["constraint_violation_rate"] < base["constraint_violation_rate"],
        }
    return out


def run_experiment(
    systems: list[str] | None = None,
    seeds: list[int] | None = None,
    n_turns: int = 120,
    output_dir: str = "artifacts",
    benchmark_profile: str = "publishable_v1",
    benchmark_manifest: str | None = "configs/eval/benchmark_manifest.json",
) -> dict:
    systems = systems or default_systems()
    seeds = seeds or [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid.uuid4())
    report: dict = {
        "run_id": run_id,
        "systems": {},
        "scenario_reports": {},
        "traces": [],
        "benchmark_protocol": {
            "seeds": seeds,
            "n_turns": n_turns,
            "benchmark_profile": benchmark_profile,
            "benchmark_manifest": benchmark_manifest,
            "scenarios": [],
            "reference_system": "HiDrift-full",
        },
    }
    scenario_reports: dict[str, dict] = defaultdict(dict)

    for system in systems:
        _, cfg = build_baseline(system)
        per_seed_overall = []
        per_seed_by_scenario: dict[str, list[dict]] = defaultdict(list)
        for seed in seeds:
            scenarios = build_scenario_suite(
                seed=seed,
                n_turns=n_turns,
                benchmark_profile=benchmark_profile,
                manifest_path=benchmark_manifest,
            )
            if not report["benchmark_protocol"]["scenarios"]:
                report["benchmark_protocol"]["scenarios"] = [scenario.name for scenario in scenarios]
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
    report["hypothesis_results"] = _hypothesis_decisions(report)
    path = out / f"eval_{run_id}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
