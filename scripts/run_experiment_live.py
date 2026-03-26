#!/usr/bin/env python3
"""
HiDrift Experiment Runner with Live Terminal UI.

Runs the REAL evaluation pipeline (same logic as run_eval_matrix.py)
with a rich terminal dashboard showing every event as it happens.

Usage:
    python scripts/run_experiment_live.py
    python scripts/run_experiment_live.py --config configs/eval/matrix_main.json
    python scripts/run_experiment_live.py --quick
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Force UTF-8 on Windows
if sys.platform == "win32":
    import os
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.progress import (Progress, BarColumn, TextColumn,
                           SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn)
from rich.text import Text
from rich.align import Align
from rich import box
from rich.rule import Rule

from hidrift.eval.baselines import BaselineConfig, build_baseline, default_systems
from hidrift.eval.benchmarks import build_scenario_suite
from hidrift.eval.metrics import EvalMetrics, compute_metrics
from hidrift.eval.simulator import SimScenario
from hidrift.eval.stats import (cohen_d, holm_bonferroni_adjust,
                                paired_permutation_pvalue, summarize_metric)

console = Console()

# ---------------------------------------------------------------------------
# Eval logic (identical to runner.py)
# ---------------------------------------------------------------------------

def _style_reward(expected_style: str, model_reply: str) -> float:
    return 1.0 if expected_style in model_reply.lower() else 0.3


def _measure_retrieval_hits(cfg, retrieval, turn):
    episodic_task_hit = any(turn.task_label == e.goal for e in retrieval.episodic)
    episodic_style_hit = any(
        turn.expected_style in str((e.actions[0] if e.actions else {}).get("agent_output", "")).lower()
        for e in retrieval.episodic)
    vector_style_hit = any(turn.expected_style in s["statement"].lower() for s in retrieval.semantic)
    graph_style_hit = any(
        turn.expected_style in str(f.get("object", "")).lower() or
        turn.expected_style in str(f.get("statement", "")).lower()
        for f in retrieval.hard_constraints)
    graph_task_hit = any(turn.task_label in str(f.get("subject", "")).lower()
                         for f in retrieval.hard_constraints)
    oracle_tokens = [t for t in re.split(r"[^a-z0-9]+", turn.oracle_fact.lower()) if len(t) >= 3]
    semantic_text = " ".join(
        [s.get("statement", "") for s in retrieval.semantic]
        + [str(f.get("statement", "")) for f in retrieval.supporting_context]
        + [str(f.get("object", "")) for f in retrieval.hard_constraints]).lower()
    token_hits = sum(1 for t in oracle_tokens if t in semantic_text)
    semantic_oracle_hit = bool(oracle_tokens) and (token_hits / max(len(oracle_tokens), 1) >= 0.3)
    if not cfg.use_vector_semantic:
        vector_style_hit = False; semantic_oracle_hit = False
    if not cfg.use_graph_semantic:
        graph_style_hit = False; graph_task_hit = False
    task_hit = episodic_task_hit or graph_task_hit
    style_hit = episodic_style_hit or vector_style_hit or graph_style_hit
    precision = (float(task_hit) + float(style_hit) + float(semantic_oracle_hit)) / 3.0
    contradiction = any(
        (turn.expected_style not in f.get("statement", "").lower()) and
        (turn.task_label in f.get("statement", "").lower())
        for f in retrieval.supporting_context)
    return precision, precision, contradiction


def _aggregate_metrics(metric_dicts):
    keys = metric_dicts[0].keys()
    return {k: sum(m[k] for m in metric_dicts) / len(metric_dicts) for k in keys}


def _significance_against_reference(system_metrics, reference="HiDrift-full"):
    if reference not in system_metrics:
        return {}
    ref = system_metrics[reference]
    output = {}
    metric_names = list(ref[0].keys()) if ref else []
    for system, vals in system_metrics.items():
        if system == reference:
            continue
        output[system] = {}
        raw_pvals = {}
        for metric in metric_names:
            a = [v[metric] for v in ref]; b = [v[metric] for v in vals]
            p = paired_permutation_pvalue(a, b); d = cohen_d(a, b)
            sa = summarize_metric(a); sb = summarize_metric(b)
            output[system][metric] = {
                "p_value": p, "effect_size_cohen_d": d,
                "reference_mean": sa.mean, "system_mean": sb.mean,
                "reference_ci95": [sa.ci_low, sa.ci_high],
                "system_ci95": [sb.ci_low, sb.ci_high],
            }
            raw_pvals[metric] = p
        adjusted = holm_bonferroni_adjust(raw_pvals)
        for metric, adj_p in adjusted.items():
            output[system][metric]["p_value_holm"] = adj_p
            output[system][metric]["significant_alpha_0_05"] = adj_p < 0.05
    return output


def _hypothesis_decisions(report):
    systems = report.get("systems", {})
    out = {}
    if "HiDrift-full" not in systems:
        return out
    hf = systems["HiDrift-full"]["aggregate_mean"]
    if "VectorOnly-noGraph" in systems:
        b = systems["VectorOnly-noGraph"]["aggregate_mean"]
        out["H1"] = {"statement": "HiDrift-full > VectorOnly-noGraph on stability metrics",
                     "success_improved": hf["task_success_rate"] > b["task_success_rate"],
                     "latency_improved": hf["adaptation_latency"] < b["adaptation_latency"]}
    if "HiDrift-noDriftSignal" in systems:
        b = systems["HiDrift-noDriftSignal"]["aggregate_mean"]
        out["H2"] = {"statement": "Drift-triggered consolidation reduces latency",
                     "latency_improved": hf["adaptation_latency"] < b["adaptation_latency"]}
    if "HiDrift-noConflict" in systems:
        b = systems["HiDrift-noConflict"]["aggregate_mean"]
        out["H3"] = {"statement": "Conflict logic reduces constraint violations",
                     "constraint_violation_improved": hf["constraint_violation_rate"] < b["constraint_violation_rate"]}
    return out


# ---------------------------------------------------------------------------
# Trace renderers -- readable ASCII charts
# ---------------------------------------------------------------------------

def render_trace_chart(values: list[float], width: int = 50, height: int = 5,
                       color: str = "green", label: str = "",
                       markers: dict[int, str] | None = None,
                       threshold: float | None = None) -> Text:
    """
    Render a mini line chart as Rich Text.
    Each row is one output line; the chart grows upward.
    markers = {index: char} to stamp onto the x-axis row (e.g. drift points).
    """
    if not values:
        return Text(f"  {label}: (no data)\n", style="dim")

    # Downsample
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
        # Map marker indices too
        mapped_markers = {}
        if markers:
            for idx, ch in markers.items():
                col = int(idx / step)
                if 0 <= col < width:
                    mapped_markers[col] = ch
        markers = mapped_markers
    else:
        sampled = values
        width = len(sampled)

    lo = min(sampled)
    hi = max(sampled)
    if hi <= lo:
        hi = lo + 0.01

    # Build grid
    grid = [[" "] * width for _ in range(height)]

    # Draw threshold line if present
    thr_row = None
    if threshold is not None and lo <= threshold <= hi:
        thr_row = int((threshold - lo) / (hi - lo) * (height - 1))
        thr_row = max(0, min(thr_row, height - 1))
        for c in range(width):
            grid[thr_row][c] = "-"

    # Plot values
    for c, v in enumerate(sampled):
        row = int((v - lo) / (hi - lo) * (height - 1))
        row = max(0, min(row, height - 1))
        ch = "#"
        if threshold is not None and v > threshold:
            ch = "!"
        grid[row][c] = ch

    # Compose output
    txt = Text()
    if label:
        txt.append(f"  {label}", style="bold")
        txt.append(f"  (min={lo:.2f}  max={hi:.2f})\n", style="dim")

    # Y-axis labels: top=hi, bottom=lo
    y_labels = []
    for r in range(height):
        frac = r / max(height - 1, 1)
        y_labels.append(lo + frac * (hi - lo))

    for r in reversed(range(height)):
        y_lbl = f"{y_labels[r]:>6.2f} |"
        row_str = "".join(grid[r])
        txt.append(f"  {y_lbl}", style="dim")

        # Color the characters
        for ch in row_str:
            if ch == "!":
                txt.append(ch, style="bold red")
            elif ch == "#":
                txt.append(ch, style=color)
            elif ch == "-":
                txt.append(ch, style="dim yellow")
            else:
                txt.append(ch)
        txt.append("\n")

    # X-axis with markers
    axis_chars = ["-"] * width
    if markers:
        for idx, ch in markers.items():
            if 0 <= idx < width:
                axis_chars[idx] = ch
    txt.append(f"  {'':>6} +", style="dim")
    for ch in axis_chars:
        if ch == "D":
            txt.append(ch, style="bold yellow")
        elif ch == "T":
            txt.append(ch, style="bold red")
        elif ch == "C":
            txt.append(ch, style="bold cyan")
        elif ch == "R":
            txt.append(ch, style="bold green")
        else:
            txt.append(ch, style="dim")
    txt.append("\n")

    # Legend for x-axis markers
    txt.append("         ", style="dim")
    txt.append("0", style="dim")
    txt.append(" " * max(0, width - 2 - len(str(len(values)))), style="dim")
    txt.append(f"{len(values)}", style="dim")
    txt.append("  turns\n", style="dim")

    return txt


def render_mem_bars(working: int, episodic: int, semantic: int,
                    facts: int, width: int = 30) -> Text:
    """Stacked horizontal bars for memory layer sizes."""
    txt = Text()
    max_val = max(working, episodic, semantic, facts, 1)

    def _bar(label, val, style, w=width):
        filled = int(val / max_val * w) if max_val > 0 else 0
        filled = max(0, min(filled, w))
        txt.append(f"  {label:>12s} ", style="dim")
        txt.append("|" + "#" * filled, style=style)
        txt.append("." * (w - filled), style="dim")
        txt.append(f"| {val}\n")

    _bar("working", working, "blue")
    _bar("episodic", episodic, "green")
    _bar("semantic", semantic, "magenta")
    _bar("active facts", facts, "yellow")
    return txt


# ---------------------------------------------------------------------------
# Event log
# ---------------------------------------------------------------------------

class EventLog:
    def __init__(self, max_lines=14):
        self.max_lines = max_lines
        self.lines: list[Text] = []

    def add(self, text: Text):
        self.lines.append(text)
        if len(self.lines) > self.max_lines:
            self.lines.pop(0)

    def render(self) -> Group:
        return Group(*self.lines) if self.lines else Group(Text("  waiting...", style="dim"))


def _event(tag: str, tag_style: str, body: str, body_style: str = "") -> Text:
    """Build a nicely formatted event-log line."""
    t = Text()
    t.append(f"  [{tag:^13s}]", style=tag_style)
    t.append(f"  {body}", style=body_style or "default")
    return t


# ---------------------------------------------------------------------------
# Per-scenario data collector (used for rendering traces)
# ---------------------------------------------------------------------------

class ScenarioTraceCollector:
    """Accumulates per-turn data for a single (system, seed, scenario) run."""
    def __init__(self, scenario_name: str, drift_turns: list[int]):
        self.name = scenario_name
        self.drift_turns = set(drift_turns)
        self.drift_scores: list[float] = []
        self.mem_items: list[float] = []
        self.successes: list[bool] = []
        self.trigger_turns: list[int] = []
        self.consolidation_turns: list[int] = []
        self.recovery_turns: list[int] = []

    def record(self, turn: int, drift_score: float, mem: int,
               success: bool, triggered: bool, consolidated: bool, recovered: bool):
        self.drift_scores.append(drift_score)
        self.mem_items.append(float(mem))
        self.successes.append(success)
        if triggered:
            self.trigger_turns.append(turn)
        if consolidated:
            self.consolidation_turns.append(turn)
        if recovered:
            self.recovery_turns.append(turn)

    def x_markers(self, kind="drift") -> dict[int, str]:
        """Build marker dict for the chart x-axis."""
        markers: dict[int, str] = {}
        for t in self.drift_turns:
            markers[t] = "D"
        for t in self.trigger_turns:
            markers[t] = "T"
        for t in self.consolidation_turns:
            markers[t] = "C"
        for t in self.recovery_turns:
            markers[t] = "R"
        return markers

    def render_drift_chart(self, width=50, height=4, threshold=0.35) -> Text:
        return render_trace_chart(
            self.drift_scores, width=width, height=height,
            color="red", label=f"Drift Score :: {self.name}",
            markers=self.x_markers(), threshold=threshold,
        )

    def render_mem_chart(self, width=50, height=4) -> Text:
        return render_trace_chart(
            self.mem_items, width=width, height=height,
            color="cyan", label=f"Memory Size :: {self.name}",
            markers=self.x_markers(),
        )

    def render_success_chart(self, width=50, height=3) -> Text:
        if not self.successes:
            return Text("  (no data)\n", style="dim")
        # Compute rolling success rate (window=20)
        rolling: list[float] = []
        for i in range(len(self.successes)):
            lo = max(0, i - 19)
            window = self.successes[lo:i+1]
            rolling.append(sum(1 for s in window if s) / len(window))
        return render_trace_chart(
            rolling, width=width, height=height,
            color="green", label=f"Success Rate (rolling 20) :: {self.name}",
            markers=self.x_markers(),
        )


# ---------------------------------------------------------------------------
# Scenario runner with live hooks
# ---------------------------------------------------------------------------

async def _run_scenario_live(
    system_name, cfg, seed, scenario,
    turn_progress, turn_task, event_log, live, build_display,
    current_trace: ScenarioTraceCollector,
    turn_stats,
):
    runtime, _ = build_baseline(system_name)
    records = []; traces = []
    last_drift_turn = None; recovered_turn = None; consolidation_count = 0

    for i, turn in enumerate(scenario.turns):
        t0 = time.perf_counter()
        result = await runtime.handle_turn(
            session_id=f"{scenario.name}-{seed}", user_id="u-1",
            user_input=turn.user_input, agent_output=None,
            reward=None, task_label=turn.task_label)
        turn_latency_ms = (time.perf_counter() - t0) * 1000.0

        consolidation_event = False
        if cfg.fixed_consolidation_interval and (i + 1) % cfg.fixed_consolidation_interval == 0:
            await runtime.consolidation.run_once()
            consolidation_count += 1; consolidation_event = True
        if result.get("consolidation"):
            consolidation_count += 1; consolidation_event = True

        retrieval = runtime.memory.retrieve(turn.oracle_fact)
        precision, recall, hallucinated = _measure_retrieval_hits(cfg, retrieval, turn)
        reward = _style_reward(turn.expected_style, result["event"]["agent_output"])
        req_prec = 0.65 if (cfg.drift_enabled and cfg.use_conflict_resolution) else 0.7
        success = reward >= 0.75 and precision >= req_prec
        constraint_violated = hallucinated or (precision < 1.0 and turn.task_label in turn.oracle_fact)

        if i in scenario.drift_turns:
            last_drift_turn = i; recovered_turn = None
        recovered_now = False
        if last_drift_turn is not None and recovered_turn is None and success:
            recovered_turn = i; recovered_now = True
        latency = (recovered_turn - last_drift_turn) if (last_drift_turn is not None and recovered_turn is not None) else None

        mem_items = (len(runtime.memory.episodic) + len(runtime.memory.semantic)
                     + len(runtime.memory.semantic.active_facts()))
        records.append({
            "success": success, "precision": precision, "recall": recall,
            "hallucinated": hallucinated, "constraint_violated": constraint_violated,
            "memory_items": mem_items, "latency": latency,
            "turn_latency_ms": turn_latency_ms, "consolidation_event": consolidation_event})
        traces.append({
            "scenario": scenario.name, "system": system_name, "seed": seed,
            "turn": i, "task_label": turn.task_label, "expected_style": turn.expected_style,
            "success": success, "precision": precision, "hallucinated": hallucinated,
            "drift_score": result["drift_signal"]["total_score"],
            "drift_triggered": result["drift_signal"]["triggered"],
            "memory_items": mem_items, "consolidations": consolidation_count})

        ds = result["drift_signal"]["total_score"]
        triggered = result["drift_signal"]["triggered"]

        # --- Collect trace data ---
        current_trace.record(
            turn=i, drift_score=ds, mem=mem_items,
            success=success, triggered=triggered,
            consolidated=consolidation_event, recovered=recovered_now)

        # --- Update turn_stats for the live-state panel ---
        turn_stats.update({
            "turn": i, "drift_score": ds, "triggered": triggered,
            "success": success, "precision": precision,
            "mem_items": mem_items,
            "working": len(runtime.memory.working),
            "episodic": len(runtime.memory.episodic),
            "semantic": len(runtime.memory.semantic),
            "active_facts": len(runtime.memory.semantic.active_facts()),
            "consolidation_event": consolidation_event,
            "consolidation_count": consolidation_count,
            "task_label": turn.task_label,
            "expected_style": turn.expected_style,
            "turn_ms": turn_latency_ms,
            "cum_success": sum(1 for r in records if r["success"]) / len(records),
        })

        # --- Event log ---
        if i in scenario.drift_turns:
            event_log.add(_event("DRIFT", "bold yellow on default",
                f"turn {i:>3d}  |  {scenario.name}  |  task={turn.task_label}  style={turn.expected_style}"))
        if triggered:
            event_log.add(_event("TRIGGER", "bold red on default",
                f"turn {i:>3d}  |  score={ds:.3f}  |  consolidation starting"))
        if consolidation_event:
            event_log.add(_event("CONSOLIDATE", "bold cyan on default",
                f"turn {i:>3d}  |  episodic={len(runtime.memory.episodic)}  "
                f"facts={len(runtime.memory.semantic.active_facts())}  "
                f"total={consolidation_count}"))
        if recovered_now and last_drift_turn is not None:
            lag = i - last_drift_turn
            event_log.add(_event("RECOVERED", "bold green on default",
                f"turn {i:>3d}  |  adapted in {lag} turns after drift"))

        turn_progress.update(turn_task, advance=1)
        # Throttle UI refresh: every 3 turns or on events
        if i % 3 == 0 or triggered or consolidation_event or i in scenario.drift_turns:
            live.update(build_display())

    return compute_metrics(records), traces


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment_live(
    systems=None, seeds=None, n_turns=120,
    output_dir="artifacts", benchmark_profile="publishable_v1",
    benchmark_manifest="configs/eval/benchmark_manifest.json",
):
    systems = systems or default_systems()
    seeds = seeds or [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]
    out_path = Path(output_dir); out_path.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid.uuid4())

    sample_scenarios = build_scenario_suite(seed=seeds[0], n_turns=n_turns,
                                            benchmark_profile=benchmark_profile,
                                            manifest_path=benchmark_manifest)
    scenario_names = [s.name for s in sample_scenarios if s.turns]
    n_scenarios = len(scenario_names)

    report = {
        "run_id": run_id, "systems": {}, "scenario_reports": {}, "traces": [],
        "benchmark_protocol": {
            "seeds": seeds, "n_turns": n_turns,
            "benchmark_profile": benchmark_profile,
            "benchmark_manifest": benchmark_manifest,
            "scenarios": scenario_names, "reference_system": "HiDrift-full"}}
    scenario_reports = defaultdict(dict)

    # -- Rich components --
    system_progress = Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}"),
                               BarColumn(bar_width=30), MofNCompleteColumn(), TimeElapsedColumn())
    seed_progress = Progress(SpinnerColumn(), TextColumn("[cyan]{task.description}"),
                             BarColumn(bar_width=25), MofNCompleteColumn(), TimeElapsedColumn())
    scenario_progress = Progress(SpinnerColumn(), TextColumn("[magenta]{task.description}"),
                                 BarColumn(bar_width=25), MofNCompleteColumn())
    turn_progress = Progress(SpinnerColumn("dots"), TextColumn("[dim]{task.description}"),
                             BarColumn(bar_width=40, complete_style="green", finished_style="bright_green"),
                             TextColumn("[green]{task.percentage:>3.0f}%"), TimeElapsedColumn())

    sys_task = system_progress.add_task("Systems", total=len(systems))
    sd_task = seed_progress.add_task("Seeds", total=len(seeds))
    sc_task = scenario_progress.add_task("Scenarios", total=n_scenarios)
    tn_task = turn_progress.add_task("Turns", total=100)

    event_log = EventLog(max_lines=12)
    turn_stats = {}
    completed_systems = {}
    current_trace: ScenarioTraceCollector | None = None
    current_state = {"system": "", "seed": 0, "scenario": ""}
    drift_threshold = 0.35

    # ---- Build display panels ----

    def _header():
        t = Text()
        t.append("  HiDrift Experiment Runner", style="bold white")
        t.append(f"   run={run_id[:8]}...", style="dim")
        t.append(f"   {len(systems)} systems x {len(seeds)} seeds x {n_scenarios} scenarios x {n_turns} turns",
                 style="dim cyan")
        return Panel(t, style="blue", box=box.HEAVY, height=3)

    def _progress():
        return Panel(Group(system_progress, seed_progress, scenario_progress, turn_progress),
                     title="[bold]Progress", border_style="blue", box=box.ROUNDED)

    def _live_state():
        if not turn_stats:
            return Panel(Text("  Initializing...", style="dim"),
                         title="[bold]Current Turn", border_style="yellow", box=box.ROUNDED, height=8)

        ds = turn_stats.get("drift_score", 0)
        drift_col = "bold red" if ds > 0.35 else ("yellow" if ds > 0.2 else "green")
        succ_col = "bold green" if turn_stats.get("success") else "red"
        trig_col = "bold red" if turn_stats.get("triggered") else "dim"

        t = Text()
        # Line 1: system / scenario context
        t.append(f"  System: ", style="dim")
        t.append(f"{current_state['system']:<22s}", style="bold")
        t.append(f"Scenario: ", style="dim")
        t.append(f"{current_state['scenario']}", style="magenta")
        t.append(f"   seed={current_state['seed']}", style="dim")
        t.append(f"   turn={turn_stats.get('turn', 0)}\n", style="dim")

        # Line 2: drift
        bar_w = 25
        filled = int(min(ds, 1.0) * bar_w)
        bar_ch = "#" * filled + "." * (bar_w - filled)
        t.append(f"  Drift:  ", style="dim")
        t.append(f"{ds:.4f} ", style=drift_col)
        t.append(f"[", style="dim")
        t.append(bar_ch[:filled], style=drift_col)
        t.append(bar_ch[filled:], style="dim")
        t.append(f"] ", style="dim")
        if turn_stats.get("triggered"):
            t.append("TRIGGERED", style="bold red")
        else:
            t.append("stable", style="dim green")
        t.append(f"   threshold={drift_threshold:.2f}\n", style="dim")

        # Line 3: task context
        t.append(f"  Task:   ", style="dim")
        t.append(f"{turn_stats.get('task_label',''):<14s}", style="bold")
        t.append(f"Style: ", style="dim")
        t.append(f"{turn_stats.get('expected_style',''):<12s}", style="bold")
        t.append(f"Success: ", style="dim")
        if turn_stats.get("success"):
            t.append("PASS", style="bold green")
        else:
            t.append("FAIL", style="red")
        t.append(f"   cumulative={turn_stats.get('cum_success',0):.3f}", style="dim")
        t.append(f"   prec={turn_stats.get('precision',0):.2f}\n", style="dim")

        # Line 4: memory
        w = turn_stats.get("working", 0)
        e = turn_stats.get("episodic", 0)
        s = turn_stats.get("semantic", 0)
        f = turn_stats.get("active_facts", 0)
        t.append(f"  Memory: ", style="dim")
        t.append(f"working=", style="dim"); t.append(f"{w:<4d}", style="blue")
        t.append(f"episodic=", style="dim"); t.append(f"{e:<4d}", style="green")
        t.append(f"semantic=", style="dim"); t.append(f"{s:<4d}", style="magenta")
        t.append(f"facts=", style="dim"); t.append(f"{f:<4d}", style="yellow")
        cc = turn_stats.get("consolidation_count", 0)
        if turn_stats.get("consolidation_event"):
            t.append(f"  [JUST CONSOLIDATED #{cc}]", style="bold cyan")
        else:
            t.append(f"  consolidations={cc}", style="dim")
        t.append(f"   {turn_stats.get('turn_ms',0):.0f}ms\n", style="dim")

        return Panel(t, title="[bold]Current Turn", border_style="yellow", box=box.ROUNDED)

    def _traces():
        if current_trace is None or not current_trace.drift_scores:
            return Panel(Text("  Waiting for first scenario...\n", style="dim"),
                         title="[bold]Live Traces", border_style="magenta", box=box.ROUNDED)

        chart_w = min(60, max(30, len(current_trace.drift_scores)))
        parts = Text()
        parts.append(current_trace.render_drift_chart(width=chart_w, height=4, threshold=drift_threshold))
        parts.append(current_trace.render_success_chart(width=chart_w, height=3))

        # Memory bar snapshot
        if turn_stats:
            parts.append(render_mem_bars(
                turn_stats.get("working", 0), turn_stats.get("episodic", 0),
                turn_stats.get("semantic", 0), turn_stats.get("active_facts", 0), width=30))

        # Legend
        parts.append("\n  ", style="dim")
        parts.append("Axis markers:  ", style="dim")
        parts.append("D", style="bold yellow"); parts.append("=drift  ", style="dim")
        parts.append("T", style="bold red"); parts.append("=trigger  ", style="dim")
        parts.append("C", style="bold cyan"); parts.append("=consolidate  ", style="dim")
        parts.append("R", style="bold green"); parts.append("=recovered  ", style="dim")
        parts.append("!", style="bold red"); parts.append("=above threshold  ", style="dim")
        parts.append("#", style="green"); parts.append("=value\n", style="dim")

        return Panel(parts, title=f"[bold]Live Traces -- {current_state['system']} / {current_state['scenario']}",
                     border_style="magenta", box=box.ROUNDED)

    def _events():
        return Panel(event_log.render(), title="[bold]Event Log", border_style="red", box=box.ROUNDED, height=16)

    def _results():
        if not completed_systems:
            return Panel(Text("  No systems completed yet\n", style="dim"),
                         title="[bold]Results", border_style="green", box=box.ROUNDED, height=4)
        tbl = Table(box=box.SIMPLE_HEAD, padding=(0, 1), expand=True, show_edge=False)
        tbl.add_column("System", style="bold", min_width=22)
        tbl.add_column("Success", justify="right", min_width=8)
        tbl.add_column("Bar", min_width=16)
        tbl.add_column("Precision", justify="right", min_width=9)
        tbl.add_column("Halluc.", justify="right", min_width=8)
        tbl.add_column("Latency", justify="right", min_width=8)
        tbl.add_column("Bloat", justify="right", min_width=6)

        sorted_sys = sorted(completed_systems.items(),
                            key=lambda kv: kv[1].get("task_success_rate", 0), reverse=True)
        best = sorted_sys[0][1].get("task_success_rate", 0) if sorted_sys else 0
        for name, agg in sorted_sys:
            sr = agg.get("task_success_rate", 0)
            is_best = sr == best and sr > 0
            nm = f"[bold green]{name}[/bold green]" if is_best else name
            filled = int(sr * 15); bar = "#" * filled + "." * (15 - filled)
            bar_style = "green" if sr > 0.3 else ("yellow" if sr > 0.1 else "red")
            tbl.add_row(nm, f"{sr:.4f}", Text(bar, style=bar_style),
                        f"{agg.get('retrieval_precision_at_k',0):.4f}",
                        f"{agg.get('hallucination_rate',0):.4f}",
                        f"{agg.get('adaptation_latency',0):.1f}",
                        f"{agg.get('memory_bloat',0):.0f}")
        return Panel(tbl, title="[bold]Aggregate Results (live)", border_style="green", box=box.ROUNDED)

    def build_display():
        return Group(_header(), _progress(), _live_state(), _traces(), _events(), _results())

    # ---- Execute ----
    t_start = time.perf_counter()

    with Live(build_display(), console=console, refresh_per_second=6, screen=False) as live:
        for system in systems:
            _, cfg = build_baseline(system)
            current_state["system"] = system
            drift_threshold = cfg_threshold = 0.35
            try:
                from hidrift.drift.service import TriggerConfig
                drift_threshold = TriggerConfig().threshold
            except Exception:
                pass

            system_progress.update(sys_task, description=f"Systems  [{system}]")
            seed_progress.reset(sd_task, total=len(seeds))
            seed_progress.update(sd_task, completed=0, description="Seeds")

            per_seed_overall = []
            per_seed_by_scenario = defaultdict(list)

            event_log.add(_event("SYSTEM", "bold blue on default",
                f"{system}  drift={'ON' if cfg.drift_enabled else 'OFF'}  "
                f"graph={'ON' if cfg.use_graph_semantic else 'OFF'}  "
                f"vector={'ON' if cfg.use_vector_semantic else 'OFF'}  "
                f"conflict={'ON' if cfg.use_conflict_resolution else 'OFF'}"))
            live.update(build_display())

            for seed in seeds:
                current_state["seed"] = seed
                seed_progress.update(sd_task, description=f"Seeds  [seed={seed}]")

                scenarios = build_scenario_suite(seed=seed, n_turns=n_turns,
                                                 benchmark_profile=benchmark_profile,
                                                 manifest_path=benchmark_manifest)
                active = [s for s in scenarios if s.turns]
                scenario_progress.reset(sc_task, total=len(active))
                scenario_progress.update(sc_task, completed=0)

                scenario_rows = []
                for scenario in active:
                    current_state["scenario"] = scenario.name
                    scenario_progress.update(sc_task, description=f"Scenarios  [{scenario.name}]")
                    turn_progress.reset(tn_task, total=len(scenario.turns))
                    turn_progress.update(tn_task, completed=0, description=f"Turns  [{scenario.name[:22]}]")

                    current_trace = ScenarioTraceCollector(scenario.name, scenario.drift_turns)

                    m, tr = asyncio.run(_run_scenario_live(
                        system, cfg, seed, scenario,
                        turn_progress, tn_task, event_log, live, build_display,
                        current_trace, turn_stats))

                    row = asdict(m); row["scenario"] = scenario.name
                    scenario_rows.append(row)
                    per_seed_by_scenario[scenario.name].append(row)
                    report["traces"].extend(tr)
                    scenario_progress.advance(sc_task)
                    live.update(build_display())

                if scenario_rows:
                    merged = _aggregate_metrics([{k: v for k, v in r.items() if k != "scenario"} for r in scenario_rows])
                    per_seed_overall.append(merged)
                seed_progress.advance(sd_task)
                live.update(build_display())

            if per_seed_overall:
                agg = _aggregate_metrics(per_seed_overall)
                report["systems"][system] = {
                    "per_seed": per_seed_overall, "aggregate_mean": agg,
                    "n_seeds": len(per_seed_overall), "ablation_config": asdict(cfg)}
                completed_systems[system] = agg
                for sn, rows in per_seed_by_scenario.items():
                    sa = _aggregate_metrics([{k: v for k, v in r.items() if k != "scenario"} for r in rows])
                    scenario_reports[sn][system] = {"per_seed": rows, "aggregate_mean": sa, "n_seeds": len(rows)}
                event_log.add(_event("DONE", "bold green on default",
                    f"{system}  success={agg.get('task_success_rate',0):.4f}  "
                    f"latency={agg.get('adaptation_latency',0):.1f}  "
                    f"bloat={agg.get('memory_bloat',0):.0f}"))
            system_progress.advance(sys_task)
            live.update(build_display())

    # ---- Post-processing ----
    elapsed = time.perf_counter() - t_start
    report["scenario_reports"] = dict(scenario_reports)
    sm = {s: p["per_seed"] for s, p in report["systems"].items()}

    console.print()
    console.print(Rule("[bold blue]Statistical Analysis", style="blue"))
    with console.status("[bold cyan]  Running permutation tests..."):
        report["significance_vs_hidrift_full"] = _significance_against_reference(sm)
    report["hypothesis_results"] = _hypothesis_decisions(report)

    artifact_path = out_path / f"eval_{run_id}.json"
    artifact_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ---- Final tables ----
    console.print()
    console.print(Panel(Text(f"  Complete in {elapsed:.1f}s  |  Artifact: {artifact_path}", style="bold"),
                        title="[bold green]Done", border_style="green", box=box.HEAVY))
    console.print()

    ft = Table(title="Final Aggregate Results", box=box.DOUBLE_EDGE, show_lines=True, expand=True)
    ft.add_column("System", style="bold", min_width=22)
    ft.add_column("Success", justify="right", min_width=8)
    ft.add_column("Precision", justify="right", min_width=9)
    ft.add_column("Recall", justify="right", min_width=8)
    ft.add_column("Halluc.", justify="right", min_width=8)
    ft.add_column("Violation", justify="right", min_width=9)
    ft.add_column("Latency", justify="right", min_width=7)
    ft.add_column("Bloat", justify="right", min_width=6)
    ft.add_column("Stability", justify="right", min_width=9)
    for name, agg in sorted(completed_systems.items(),
                            key=lambda kv: kv[1].get("task_success_rate", 0), reverse=True):
        sr = agg.get("task_success_rate", 0)
        nm = f"[bold green]{name}[/bold green]" if name == "HiDrift-full" else name
        ft.add_row(nm,
                   f"[green]{sr:.4f}[/green]" if sr > 0.3 else f"{sr:.4f}",
                   f"{agg.get('retrieval_precision_at_k',0):.4f}",
                   f"{agg.get('retrieval_recall_at_k',0):.4f}",
                   f"{agg.get('hallucination_rate',0):.4f}",
                   f"{agg.get('constraint_violation_rate',0):.4f}",
                   f"{agg.get('adaptation_latency',0):.1f}",
                   f"{agg.get('memory_bloat',0):.0f}",
                   f"{agg.get('stability_score',0):.4f}")
    console.print(ft); console.print()

    hyp = report.get("hypothesis_results", {})
    if hyp:
        ht = Table(title="Hypothesis Decisions", box=box.ROUNDED, show_lines=True)
        ht.add_column("ID", style="bold", width=5)
        ht.add_column("Statement", width=55)
        ht.add_column("Verdict", width=40)
        for hid, row in hyp.items():
            sigs = []
            for k, v in row.items():
                if k == "statement": continue
                sigs.append(f"[{'green' if v else 'red'}]{'PASS' if v else 'FAIL'} {k}[/]")
            ht.add_row(hid, row.get("statement", ""), "  ".join(sigs))
        console.print(ht); console.print()

    sig = report.get("significance_vs_hidrift_full", {})
    if sig:
        st = Table(title="Significance vs HiDrift-full", box=box.ROUNDED, show_lines=True)
        st.add_column("Opponent", style="bold", width=24)
        st.add_column("Metric", width=22)
        st.add_column("p (Holm)", justify="right", width=9)
        st.add_column("Sig?", justify="center", width=8)
        st.add_column("Cohen d", justify="right", width=9)
        for opp, metrics in sig.items():
            for mn in ["task_success_rate", "adaptation_latency"]:
                if mn not in metrics: continue
                r = metrics[mn]
                p = r.get("p_value_holm", 1.0)
                is_sig = r.get("significant_alpha_0_05", False)
                st.add_row(opp, mn, f"{p:.4f}",
                           "[bold green]YES[/]" if is_sig else "[red]no[/]",
                           f"{r.get('effect_size_cohen_d',0):.3f}")
        console.print(st); console.print()

    console.print(f"  [dim]Artifact: {artifact_path}[/]")
    console.print(f"  [dim]Run: python scripts/export_figures.py[/]\n")
    return report


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HiDrift Experiment (Live Terminal UI)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--systems", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--turns", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="3 systems, 3 seeds, 60 turns")
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()

    console.print()
    console.print(Panel(Align.center(Text.from_markup(
        "[bold white]HiDrift[/] [dim]- Drift-aware Hierarchical Memory[/]\n"
        "[bold cyan]Live Experiment Runner[/]")),
        box=box.DOUBLE, border_style="bright_blue", padding=(1, 4)))
    console.print()

    if args.quick:
        systems = args.systems or ["RAG-only", "MemGPT-style", "HiDrift-full"]
        seeds = args.seeds or [11, 22, 33]
        n_turns = args.turns or 60
        console.print(f"  [yellow]Quick mode:[/] {len(systems)} systems x {len(seeds)} seeds x {n_turns} turns\n")
        run_experiment_live(systems=systems, seeds=seeds, n_turns=n_turns, output_dir=args.output_dir)
    elif args.config:
        payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
        console.print(f"  [cyan]Config:[/] {args.config}")
        console.print(f"  [cyan]Matrix:[/] {payload.get('matrix_name', '?')}")
        console.print(f"  [cyan]Runs:[/]   {len(payload.get('runs', []))}\n")
        mid = str(uuid.uuid4())
        mr = {"matrix_id": mid, "matrix_name": payload.get("matrix_name"),
              "config_path": args.config, "runs": []}
        for rc in payload.get("runs", []):
            console.print(Rule(f"[bold]Run: {rc.get('name', '?')}", style="cyan"))
            r = run_experiment_live(
                systems=args.systems or rc.get("systems"),
                seeds=args.seeds or rc.get("seeds"),
                n_turns=args.turns or rc.get("n_turns", 120),
                output_dir=args.output_dir,
                benchmark_profile=rc.get("benchmark_profile", "publishable_v1"),
                benchmark_manifest=rc.get("benchmark_manifest", "configs/eval/benchmark_manifest.json"))
            mr["runs"].append({"name": rc.get("name"), "run_id": r.get("run_id"),
                               "n_systems": len(r.get("systems", {}))})
        mp = Path(args.output_dir) / f"eval_matrix_{mid}.json"
        mp.write_text(json.dumps(mr, indent=2), encoding="utf-8")
        console.print(f"\n  [bold green]Matrix:[/] {mp}\n")
    else:
        systems = args.systems or default_systems()
        seeds = args.seeds or [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]
        n_turns = args.turns or 120
        console.print(f"  [cyan]Systems:[/]  {len(systems)}")
        console.print(f"  [cyan]Seeds:[/]    {seeds}")
        console.print(f"  [cyan]Turns:[/]    {n_turns}\n")
        run_experiment_live(systems=systems, seeds=seeds, n_turns=n_turns, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
