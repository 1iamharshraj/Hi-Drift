"""
Live visual demo of HiDrift experiment.

Shows real-time plots updating as turns are processed:
  - Drift score over turns (with trigger markers)
  - Memory hierarchy sizes (working / episodic / semantic facts)
  - Cumulative success rate
  - Constraint violation rolling rate

Run:
    python scripts/run_live_demo.py
    python scripts/run_live_demo.py --system HiDrift-full --scenario personal_assistant_drift --turns 120
    python scripts/run_live_demo.py --side-by-side   # compare HiDrift-full vs RAG-only
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from hidrift.eval.baselines import BaselineConfig, build_baseline
from hidrift.eval.simulator import (
    build_personal_assistant_scenario,
    build_tool_api_drift_scenario,
    build_contradiction_drift_scenario,
)

SCENARIOS = {
    "personal_assistant_drift": build_personal_assistant_scenario,
    "tool_api_drift": build_tool_api_drift_scenario,
    "contradiction_drift": build_contradiction_drift_scenario,
}

COLORS = {
    "drift_score": "#e74c3c",
    "threshold": "#95a5a6",
    "trigger": "#f39c12",
    "working": "#3498db",
    "episodic": "#2ecc71",
    "semantic_items": "#9b59b6",
    "active_facts": "#e67e22",
    "success": "#27ae60",
    "violation": "#e74c3c",
    "consolidation": "#f1c40f",
}


def _style_reward(expected_style: str, model_reply: str) -> float:
    return 1.0 if expected_style in model_reply.lower() else 0.3


class LiveDashboard:
    """Real-time 4-panel matplotlib dashboard."""

    def __init__(self, title: str, n_turns: int, drift_turns: list[int], threshold: float):
        self.n_turns = n_turns
        self.drift_turns = drift_turns
        self.threshold = threshold

        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 8))
        self.fig.suptitle(title, fontsize=13, fontweight="bold")
        self.fig.patch.set_facecolor("#fafafa")

        # Storage
        self.turns: list[int] = []
        self.drift_scores: list[float] = []
        self.drift_triggered: list[bool] = []
        self.working_sizes: list[int] = []
        self.episodic_sizes: list[int] = []
        self.semantic_item_sizes: list[int] = []
        self.active_fact_sizes: list[int] = []
        self.cum_successes: list[float] = []
        self.violation_rolling: list[float] = []
        self._violations_raw: list[float] = []
        self._successes_raw: list[float] = []
        self.consolidation_turns: list[int] = []

        self._setup_axes()
        self.fig.tight_layout(rect=[0, 0.02, 1, 0.95])
        plt.show(block=False)
        plt.pause(0.01)

    def _setup_axes(self):
        ax1, ax2, ax3, ax4 = self.axes.flat

        # Panel 1: Drift score
        ax1.set_title("Drift Score (live)", fontsize=10)
        ax1.set_xlabel("turn")
        ax1.set_ylabel("score")
        ax1.set_xlim(0, self.n_turns)
        ax1.set_ylim(0, 1.5)
        ax1.axhline(self.threshold, color=COLORS["threshold"], ls="--", lw=1, label=f"threshold={self.threshold}")
        for dt in self.drift_turns:
            ax1.axvline(dt, color="#bdc3c7", ls=":", lw=0.8, alpha=0.7)
        self._line_drift, = ax1.plot([], [], color=COLORS["drift_score"], lw=1.2, label="drift score")
        ax1.legend(fontsize=7, loc="upper right")

        # Panel 2: Memory sizes
        ax2.set_title("Memory Hierarchy (live)", fontsize=10)
        ax2.set_xlabel("turn")
        ax2.set_ylabel("count")
        ax2.set_xlim(0, self.n_turns)
        self._line_working, = ax2.plot([], [], color=COLORS["working"], lw=1.2, label="working")
        self._line_episodic, = ax2.plot([], [], color=COLORS["episodic"], lw=1.2, label="episodic")
        self._line_sem_items, = ax2.plot([], [], color=COLORS["semantic_items"], lw=1.2, label="semantic items")
        self._line_facts, = ax2.plot([], [], color=COLORS["active_facts"], lw=1.2, label="active facts")
        ax2.legend(fontsize=7, loc="upper left")

        # Panel 3: Cumulative success rate
        ax3.set_title("Cumulative Success Rate (live)", fontsize=10)
        ax3.set_xlabel("turn")
        ax3.set_ylabel("rate")
        ax3.set_xlim(0, self.n_turns)
        ax3.set_ylim(0, 1.05)
        self._line_success, = ax3.plot([], [], color=COLORS["success"], lw=1.5)
        for dt in self.drift_turns:
            ax3.axvline(dt, color="#bdc3c7", ls=":", lw=0.8, alpha=0.7)

        # Panel 4: Constraint violation rolling rate
        ax4.set_title("Constraint Violation (rolling 20-turn)", fontsize=10)
        ax4.set_xlabel("turn")
        ax4.set_ylabel("rate")
        ax4.set_xlim(0, self.n_turns)
        ax4.set_ylim(0, 1.05)
        self._line_violation, = ax4.plot([], [], color=COLORS["violation"], lw=1.2)
        for dt in self.drift_turns:
            ax4.axvline(dt, color="#bdc3c7", ls=":", lw=0.8, alpha=0.7)

    def update(
        self,
        turn: int,
        drift_score: float,
        triggered: bool,
        working_size: int,
        episodic_size: int,
        semantic_item_size: int,
        active_fact_count: int,
        success: bool,
        constraint_violated: bool,
        consolidation_event: bool,
    ):
        self.turns.append(turn)
        self.drift_scores.append(drift_score)
        self.drift_triggered.append(triggered)
        self.working_sizes.append(working_size)
        self.episodic_sizes.append(episodic_size)
        self.semantic_item_sizes.append(semantic_item_size)
        self.active_fact_sizes.append(active_fact_count)

        self._successes_raw.append(1.0 if success else 0.0)
        cum = sum(self._successes_raw) / len(self._successes_raw)
        self.cum_successes.append(cum)

        self._violations_raw.append(1.0 if constraint_violated else 0.0)
        window = self._violations_raw[-20:]
        self.violation_rolling.append(sum(window) / len(window))

        if consolidation_event:
            self.consolidation_turns.append(turn)

        ax1, ax2, ax3, ax4 = self.axes.flat

        # Update lines
        self._line_drift.set_data(self.turns, self.drift_scores)
        if triggered:
            ax1.axvline(turn, color=COLORS["trigger"], lw=1.5, alpha=0.6)
        # Auto-scale y for drift
        if self.drift_scores:
            ax1.set_ylim(0, max(max(self.drift_scores) * 1.2, self.threshold * 1.5))

        self._line_working.set_data(self.turns, self.working_sizes)
        self._line_episodic.set_data(self.turns, self.episodic_sizes)
        self._line_sem_items.set_data(self.turns, self.semantic_item_sizes)
        self._line_facts.set_data(self.turns, self.active_fact_sizes)
        if consolidation_event:
            ax2.axvline(turn, color=COLORS["consolidation"], lw=1, alpha=0.4)
        # Auto-scale y for memory
        all_mem = self.episodic_sizes + self.semantic_item_sizes + self.active_fact_sizes
        if all_mem:
            ax2.set_ylim(0, max(max(all_mem) * 1.2, 5))

        self._line_success.set_data(self.turns, self.cum_successes)
        self._line_violation.set_data(self.turns, self.violation_rolling)

        # Refresh
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def finish(self):
        """Add legend annotations and hold the plot open."""
        ax1 = self.axes.flat[0]
        n_triggers = sum(1 for t in self.drift_triggered if t)
        n_consol = len(self.consolidation_turns)
        ax1.set_title(f"Drift Score  |  triggers={n_triggers}  consolidations={n_consol}", fontsize=10)

        ax2 = self.axes.flat[1]
        ax2.set_title(
            f"Memory  |  episodic={self.episodic_sizes[-1]}  facts={self.active_fact_sizes[-1]}",
            fontsize=10,
        )

        ax3 = self.axes.flat[2]
        ax3.set_title(f"Cumulative Success = {self.cum_successes[-1]:.3f}", fontsize=10)

        self.fig.tight_layout(rect=[0, 0.02, 1, 0.95])
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


async def run_live_single(system_name: str, scenario_name: str, n_turns: int, seed: int):
    """Run one system on one scenario with live dashboard."""
    builder = SCENARIOS.get(scenario_name)
    if not builder:
        print(f"Unknown scenario: {scenario_name}. Pick from: {list(SCENARIOS.keys())}")
        return

    scenario = builder(seed=seed, n_turns=n_turns)
    runtime, cfg = build_baseline(system_name)
    dashboard = LiveDashboard(
        title=f"{system_name}  |  {scenario_name}  |  seed={seed}  |  {n_turns} turns",
        n_turns=n_turns,
        drift_turns=scenario.drift_turns,
        threshold=runtime.drift.config.threshold,
    )

    print(f"\n{'='*70}")
    print(f"  Running: {system_name} on {scenario_name}")
    print(f"  Turns: {n_turns}  |  Seed: {seed}  |  Drift at turns: {scenario.drift_turns}")
    print(f"{'='*70}\n")

    for i, turn in enumerate(scenario.turns):
        t0 = time.perf_counter()
        result = await runtime.handle_turn(
            session_id=f"live-{scenario_name}-{seed}",
            user_id="u-live",
            user_input=turn.user_input,
            agent_output=None,
            reward=None,
            task_label=turn.task_label,
        )
        elapsed = (time.perf_counter() - t0) * 1000

        consolidation_event = False
        if cfg.fixed_consolidation_interval and (i + 1) % cfg.fixed_consolidation_interval == 0:
            await runtime.consolidation.run_once()
            consolidation_event = True
        if result.get("consolidation"):
            consolidation_event = True

        drift_sig = result["drift_signal"]
        reward = _style_reward(turn.expected_style, result["event"]["agent_output"])
        required_prec = 0.65 if (cfg.drift_enabled and cfg.use_conflict_resolution) else 0.7

        # Simple precision proxy
        retrieval = runtime.memory.retrieve(turn.oracle_fact)
        task_hit = any(turn.task_label == e.goal for e in retrieval.episodic)
        style_hit = any(turn.expected_style in s["statement"].lower() for s in retrieval.semantic)
        precision = (float(task_hit) + float(style_hit)) / 2.0
        success = reward >= 0.75 and precision >= required_prec
        constraint_violated = precision < 1.0 and turn.task_label in turn.oracle_fact

        working_size = len(runtime.memory.working)
        episodic_size = len(runtime.memory.episodic)
        semantic_item_size = len(runtime.memory.semantic)
        active_facts = len(runtime.memory.semantic.active_facts())

        # Log key events to terminal
        marker = ""
        if i in scenario.drift_turns:
            marker += " ** DRIFT INJECTED **"
        if drift_sig["triggered"]:
            marker += " >> DRIFT TRIGGERED -> consolidation"
        if consolidation_event:
            marker += " [CONSOLIDATED]"

        print(
            f"  turn {i:>3d}/{n_turns}  "
            f"drift={drift_sig['total_score']:.3f}  "
            f"mem=[W:{working_size} E:{episodic_size} S:{semantic_item_size} F:{active_facts}]  "
            f"{'OK' if success else '--'}  "
            f"{elapsed:.0f}ms"
            f"{marker}"
        )

        dashboard.update(
            turn=i,
            drift_score=drift_sig["total_score"],
            triggered=drift_sig["triggered"],
            working_size=working_size,
            episodic_size=episodic_size,
            semantic_item_size=semantic_item_size,
            active_fact_count=active_facts,
            success=success,
            constraint_violated=constraint_violated,
            consolidation_event=consolidation_event,
        )

    dashboard.finish()
    print(f"\n  Done. Final success rate: {dashboard.cum_successes[-1]:.3f}")
    print(f"  Drift triggers: {sum(1 for t in dashboard.drift_triggered if t)}")
    print(f"  Consolidation events: {len(dashboard.consolidation_turns)}")
    print(f"\n  Close the plot window to exit.")
    plt.ioff()
    plt.show()


async def run_side_by_side(scenario_name: str, n_turns: int, seed: int):
    """Run HiDrift-full vs RAG-only side by side for visual comparison."""
    builder = SCENARIOS.get(scenario_name)
    if not builder:
        print(f"Unknown scenario: {scenario_name}")
        return

    systems = ["HiDrift-full", "RAG-only"]
    scenario = builder(seed=seed, n_turns=n_turns)

    plt.ion()
    fig, axes = plt.subplots(4, 2, figsize=(16, 10), sharex="row", sharey="row")
    fig.suptitle(
        f"Side-by-Side: {systems[0]} vs {systems[1]}  |  {scenario_name}  |  {n_turns} turns",
        fontsize=13, fontweight="bold",
    )
    fig.patch.set_facecolor("#fafafa")

    panel_titles = ["Drift Score", "Memory Hierarchy", "Cumulative Success", "Violation (rolling)"]
    for row in range(4):
        for col in range(2):
            ax = axes[row][col]
            if row == 0:
                ax.set_title(f"{systems[col]}\n{panel_titles[row]}", fontsize=9)
            else:
                ax.set_title(panel_titles[row], fontsize=9)
            ax.set_xlim(0, n_turns)
            for dt in scenario.drift_turns:
                ax.axvline(dt, color="#bdc3c7", ls=":", lw=0.8, alpha=0.6)

    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    plt.show(block=False)
    plt.pause(0.01)

    for col, system_name in enumerate(systems):
        runtime, cfg = build_baseline(system_name)
        turns_x = []
        drift_scores = []
        working_s, episodic_s, facts_s = [], [], []
        cum_success = []
        violation_roll = []
        successes_raw = []
        violations_raw = []

        print(f"\n  Running {system_name}...")

        for i, turn in enumerate(scenario.turns):
            result = await runtime.handle_turn(
                session_id=f"sbs-{scenario_name}-{seed}",
                user_id="u-sbs",
                user_input=turn.user_input,
                agent_output=None,
                reward=None,
                task_label=turn.task_label,
            )
            if cfg.fixed_consolidation_interval and (i + 1) % cfg.fixed_consolidation_interval == 0:
                await runtime.consolidation.run_once()
            if result.get("consolidation"):
                pass  # already counted

            drift_sig = result["drift_signal"]
            reward = _style_reward(turn.expected_style, result["event"]["agent_output"])
            retrieval = runtime.memory.retrieve(turn.oracle_fact)
            task_hit = any(turn.task_label == e.goal for e in retrieval.episodic)
            style_hit = any(turn.expected_style in s["statement"].lower() for s in retrieval.semantic)
            precision = (float(task_hit) + float(style_hit)) / 2.0
            req_prec = 0.65 if (cfg.drift_enabled and cfg.use_conflict_resolution) else 0.7
            success = reward >= 0.75 and precision >= req_prec
            constraint_violated = precision < 1.0 and turn.task_label in turn.oracle_fact

            turns_x.append(i)
            drift_scores.append(drift_sig["total_score"])
            working_s.append(len(runtime.memory.working))
            episodic_s.append(len(runtime.memory.episodic))
            facts_s.append(len(runtime.memory.semantic.active_facts()))
            successes_raw.append(1.0 if success else 0.0)
            cum_success.append(sum(successes_raw) / len(successes_raw))
            violations_raw.append(1.0 if constraint_violated else 0.0)
            window = violations_raw[-20:]
            violation_roll.append(sum(window) / len(window))

            # Update every 5 turns for speed
            if i % 5 == 0 or i == len(scenario.turns) - 1:
                axes[0][col].cla()
                axes[0][col].plot(turns_x, drift_scores, color=COLORS["drift_score"], lw=1)
                axes[0][col].axhline(runtime.drift.config.threshold, color=COLORS["threshold"], ls="--", lw=0.8)
                axes[0][col].set_title(f"{system_name}\nDrift Score", fontsize=9)
                axes[0][col].set_xlim(0, n_turns)
                for dt in scenario.drift_turns:
                    axes[0][col].axvline(dt, color="#bdc3c7", ls=":", lw=0.8, alpha=0.6)

                axes[1][col].cla()
                axes[1][col].plot(turns_x, episodic_s, color=COLORS["episodic"], lw=1, label="episodic")
                axes[1][col].plot(turns_x, facts_s, color=COLORS["active_facts"], lw=1, label="active facts")
                axes[1][col].plot(turns_x, working_s, color=COLORS["working"], lw=1, label="working")
                axes[1][col].set_title("Memory Hierarchy", fontsize=9)
                axes[1][col].set_xlim(0, n_turns)
                axes[1][col].legend(fontsize=6, loc="upper left")

                axes[2][col].cla()
                axes[2][col].plot(turns_x, cum_success, color=COLORS["success"], lw=1.3)
                axes[2][col].set_title("Cumulative Success", fontsize=9)
                axes[2][col].set_xlim(0, n_turns)
                axes[2][col].set_ylim(0, 1.05)
                for dt in scenario.drift_turns:
                    axes[2][col].axvline(dt, color="#bdc3c7", ls=":", lw=0.8, alpha=0.6)

                axes[3][col].cla()
                axes[3][col].plot(turns_x, violation_roll, color=COLORS["violation"], lw=1)
                axes[3][col].set_title("Violation (rolling 20)", fontsize=9)
                axes[3][col].set_xlim(0, n_turns)
                axes[3][col].set_ylim(0, 1.05)
                for dt in scenario.drift_turns:
                    axes[3][col].axvline(dt, color="#bdc3c7", ls=":", lw=0.8, alpha=0.6)

                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(0.001)

        print(f"  {system_name} done. Final success: {cum_success[-1]:.3f}")

    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.canvas.draw_idle()
    print(f"\n  Close the plot window to exit.")
    plt.ioff()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Live visual demo of HiDrift experiment")
    parser.add_argument("--system", default="HiDrift-full", help="System to run (default: HiDrift-full)")
    parser.add_argument("--scenario", default="personal_assistant_drift",
                        choices=list(SCENARIOS.keys()), help="Scenario to run")
    parser.add_argument("--turns", type=int, default=120, help="Number of turns (default: 120)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--side-by-side", action="store_true",
                        help="Run HiDrift-full vs RAG-only side-by-side comparison")
    args = parser.parse_args()

    if args.side_by_side:
        asyncio.run(run_side_by_side(args.scenario, args.turns, args.seed))
    else:
        asyncio.run(run_live_single(args.system, args.scenario, args.turns, args.seed))


if __name__ == "__main__":
    main()
