from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np


def _load_latest_eval_report(artifacts: Path) -> dict | None:
    pattern = re.compile(r"^eval_[0-9a-fA-F-]{36}\.json$")
    candidates = [p for p in artifacts.glob("eval_*.json") if pattern.match(p.name)]
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict) and "systems" in payload:
            return payload
    return None


def _ci95(values: list[float]) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    arr = np.array(values, dtype=float)
    return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def _write_main_table(report: dict, out_path: Path) -> None:
    systems = report.get("systems", {})
    lines = [
        "| System | Success(mean) | Success CI95 | Precision(mean) | Hallucination(mean) | Adapt Latency(mean) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, payload in systems.items():
        per_seed = payload.get("per_seed", [])
        success_vals = [r.get("task_success_rate", 0.0) for r in per_seed]
        success_ci = _ci95(success_vals)
        agg = payload.get("aggregate_mean", {})
        lines.append(
            f"| {name} | {agg.get('task_success_rate', 0.0):.4f} | [{success_ci[0]:.4f}, {success_ci[1]:.4f}] | "
            f"{agg.get('retrieval_precision_at_k', 0.0):.4f} | {agg.get('hallucination_rate', 0.0):.4f} | "
            f"{agg.get('adaptation_latency', 0.0):.4f} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_significance_table(report: dict, out_path: Path) -> None:
    stats = report.get("significance_vs_hidrift_full", {})
    lines = [
        "| System vs HiDrift-full | Metric | p_value | Effect Size (d) |",
        "| --- | --- | --- | --- |",
    ]
    for system, metrics in stats.items():
        for metric, row in metrics.items():
            lines.append(
                f"| {system} | {metric} | {row.get('p_value', 1.0):.6f} | {row.get('effect_size_cohen_d', 0.0):.4f} |"
            )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_scenario_table(report: dict, out_path: Path) -> None:
    scenario_reports = report.get("scenario_reports", {})
    lines = [
        "| Scenario | System | Success | Precision | Hallucination | Adapt Latency |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for scenario, systems in scenario_reports.items():
        for system, payload in systems.items():
            agg = payload.get("aggregate_mean", {})
            lines.append(
                f"| {scenario} | {system} | {agg.get('task_success_rate', 0.0):.4f} | "
                f"{agg.get('retrieval_precision_at_k', 0.0):.4f} | {agg.get('hallucination_rate', 0.0):.4f} | "
                f"{agg.get('adaptation_latency', 0.0):.4f} |"
            )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(report: dict, fig_dir: Path) -> list[Path]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return []
    created: list[Path] = []
    systems = list(report.get("systems", {}).keys())
    if not systems:
        return created
    agg = [report["systems"][s]["aggregate_mean"] for s in systems]

    # Main metric comparison with error bars from seeds.
    fig, ax = plt.subplots(figsize=(10, 4))
    success_means = [a.get("task_success_rate", 0.0) for a in agg]
    success_err = []
    for s in systems:
        vals = [r.get("task_success_rate", 0.0) for r in report["systems"][s]["per_seed"]]
        if len(vals) > 1:
            success_err.append(float(np.std(vals)))
        else:
            success_err.append(0.0)
    ax.bar(systems, success_means, yerr=success_err, capsize=4)
    ax.set_title("task_success_rate_with_seed_variance")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    p1 = fig_dir / "task_success_with_errorbars.png"
    fig.savefig(p1, dpi=160)
    plt.close(fig)
    created.append(p1)

    # Adaptation latency distribution across systems.
    fig_lat, ax_lat = plt.subplots(figsize=(10, 4))
    latency_data = []
    labels = []
    for s in systems:
        vals = [r.get("adaptation_latency", 0.0) for r in report["systems"][s]["per_seed"]]
        latency_data.append(vals)
        labels.append(s)
    ax_lat.boxplot(latency_data, tick_labels=labels, showmeans=True)
    ax_lat.set_title("adaptation_latency_distribution")
    ax_lat.tick_params(axis="x", rotation=20)
    fig_lat.tight_layout()
    p_lat = fig_dir / "adaptation_latency_distribution.png"
    fig_lat.savefig(p_lat, dpi=160)
    plt.close(fig_lat)
    created.append(p_lat)

    # Scenario heatmap.
    scenario_reports = report.get("scenario_reports", {})
    scenarios = list(scenario_reports.keys())
    if scenarios:
        matrix = np.zeros((len(scenarios), len(systems)), dtype=float)
        for i, sc in enumerate(scenarios):
            for j, sy in enumerate(systems):
                matrix[i, j] = scenario_reports.get(sc, {}).get(sy, {}).get("aggregate_mean", {}).get("task_success_rate", 0.0)
        fig2, ax2 = plt.subplots(figsize=(1.6 * max(3, len(systems)), 0.8 * max(3, len(scenarios))))
        im = ax2.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0)
        ax2.set_xticks(range(len(systems)))
        ax2.set_xticklabels(systems, rotation=20, ha="right")
        ax2.set_yticks(range(len(scenarios)))
        ax2.set_yticklabels(scenarios)
        ax2.set_title("scenario_success_heatmap")
        fig2.colorbar(im, ax=ax2, fraction=0.03)
        fig2.tight_layout()
        p2 = fig_dir / "scenario_success_heatmap.png"
        fig2.savefig(p2, dpi=160)
        plt.close(fig2)
        created.append(p2)

    # Trace-based drift/consolidation trends.
    traces = report.get("traces", [])
    if traces:
        by_system: dict[str, list[dict]] = {}
        for row in traces:
            by_system.setdefault(row["system"], []).append(row)
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        for system, rows in by_system.items():
            rows = sorted(rows, key=lambda r: (r["seed"], r["turn"]))
            xs = list(range(len(rows)))
            ys = [r.get("drift_score", 0.0) for r in rows]
            if ys:
                ax3.plot(xs, ys, alpha=0.5, label=system)
        ax3.set_title("drift_score_trace_by_system")
        ax3.set_xlabel("trace index")
        ax3.set_ylabel("drift score")
        ax3.legend(loc="upper right", fontsize=7)
        fig3.tight_layout()
        p3 = fig_dir / "drift_score_trace.png"
        fig3.savefig(p3, dpi=160)
        plt.close(fig3)
        created.append(p3)

        fig4, ax4 = plt.subplots(figsize=(10, 4))
        for system, rows in by_system.items():
            rows = sorted(rows, key=lambda r: (r["seed"], r["turn"]))
            xs = list(range(len(rows)))
            ys = [r.get("memory_items", 0.0) for r in rows]
            if ys:
                ax4.plot(xs, ys, alpha=0.5, label=system)
        ax4.set_title("memory_growth_trace_by_system")
        ax4.set_xlabel("trace index")
        ax4.set_ylabel("memory items")
        ax4.legend(loc="upper left", fontsize=7)
        fig4.tight_layout()
        p4 = fig_dir / "memory_growth_trace.png"
        fig4.savefig(p4, dpi=160)
        plt.close(fig4)
        created.append(p4)

        fig5, ax5 = plt.subplots(figsize=(10, 4))
        for system, rows in by_system.items():
            rows = sorted(rows, key=lambda r: (r["seed"], r["turn"]))
            xs = list(range(len(rows)))
            ys = [1.0 if r.get("hallucinated", False) else 0.0 for r in rows]
            if ys:
                # Smoothed rolling mean.
                window = 50
                smoothed = []
                for i in range(len(ys)):
                    lo = max(0, i - window + 1)
                    segment = ys[lo : i + 1]
                    smoothed.append(float(np.mean(segment)))
                ax5.plot(xs, smoothed, alpha=0.8, label=system)
        ax5.set_title("constraint_violation_trend_proxy")
        ax5.set_xlabel("trace index")
        ax5.set_ylabel("violation rate (rolling)")
        ax5.legend(loc="upper right", fontsize=7)
        fig5.tight_layout()
        p5 = fig_dir / "constraint_violation_trend.png"
        fig5.savefig(p5, dpi=160)
        plt.close(fig5)
        created.append(p5)

    return created


def main() -> None:
    artifacts = Path("artifacts")
    report = _load_latest_eval_report(artifacts)
    if report is None:
        print("No valid eval report found in artifacts/.")
        return
    fig_dir = Path("paper/figures")
    table_dir = Path("paper/tables")
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    (table_dir / "aggregate_metrics.json").write_text(json.dumps(report.get("systems", {}), indent=2), encoding="utf-8")
    _write_main_table(report, table_dir / "aggregate_metrics.md")
    _write_significance_table(report, table_dir / "significance_report.md")
    _write_scenario_table(report, table_dir / "scenario_metrics.md")
    charts = _plot(report, fig_dir)
    print(f"Exported tables: {table_dir / 'aggregate_metrics.md'}, {table_dir / 'significance_report.md'}, {table_dir / 'scenario_metrics.md'}")
    if charts:
        for chart in charts:
            print(f"Exported chart: {chart}")
    else:
        print("Matplotlib not installed; chart export skipped.")


if __name__ == "__main__":
    main()
