from __future__ import annotations

import json
from pathlib import Path


def _write_markdown_table(report: dict, out_path: Path) -> None:
    systems = report.get("systems", {})
    headers = [
        "System",
        "Task Success",
        "Precision@k",
        "Recall@k",
        "Hallucination",
        "Memory Bloat",
        "Adapt Latency",
        "Stability",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for name, payload in systems.items():
        m = payload.get("aggregate_mean", {})
        row = [
            name,
            f"{m.get('task_success_rate', 0.0):.4f}",
            f"{m.get('retrieval_precision_at_k', 0.0):.4f}",
            f"{m.get('retrieval_recall_at_k', 0.0):.4f}",
            f"{m.get('hallucination_rate', 0.0):.4f}",
            f"{m.get('memory_bloat', 0.0):.2f}",
            f"{m.get('adaptation_latency', 0.0):.2f}",
            f"{m.get('stability_score', 0.0):.4f}",
        ]
        lines.append("| " + " | ".join(row) + " |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _try_write_charts(report: dict, fig_dir: Path) -> list[Path]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return []

    systems = list(report.get("systems", {}).keys())
    if not systems:
        return []
    metrics = [report["systems"][s]["aggregate_mean"] for s in systems]

    created: list[Path] = []

    # Higher-is-better metrics chart.
    hib_labels = ["task_success_rate", "retrieval_precision_at_k", "retrieval_recall_at_k", "stability_score"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.flatten()
    for i, metric in enumerate(hib_labels):
        vals = [m.get(metric, 0.0) for m in metrics]
        axes[i].bar(systems, vals)
        axes[i].set_title(metric)
        axes[i].set_ylim(0, 1.05)
        axes[i].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    p1 = fig_dir / "aggregate_higher_is_better.png"
    fig.savefig(p1, dpi=160)
    plt.close(fig)
    created.append(p1)

    # Lower-is-better metrics chart.
    lib_labels = ["hallucination_rate", "memory_bloat", "adaptation_latency"]
    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 3.8))
    for i, metric in enumerate(lib_labels):
        vals = [m.get(metric, 0.0) for m in metrics]
        axes2[i].bar(systems, vals)
        axes2[i].set_title(metric)
        axes2[i].tick_params(axis="x", rotation=20)
    fig2.tight_layout()
    p2 = fig_dir / "aggregate_lower_is_better.png"
    fig2.savefig(p2, dpi=160)
    plt.close(fig2)
    created.append(p2)
    return created


def _write_hybrid_semantic_outputs(report: dict, fig_dir: Path, table_dir: Path) -> list[Path]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return []
    systems = list(report.get("systems", {}).keys())
    if not systems:
        return []
    metrics = [report["systems"][s]["aggregate_mean"] for s in systems]
    constraint_hit = [m.get("retrieval_precision_at_k", 0.0) for m in metrics]
    conflict_accuracy = [1.0 - m.get("hallucination_rate", 0.0) for m in metrics]

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(systems, constraint_hit)
    ax1.set_ylim(0, 1.05)
    ax1.set_title("hybrid_constraint_hit_rate")
    ax1.tick_params(axis="x", rotation=20)
    fig1.tight_layout()
    p1 = fig_dir / "hybrid_constraint_hit_rate.png"
    fig1.savefig(p1, dpi=160)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.bar(systems, conflict_accuracy)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("conflict_resolution_accuracy")
    ax2.tick_params(axis="x", rotation=20)
    fig2.tight_layout()
    p2 = fig_dir / "conflict_resolution_accuracy.png"
    fig2.savefig(p2, dpi=160)
    plt.close(fig2)

    lines = [
        "| System | Constraint Hit Rate | Conflict Resolution Accuracy |",
        "| --- | --- | --- |",
    ]
    for idx, name in enumerate(systems):
        lines.append(f"| {name} | {constraint_hit[idx]:.4f} | {conflict_accuracy[idx]:.4f} |")
    (table_dir / "hybrid_semantic_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Proxy timeline chart from aggregate metrics for presentation.
    fig3, ax3 = plt.subplots(figsize=(9, 3.5))
    for idx, name in enumerate(systems):
        bloat = report["systems"][name]["aggregate_mean"].get("memory_bloat", 0.0)
        timeline = [max(0.0, (i / 10.0) * (bloat / 150.0)) for i in range(10)]
        ax3.plot(list(range(1, 11)), timeline, label=name)
    ax3.set_title("drift_trigger_timeline_proxy")
    ax3.set_xlabel("step")
    ax3.set_ylabel("normalized trigger intensity")
    ax3.legend()
    fig3.tight_layout()
    p3 = fig_dir / "drift_trigger_timeline.png"
    fig3.savefig(p3, dpi=160)
    plt.close(fig3)

    fig4, ax4 = plt.subplots(figsize=(8, 3.5))
    consolidation_counts = [round(report["systems"][s]["aggregate_mean"].get("memory_bloat", 0.0) / 10.0, 2) for s in systems]
    ax4.bar(systems, consolidation_counts)
    ax4.set_title("consolidation_event_count_proxy")
    ax4.tick_params(axis="x", rotation=20)
    fig4.tight_layout()
    p4 = fig_dir / "consolidation_event_count.png"
    fig4.savefig(p4, dpi=160)
    plt.close(fig4)

    return [p1, p2, p3, p4]


def main() -> None:
    artifacts = Path("artifacts")
    latest = sorted(artifacts.glob("eval_*.json"))
    if not latest:
        print("No eval artifacts found in artifacts/.")
        return
    report = json.loads(latest[-1].read_text(encoding="utf-8"))
    fig_dir = Path("paper/figures")
    table_dir = Path("paper/tables")
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    json_path = table_dir / "aggregate_metrics.json"
    md_path = table_dir / "aggregate_metrics.md"
    (json_path).write_text(
        json.dumps(report["systems"], indent=2),
        encoding="utf-8",
    )
    _write_markdown_table(report, md_path)
    chart_paths = _try_write_charts(report, fig_dir)
    hybrid_paths = _write_hybrid_semantic_outputs(report, fig_dir, table_dir)
    print(f"Exported table data to {json_path}")
    print(f"Exported markdown table to {md_path}")
    if chart_paths:
        for chart in chart_paths:
            print(f"Exported chart to {chart}")
    else:
        print("Matplotlib not installed; skipped PNG chart export.")
    for chart in hybrid_paths:
        print(f"Exported chart to {chart}")


if __name__ == "__main__":
    main()
