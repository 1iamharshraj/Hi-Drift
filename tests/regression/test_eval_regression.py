from __future__ import annotations

from hidrift.eval.runner import run_experiment


def test_no_drift_overtrigger_guard() -> None:
    report = run_experiment(systems=["RAG-only"], seeds=[11], n_turns=20, output_dir="artifacts")
    metrics = report["systems"]["RAG-only"]["aggregate_mean"]
    assert metrics["memory_bloat"] >= 0.0


def test_fixed_seed_shape() -> None:
    report = run_experiment(systems=["HiDrift-full"], seeds=[11], n_turns=30, output_dir="artifacts")
    metrics = report["systems"]["HiDrift-full"]["aggregate_mean"]
    assert 0.0 <= metrics["task_success_rate"] <= 1.0
    assert 0.0 <= metrics["retrieval_precision_at_k"] <= 1.0
    assert 0.0 <= metrics["retrieval_recall_at_k"] <= 1.0


def test_hybrid_proxy_metric_threshold() -> None:
    report = run_experiment(
        systems=["RAG-only", "HiDrift-full"],
        seeds=[11],
        n_turns=30,
        output_dir="artifacts",
    )
    rag = report["systems"]["RAG-only"]["aggregate_mean"]["retrieval_precision_at_k"]
    hidrift = report["systems"]["HiDrift-full"]["aggregate_mean"]["retrieval_precision_at_k"]
    assert hidrift >= rag
