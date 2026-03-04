from __future__ import annotations

from hidrift.eval.publication import evaluate_publication_readiness
from hidrift.eval.registry import validate_benchmark_registry


def _base_report() -> dict:
    return {
        "systems": {
            "HiDrift-full": {
                "n_seeds": 10,
                "aggregate_mean": {
                    "task_success_rate": 0.8,
                    "adaptation_latency": 0.8,
                    "memory_bloat": 60.0,
                },
            },
            "RAG-only": {
                "n_seeds": 10,
                "aggregate_mean": {
                    "task_success_rate": 0.6,
                    "adaptation_latency": 1.2,
                    "memory_bloat": 100.0,
                },
            },
            "HierMemory-noDrift": {
                "n_seeds": 10,
                "aggregate_mean": {
                    "task_success_rate": 0.65,
                    "adaptation_latency": 1.2,
                    "memory_bloat": 95.0,
                },
            },
            "MemGPT-style": {"n_seeds": 10, "aggregate_mean": {}},
            "GenerativeAgents-style": {"n_seeds": 10, "aggregate_mean": {}},
            "FlatMem-TopK": {"n_seeds": 10, "aggregate_mean": {}},
        },
        "scenario_reports": {
            "locomo_like_trace": {},
            "longmem_like_trace": {},
            "personal_assistant_drift": {},
        },
        "hypothesis_results": {"H1": {}, "H2": {}, "H3": {}},
        "significance_vs_hidrift_full": {
            "RAG-only": {
                "task_success_rate": {"p_value_holm": 0.01},
                "adaptation_latency": {"p_value_holm": 0.02},
            },
            "HierMemory-noDrift": {
                "task_success_rate": {"p_value_holm": 0.03},
                "adaptation_latency": {"p_value_holm": 0.01},
            },
        },
    }


def _policy() -> dict:
    return {
        "reference_system": "HiDrift-full",
        "primary_baseline": "RAG-only",
        "latency_baseline_system": "HierMemory-noDrift",
        "bloat_baseline_system": "RAG-only",
        "required_systems": [
            "RAG-only",
            "HierMemory-noDrift",
            "MemGPT-style",
            "GenerativeAgents-style",
            "FlatMem-TopK",
            "HiDrift-full",
        ],
        "required_external_scenarios": ["locomo_like_trace", "longmem_like_trace"],
        "min_seeds": 10,
        "min_success_gain_vs_primary_baseline": 0.1,
        "min_latency_reduction_ratio": 0.25,
        "min_bloat_reduction_ratio": 0.3,
        "alpha": 0.05,
        "use_holm_adjusted_p": True,
        "significance_opponents": ["RAG-only", "HierMemory-noDrift"],
        "min_hypotheses": 3,
    }


def test_publication_readiness_passes_on_strong_report() -> None:
    result = evaluate_publication_readiness(_base_report(), _policy())
    assert result["passed"] is True
    assert all(g["passed"] for g in result["gates"])


def test_publication_readiness_fails_if_external_benchmark_missing() -> None:
    report = _base_report()
    report["scenario_reports"].pop("longmem_like_trace")
    result = evaluate_publication_readiness(report, _policy())
    assert result["passed"] is False
    gate = next(g for g in result["gates"] if g["gate_id"] == "G2_external_benchmarks")
    assert gate["passed"] is False


def test_benchmark_registry_validation() -> None:
    report = _base_report()
    registry = {
        "tracks": [
            {
                "track_id": "external_proxy_v1",
                "required_for_publication": True,
                "scenarios": ["locomo_like_trace", "longmem_like_trace"],
            }
        ]
    }
    result = validate_benchmark_registry(report, registry)
    assert result["passed"] is True
