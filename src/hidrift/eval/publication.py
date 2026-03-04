from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class GateResult:
    gate_id: str
    passed: bool
    details: str


def _mean_metric(report: dict[str, Any], system: str, metric: str) -> float:
    return float(report.get("systems", {}).get(system, {}).get("aggregate_mean", {}).get(metric, 0.0))


def _scenario_names(report: dict[str, Any]) -> set[str]:
    return set(report.get("scenario_reports", {}).keys())


def _system_names(report: dict[str, Any]) -> set[str]:
    return set(report.get("systems", {}).keys())


def _n_seeds(report: dict[str, Any], reference: str = "HiDrift-full") -> int:
    return int(report.get("systems", {}).get(reference, {}).get("n_seeds", 0))


def _pval(report: dict[str, Any], opponent: str, metric: str, adjusted: bool = True) -> float:
    block = report.get("significance_vs_hidrift_full", {}).get(opponent, {}).get(metric, {})
    key = "p_value_holm" if adjusted else "p_value"
    return float(block.get(key, 1.0))


def evaluate_publication_readiness(report: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    gates: list[GateResult] = []
    systems = _system_names(report)
    scenarios = _scenario_names(report)
    reference = policy.get("reference_system", "HiDrift-full")
    baseline_reference = policy.get("primary_baseline", "RAG-only")
    expected_external = set(policy.get("required_external_scenarios", []))
    required_systems = set(policy.get("required_systems", []))

    # Gate 1: minimum seeds for statistical reliability.
    min_seeds = int(policy.get("min_seeds", 10))
    observed_seeds = _n_seeds(report, reference=reference)
    gates.append(
        GateResult(
            gate_id="G1_min_seeds",
            passed=observed_seeds >= min_seeds,
            details=f"observed={observed_seeds}, required>={min_seeds}",
        )
    )

    # Gate 2: external benchmark coverage.
    missing_external = sorted(expected_external - scenarios)
    gates.append(
        GateResult(
            gate_id="G2_external_benchmarks",
            passed=not missing_external,
            details="missing=" + (",".join(missing_external) if missing_external else "none"),
        )
    )

    # Gate 3: baseline coverage.
    missing_systems = sorted(required_systems - systems)
    gates.append(
        GateResult(
            gate_id="G3_baseline_coverage",
            passed=not missing_systems,
            details="missing=" + (",".join(missing_systems) if missing_systems else "none"),
        )
    )

    # Gate 4: practical gain over primary baseline.
    min_success_gain = float(policy.get("min_success_gain_vs_primary_baseline", 0.10))
    ref_success = _mean_metric(report, reference, "task_success_rate")
    base_success = _mean_metric(report, baseline_reference, "task_success_rate")
    gain = ref_success - base_success
    gates.append(
        GateResult(
            gate_id="G4_success_gain",
            passed=gain >= min_success_gain,
            details=f"gain={gain:.4f}, required>={min_success_gain:.4f}",
        )
    )

    # Gate 5: adaptation latency improvement over hierarchical non-drift baseline.
    latency_baseline = policy.get("latency_baseline_system", "HierMemory-noDrift")
    min_latency_reduction = float(policy.get("min_latency_reduction_ratio", 0.25))
    ref_latency = _mean_metric(report, reference, "adaptation_latency")
    base_latency = _mean_metric(report, latency_baseline, "adaptation_latency")
    reduction = 0.0 if base_latency <= 0 else (base_latency - ref_latency) / base_latency
    gates.append(
        GateResult(
            gate_id="G5_latency_reduction",
            passed=reduction >= min_latency_reduction,
            details=f"reduction={reduction:.4f}, required>={min_latency_reduction:.4f}",
        )
    )

    # Gate 6: memory-bloat reduction over non-consolidating baseline.
    bloat_baseline = policy.get("bloat_baseline_system", "RAG-only")
    min_bloat_reduction = float(policy.get("min_bloat_reduction_ratio", 0.30))
    ref_bloat = _mean_metric(report, reference, "memory_bloat")
    base_bloat = _mean_metric(report, bloat_baseline, "memory_bloat")
    bloat_reduction = 0.0 if base_bloat <= 0 else (base_bloat - ref_bloat) / base_bloat
    gates.append(
        GateResult(
            gate_id="G6_bloat_reduction",
            passed=bloat_reduction >= min_bloat_reduction,
            details=f"reduction={bloat_reduction:.4f}, required>={min_bloat_reduction:.4f}",
        )
    )

    # Gate 7: significance on success and latency vs key opponents.
    sig_opponents = policy.get("significance_opponents", ["RAG-only", "HierMemory-noDrift"])
    alpha = float(policy.get("alpha", 0.05))
    adjusted = bool(policy.get("use_holm_adjusted_p", True))
    sig_failures = []
    for opponent in sig_opponents:
        if opponent not in systems:
            sig_failures.append(f"{opponent}:missing_system")
            continue
        p_success = _pval(report, opponent=opponent, metric="task_success_rate", adjusted=adjusted)
        p_latency = _pval(report, opponent=opponent, metric="adaptation_latency", adjusted=adjusted)
        if p_success >= alpha:
            sig_failures.append(f"{opponent}:task_success_rate_p={p_success:.4f}")
        if p_latency >= alpha:
            sig_failures.append(f"{opponent}:adaptation_latency_p={p_latency:.4f}")
    gates.append(
        GateResult(
            gate_id="G7_significance",
            passed=not sig_failures,
            details="failures=" + (";".join(sig_failures) if sig_failures else "none"),
        )
    )

    # Gate 8: mandatory hypothesis records present.
    min_hypotheses = int(policy.get("min_hypotheses", 3))
    observed_h = len(report.get("hypothesis_results", {}))
    gates.append(
        GateResult(
            gate_id="G8_hypothesis_reporting",
            passed=observed_h >= min_hypotheses,
            details=f"observed={observed_h}, required>={min_hypotheses}",
        )
    )

    passed = all(g.passed for g in gates)
    return {
        "passed": passed,
        "reference_system": reference,
        "policy_summary": {
            "min_seeds": min_seeds,
            "required_external_scenarios": sorted(expected_external),
            "required_systems": sorted(required_systems),
            "alpha": alpha,
            "use_holm_adjusted_p": adjusted,
        },
        "gates": [g.__dict__ for g in gates],
    }
