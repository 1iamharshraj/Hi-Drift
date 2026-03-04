from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from hidrift.eval.official_benchmarks import load_official_scenarios
from hidrift.eval.simulator import (
    SimScenario,
    SimTurn,
    build_contradiction_drift_scenario,
    build_personal_assistant_scenario,
    build_semi_real_trace_scenario,
    build_tool_api_drift_scenario,
)


@dataclass
class ExternalScenarioSpec:
    name: str
    path: str
    enabled: bool = True


def _load_jsonl_scenario(path: str, name: str) -> SimScenario:
    file_path = Path(path)
    if not file_path.exists():
        return SimScenario(name=name, turns=[], drift_turns=[])
    turns: list[SimTurn] = []
    drift_turns: list[int] = []
    for idx, line in enumerate(file_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        turns.append(
            SimTurn(
                user_input=row["user_input"],
                expected_style=row["expected_style"],
                task_label=row["task_label"],
                oracle_fact=row["oracle_fact"],
            )
        )
        if row.get("drift", False):
            drift_turns.append(idx)
    return SimScenario(name=name, turns=turns, drift_turns=drift_turns)


def default_external_specs() -> list[ExternalScenarioSpec]:
    return [
        ExternalScenarioSpec(name="locomo_like_trace", path="data/benchmarks/external/locomo_like_trace.jsonl"),
        ExternalScenarioSpec(name="longmem_like_trace", path="data/benchmarks/external/longmem_like_trace.jsonl"),
    ]


def _load_manifest_specs(manifest_path: str) -> list[ExternalScenarioSpec]:
    path = Path(manifest_path)
    if not path.exists():
        return default_external_specs()
    payload = json.loads(path.read_text(encoding="utf-8"))
    specs = []
    for item in payload.get("external_scenarios", []):
        specs.append(
            ExternalScenarioSpec(
                name=item["name"],
                path=item["path"],
                enabled=item.get("enabled", True),
            )
        )
    return specs or default_external_specs()


def build_scenario_suite(
    seed: int,
    n_turns: int,
    benchmark_profile: str = "internal_v1",
    manifest_path: str | None = None,
) -> list[SimScenario]:
    scenarios: list[SimScenario] = []
    include_internal = benchmark_profile in {"internal_v1", "publishable_v1", "mixed_v1", "iccv_v1"}
    include_external = benchmark_profile in {"external_v1", "publishable_v1", "mixed_v1"}
    include_official = benchmark_profile in {"iccv_v1"}

    if include_internal:
        scenarios.extend(
            [
                build_personal_assistant_scenario(seed=seed, n_turns=n_turns),
                build_tool_api_drift_scenario(seed=seed, n_turns=n_turns),
                build_contradiction_drift_scenario(seed=seed, n_turns=n_turns),
                build_semi_real_trace_scenario(),
            ]
        )

    if include_external:
        specs = _load_manifest_specs(manifest_path) if manifest_path else default_external_specs()
        for spec in specs:
            if not spec.enabled:
                continue
            scenarios.append(_load_jsonl_scenario(path=spec.path, name=spec.name))

    if include_official and manifest_path:
        official_scenarios, _ = load_official_scenarios(manifest_path, max_turns=n_turns)
        scenarios.extend(official_scenarios)

    # Filter empty scenarios so eval loops remain stable if external files are missing.
    return [scenario for scenario in scenarios if scenario.turns]
