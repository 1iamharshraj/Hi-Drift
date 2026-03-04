from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from hidrift.eval.simulator import SimScenario, SimTurn


@dataclass
class OfficialBenchmarkSpec:
    name: str
    path: str
    required: bool = True


def _load_jsonl(path: str, scenario_name: str, max_turns: int | None = None) -> SimScenario:
    p = Path(path)
    if not p.exists():
        return SimScenario(name=scenario_name, turns=[], drift_turns=[])
    turns: list[SimTurn] = []
    drift_turns: list[int] = []
    for idx, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
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
        if max_turns is not None and len(turns) >= max_turns:
            break
    return SimScenario(name=scenario_name, turns=turns, drift_turns=drift_turns)


def load_official_scenarios(manifest_path: str, max_turns: int | None = None) -> tuple[list[SimScenario], list[str]]:
    """
    Load "official" benchmark scenarios from local materialized files.
    Returns scenarios and a list of missing required scenario names.
    """
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    specs = [
        OfficialBenchmarkSpec(
            name=item["name"],
            path=item["path"],
            required=item.get("required", True),
        )
        for item in payload.get("official_scenarios", [])
    ]
    scenarios: list[SimScenario] = []
    missing_required: list[str] = []
    for spec in specs:
        scenario = _load_jsonl(spec.path, scenario_name=spec.name, max_turns=max_turns)
        if spec.required and not scenario.turns:
            missing_required.append(spec.name)
        scenarios.append(scenario)
    return [s for s in scenarios if s.turns], missing_required
