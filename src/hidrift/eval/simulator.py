from __future__ import annotations

import random
from pathlib import Path
from dataclasses import dataclass
import json


KNOWN_STYLES = {"concise", "detailed", "bullet"}


@dataclass
class SimTurn:
    user_input: str
    expected_style: str
    task_label: str
    oracle_fact: str


@dataclass
class SimScenario:
    name: str
    turns: list[SimTurn]
    drift_turns: list[int]


def build_personal_assistant_scenario(seed: int, n_turns: int = 120) -> SimScenario:
    rng = random.Random(seed)
    turns: list[SimTurn] = []
    drift_turns = [n_turns // 3, (2 * n_turns) // 3]
    style = "concise"
    task = "calendar"
    for i in range(n_turns):
        if i == drift_turns[0]:
            style = "detailed"
            task = "travel"
        if i == drift_turns[1]:
            style = "bullet"
            task = "project"
        fact = f"user_pref_style:{style};task:{task}"
        utterance = f"Turn {i}: help with {task} in {style} format."
        if rng.random() < 0.15:
            utterance += " Include timezone and reminders."
        turns.append(
            SimTurn(
                user_input=utterance,
                expected_style=style,
                task_label=task,
                oracle_fact=fact,
            )
        )
    return SimScenario(name="personal_assistant_drift", turns=turns, drift_turns=drift_turns)


def build_tool_api_drift_scenario(seed: int, n_turns: int = 120) -> SimScenario:
    rng = random.Random(seed + 991)
    turns: list[SimTurn] = []
    drift_turns = [n_turns // 2]
    api_version = "v1"
    for i in range(n_turns):
        if i == drift_turns[0]:
            api_version = "v2"
        task = "tooling"
        style = "detailed" if api_version == "v2" else "concise"
        user_input = f"Use calendar API {api_version} for reminder sync."
        if rng.random() < 0.2:
            user_input += " Validate schema fields."
        turns.append(
            SimTurn(
                user_input=user_input,
                expected_style=style,
                task_label=task,
                oracle_fact=f"api_schema:calendar_{api_version}",
            )
        )
    return SimScenario(name="tool_api_drift", turns=turns, drift_turns=drift_turns)


def build_contradiction_drift_scenario(seed: int, n_turns: int = 120) -> SimScenario:
    turns: list[SimTurn] = []
    drift_turns = [n_turns // 3, (2 * n_turns) // 3]
    pref = "concise"
    for i in range(n_turns):
        if i == drift_turns[0]:
            pref = "detailed"
        if i == drift_turns[1]:
            pref = "concise"
        turns.append(
            SimTurn(
                user_input=f"My reporting preference is {pref} for task handoff.",
                expected_style=pref,
                task_label="project",
                oracle_fact=f"user_pref_style:{pref};task:project",
            )
        )
    return SimScenario(name="contradiction_drift", turns=turns, drift_turns=drift_turns)


def build_semi_real_trace_scenario(path: str = "data/benchmarks/semi_real_trace.jsonl") -> SimScenario:
    p = Path(path)
    turns: list[SimTurn] = []
    drift_turns: list[int] = []
    if not p.exists():
        return SimScenario(name="semi_real_trace", turns=[], drift_turns=[])
    for idx, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
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
    return SimScenario(name="semi_real_trace", turns=turns, drift_turns=drift_turns)


def build_benchmark_suite(seed: int, n_turns: int = 120) -> list[SimScenario]:
    return [
        build_personal_assistant_scenario(seed=seed, n_turns=n_turns),
        build_tool_api_drift_scenario(seed=seed, n_turns=n_turns),
        build_contradiction_drift_scenario(seed=seed, n_turns=n_turns),
        build_semi_real_trace_scenario(),
    ]
