from __future__ import annotations

import random
from dataclasses import dataclass


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

