from __future__ import annotations

import json

from hidrift.eval.baselines import default_systems
from hidrift.eval.benchmarks import build_scenario_suite


def test_default_systems_include_external_style_baselines() -> None:
    systems = default_systems()
    assert "MemGPT-style" in systems
    assert "GenerativeAgents-style" in systems
    assert "FlatMem-TopK" in systems


def test_build_scenario_suite_with_manifest(tmp_path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "user_input": "Respond concise for calendar.",
                        "expected_style": "concise",
                        "task_label": "calendar",
                        "oracle_fact": "user_pref_style:concise;task:calendar",
                        "drift": False,
                    }
                ),
                json.dumps(
                    {
                        "user_input": "Now switch to bullet for project updates.",
                        "expected_style": "bullet",
                        "task_label": "project",
                        "oracle_fact": "user_pref_style:bullet;task:project",
                        "drift": True,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "external_scenarios": [
                    {
                        "name": "tmp_external_trace",
                        "path": str(trace),
                        "enabled": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    scenarios = build_scenario_suite(
        seed=11,
        n_turns=8,
        benchmark_profile="external_v1",
        manifest_path=str(manifest),
    )
    assert len(scenarios) == 1
    assert scenarios[0].name == "tmp_external_trace"
    assert len(scenarios[0].turns) == 2
    assert scenarios[0].drift_turns == [1]
