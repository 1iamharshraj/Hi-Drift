from __future__ import annotations

import json

from hidrift.eval.official_benchmarks import load_official_scenarios


def test_load_official_scenarios_from_manifest(tmp_path) -> None:
    locomo = tmp_path / "locomo.jsonl"
    locomo.write_text(
        json.dumps(
            {
                "user_input": "Need concise calendar summary.",
                "expected_style": "concise",
                "task_label": "calendar",
                "oracle_fact": "user_pref_style:concise;task:calendar",
                "drift": True,
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "official_scenarios": [
                    {"name": "locomo_official", "path": str(locomo), "required": True},
                    {"name": "longmem_official", "path": str(tmp_path / "missing.jsonl"), "required": True},
                ]
            }
        ),
        encoding="utf-8",
    )
    scenarios, missing = load_official_scenarios(str(manifest))
    assert len(scenarios) == 1
    assert scenarios[0].name == "locomo_official"
    assert missing == ["longmem_official"]
