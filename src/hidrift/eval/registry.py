from __future__ import annotations

from typing import Any


def validate_benchmark_registry(report: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    scenarios = set(report.get("scenario_reports", {}).keys())
    track_results = []
    overall_ok = True

    for track in registry.get("tracks", []):
        track_id = track.get("track_id", "unknown")
        required = bool(track.get("required_for_publication", False))
        expected = set(track.get("scenarios", []))
        missing = sorted(expected - scenarios)
        ok = not missing
        if required and not ok:
            overall_ok = False
        track_results.append(
            {
                "track_id": track_id,
                "required_for_publication": required,
                "expected_scenarios": sorted(expected),
                "missing_scenarios": missing,
                "passed": ok,
            }
        )

    return {
        "passed": overall_ok,
        "observed_scenarios": sorted(scenarios),
        "track_results": track_results,
    }
