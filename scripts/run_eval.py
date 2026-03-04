from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hidrift.eval.runner import run_experiment


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HiDrift evaluation.")
    parser.add_argument("--systems", nargs="*", default=None, help="Optional list of system names.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Optional integer seed list.")
    parser.add_argument("--n-turns", type=int, default=120, help="Number of turns per synthetic scenario.")
    parser.add_argument("--output-dir", default="artifacts", help="Directory for eval JSON outputs.")
    parser.add_argument(
        "--benchmark-profile",
        default="publishable_v1",
        choices=["internal_v1", "external_v1", "mixed_v1", "publishable_v1", "iccv_v1"],
        help="Benchmark suite profile to execute.",
    )
    parser.add_argument(
        "--benchmark-manifest",
        default="configs/eval/benchmark_manifest.json",
        help="JSON manifest that defines external benchmark traces.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    report = run_experiment(
        systems=args.systems,
        seeds=args.seeds,
        n_turns=args.n_turns,
        output_dir=args.output_dir,
        benchmark_profile=args.benchmark_profile,
        benchmark_manifest=args.benchmark_manifest,
    )
    systems = report.get("systems", {})
    summary = {
        "run_id": report.get("run_id"),
        "n_systems": len(systems),
        "systems": {k: v.get("aggregate_mean", {}) for k, v in systems.items()},
        "scenario_count": len(report.get("scenario_reports", {})),
        "hypothesis_count": len(report.get("hypothesis_results", {})),
        "benchmark_profile": report.get("benchmark_protocol", {}).get("benchmark_profile"),
    }
    print(json.dumps(summary, indent=2))
