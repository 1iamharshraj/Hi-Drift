from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hidrift.eval.runner import run_experiment


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an evaluation matrix from JSON config.")
    parser.add_argument(
        "--config",
        default="configs/eval/matrix_publishable.json",
        help="Path to matrix JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where eval and matrix outputs are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    matrix_name = payload.get("matrix_name", config_path.stem)
    runs = payload.get("runs", [])
    matrix_id = str(uuid.uuid4())
    matrix_report: dict = {
        "matrix_id": matrix_id,
        "matrix_name": matrix_name,
        "config_path": str(config_path),
        "runs": [],
    }
    for run_cfg in runs:
        report = run_experiment(
            systems=run_cfg.get("systems"),
            seeds=run_cfg.get("seeds"),
            n_turns=run_cfg.get("n_turns", 120),
            output_dir=args.output_dir,
            benchmark_profile=run_cfg.get("benchmark_profile", "publishable_v1"),
            benchmark_manifest=run_cfg.get("benchmark_manifest", "configs/eval/benchmark_manifest.json"),
        )
        matrix_report["runs"].append(
            {
                "name": run_cfg.get("name", "unnamed"),
                "run_id": report.get("run_id"),
                "n_systems": len(report.get("systems", {})),
                "scenario_count": len(report.get("scenario_reports", {})),
                "hypothesis_count": len(report.get("hypothesis_results", {})),
            }
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"eval_matrix_{matrix_id}.json"
    out.write_text(json.dumps(matrix_report, indent=2), encoding="utf-8")
    print(json.dumps(matrix_report, indent=2))


if __name__ == "__main__":
    main()
