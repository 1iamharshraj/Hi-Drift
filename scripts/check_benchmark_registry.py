from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hidrift.eval.registry import validate_benchmark_registry


EVAL_REPORT_RE = re.compile(r"^eval_([0-9a-fA-F-]{36})\.json$")


def _latest_eval_report(artifacts_dir: Path) -> Path | None:
    candidates = [p for p in artifacts_dir.glob("eval_*.json") if EVAL_REPORT_RE.match(p.name)]
    if not candidates:
        return None
    scored: list[tuple[int, int, float, Path]] = []
    for path in candidates:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        n_scenarios = len(report.get("scenario_reports", {}))
        n_systems = len(report.get("systems", {}))
        scored.append((n_scenarios, n_systems, path.stat().st_mtime, path))
    if not scored:
        return None
    scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return scored[0][3]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate benchmark registry coverage against eval report.")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--eval-report", default=None)
    parser.add_argument("--registry", default="configs/eval/benchmark_registry.json")
    parser.add_argument("--output-json", default="paper/tables/benchmark_registry_check.json")
    parser.add_argument("--output-md", default="paper/tables/benchmark_registry_check.md")
    return parser.parse_args()


def _to_markdown(result: dict) -> str:
    lines = [
        "# Benchmark Registry Check",
        "",
        f"- overall: {'PASS' if result.get('passed') else 'FAIL'}",
        "",
        "| track_id | required_for_publication | passed | missing_scenarios |",
        "|---|---:|---:|---|",
    ]
    for track in result.get("track_results", []):
        missing = ",".join(track.get("missing_scenarios", [])) or "none"
        lines.append(
            f"| {track.get('track_id')} | {str(track.get('required_for_publication')).lower()} | "
            f"{str(track.get('passed')).lower()} | {missing} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    eval_path = Path(args.eval_report) if args.eval_report else _latest_eval_report(Path(args.artifacts))
    if eval_path is None or not eval_path.exists():
        print("No eval report found. Run scripts/run_eval.py first.")
        return 2
    report = json.loads(eval_path.read_text(encoding="utf-8"))
    registry = json.loads(Path(args.registry).read_text(encoding="utf-8"))
    result = validate_benchmark_registry(report, registry)
    result["eval_report_path"] = str(eval_path)
    result["registry_path"] = args.registry

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    out_md.write_text(_to_markdown(result), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "PASS" if result.get("passed") else "FAIL",
                "eval_report_path": str(eval_path),
                "output_json": str(out_json),
                "output_md": str(out_md),
            },
            indent=2,
        )
    )
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
