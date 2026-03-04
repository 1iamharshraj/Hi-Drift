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

from hidrift.eval.publication import evaluate_publication_readiness


EVAL_REPORT_RE = re.compile(r"^eval_([0-9a-fA-F-]{36})\.json$")


def _latest_eval_report(artifacts_dir: Path, reference_system: str) -> Path | None:
    candidates = [p for p in artifacts_dir.glob("eval_*.json") if EVAL_REPORT_RE.match(p.name)]
    if not candidates:
        return None
    scored: list[tuple[int, int, float, Path]] = []
    for path in candidates:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        systems = report.get("systems", {})
        n_systems = len(systems)
        n_seeds = int(systems.get(reference_system, {}).get("n_seeds", 0))
        scored.append((n_seeds, n_systems, path.stat().st_mtime, path))
    if not scored:
        return None
    scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return scored[0][3]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check publication readiness gates from latest eval artifact.")
    parser.add_argument("--artifacts", default="artifacts", help="Artifact directory containing eval_*.json")
    parser.add_argument("--eval-report", default=None, help="Optional explicit eval report path")
    parser.add_argument("--policy", default="configs/eval/publication_policy.json", help="Publication policy JSON")
    parser.add_argument("--output-json", default="paper/tables/publication_readiness.json", help="Output JSON report path")
    parser.add_argument("--output-md", default="paper/tables/publication_readiness.md", help="Output markdown report path")
    return parser.parse_args()


def _to_markdown(result: dict) -> str:
    lines = [
        "# Publication Readiness",
        "",
        f"- overall: {'PASS' if result.get('passed') else 'FAIL'}",
        f"- reference_system: {result.get('reference_system')}",
        "",
        "| gate_id | passed | details |",
        "|---|---:|---|",
    ]
    for gate in result.get("gates", []):
        lines.append(f"| {gate.get('gate_id')} | {str(gate.get('passed')).lower()} | {gate.get('details')} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    policy = json.loads(Path(args.policy).read_text(encoding="utf-8"))
    eval_path = (
        Path(args.eval_report)
        if args.eval_report
        else _latest_eval_report(Path(args.artifacts), reference_system=policy.get("reference_system", "HiDrift-full"))
    )
    if eval_path is None or not eval_path.exists():
        print("No eval report found. Run scripts/run_eval.py first.")
        return 2
    report = json.loads(eval_path.read_text(encoding="utf-8"))
    result = evaluate_publication_readiness(report, policy)
    result["eval_report_path"] = str(eval_path)

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    out_md.write_text(_to_markdown(result), encoding="utf-8")

    summary = {
        "status": "PASS" if result.get("passed") else "FAIL",
        "eval_report_path": str(eval_path),
        "output_json": str(out_json),
        "output_md": str(out_md),
    }
    print(json.dumps(summary, indent=2))
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
