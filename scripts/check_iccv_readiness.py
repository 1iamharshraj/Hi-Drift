from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _latest_matrix_report(artifacts_dir: Path) -> Path | None:
    candidates = sorted(artifacts_dir.glob("eval_matrix_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if payload.get("matrix_name") == "iccv_v1":
            return path
    return candidates[0]


def _resolve_iccv_eval_report(artifacts_dir: Path) -> Path | None:
    matrix = _latest_matrix_report(artifacts_dir)
    if matrix is None:
        return None
    payload = json.loads(matrix.read_text(encoding="utf-8"))
    runs = payload.get("runs", [])
    if not runs:
        return None
    run_id = runs[-1].get("run_id")
    if not run_id:
        return None
    candidate = artifacts_dir / f"eval_{run_id}.json"
    return candidate if candidate.exists() else None


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode, output.strip()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    py = sys.executable
    artifacts_dir = root / "artifacts"
    eval_report = _resolve_iccv_eval_report(artifacts_dir)
    if eval_report is None:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "error": "Could not resolve ICCV eval report from latest eval_matrix_*.json. Run make eval_iccv first.",
                },
                indent=2,
            )
        )
        return 2

    steps = [
        [
            py,
            str(root / "scripts" / "check_benchmark_registry.py"),
            "--eval-report",
            str(eval_report),
            "--registry",
            "configs/eval/benchmark_registry_iccv.json",
            "--output-json",
            "paper/tables/iccv_benchmark_registry_check.json",
            "--output-md",
            "paper/tables/iccv_benchmark_registry_check.md",
        ],
        [
            py,
            str(root / "scripts" / "check_publication_readiness.py"),
            "--eval-report",
            str(eval_report),
            "--policy",
            "configs/eval/iccv_policy.json",
            "--output-json",
            "paper/tables/iccv_publication_readiness.json",
            "--output-md",
            "paper/tables/iccv_publication_readiness.md",
        ],
    ]
    results = []
    ok = True
    for cmd in steps:
        code, output = _run(cmd)
        results.append({"cmd": " ".join(cmd), "exit_code": code, "output": output})
        if code != 0:
            ok = False

    summary = {
        "status": "PASS" if ok else "FAIL",
        "eval_report": str(eval_report),
        "results": results,
    }
    out = root / "paper" / "tables" / "iccv_readiness_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
