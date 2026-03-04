from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode, output.strip()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    py = sys.executable

    steps = [
        [
            py,
            str(root / "scripts" / "check_benchmark_registry.py"),
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
        "results": results,
    }
    out = root / "paper" / "tables" / "iccv_readiness_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
