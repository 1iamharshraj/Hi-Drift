from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hidrift.eval.runner import run_experiment


if __name__ == "__main__":
    report = run_experiment()
    systems = report.get("systems", {})
    summary = {
        "run_id": report.get("run_id"),
        "n_systems": len(systems),
        "systems": {k: v.get("aggregate_mean", {}) for k, v in systems.items()},
        "scenario_count": len(report.get("scenario_reports", {})),
    }
    print(json.dumps(summary, indent=2))
