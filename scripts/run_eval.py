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
    print(json.dumps(report, indent=2))
