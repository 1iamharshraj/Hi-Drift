from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hidrift.drift.calibration import calibrate_threshold


def main() -> None:
    # Placeholder no-drift scores for initial calibration.
    no_drift_scores = [0.05, 0.08, 0.11, 0.14, 0.17, 0.19, 0.22, 0.24, 0.28, 0.31]
    threshold = calibrate_threshold(no_drift_scores, quantile=0.95)
    out = {"quantile": 0.95, "threshold": threshold, "n_samples": len(no_drift_scores)}
    path = Path("artifacts/calibration.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
