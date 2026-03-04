from __future__ import annotations

from hidrift.drift.scoring import DriftScorer, DriftWeights


def test_drift_score_weighted_sum() -> None:
    scorer = DriftScorer(DriftWeights(alpha=0.45, beta=0.30, gamma=0.25))
    score = scorer.score(behavioral_shift=0.4, task_shift=0.2, performance_drop=0.1)
    assert abs(score - (0.45 * 0.4 + 0.30 * 0.2 + 0.25 * 0.1)) < 1e-9

