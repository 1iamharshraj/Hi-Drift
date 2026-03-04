from __future__ import annotations

from hidrift.memory.router import MemoryRouter


def test_keep_probability_monotonicity() -> None:
    router = MemoryRouter()
    low = router.keep_probability(importance=0.2, drift_penalty=0.5)
    high = router.keep_probability(importance=0.8, drift_penalty=0.5)
    assert high > low


def test_importance_score_bounds() -> None:
    router = MemoryRouter()
    score = router.importance_score(recency=1.0, usage=0.5, reward=1.0, stability=0.8)
    assert 0.0 <= score <= 1.0

