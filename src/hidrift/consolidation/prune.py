from __future__ import annotations

import math
from datetime import datetime, timezone

from hidrift.schemas import EpisodeRecord


def apply_exponential_decay(episodes: list[EpisodeRecord], k: float = 0.08) -> None:
    now = datetime.now(timezone.utc)
    for ep in episodes:
        age_days = max((now - ep.end_ts).total_seconds(), 0.0) / 86400.0
        ep.importance *= math.exp(-k * age_days)


def prune_low_importance(episodes: list[EpisodeRecord], min_importance: float) -> list[EpisodeRecord]:
    return [ep for ep in episodes if ep.importance >= min_importance]

