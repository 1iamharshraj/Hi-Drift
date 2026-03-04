from __future__ import annotations

import math
from datetime import datetime, timezone

from hidrift.schemas import EpisodeRecord
from hidrift.utils import cosine_similarity


class EpisodicMemory:
    def __init__(self) -> None:
        self._episodes: dict[str, EpisodeRecord] = {}

    def add(self, episode: EpisodeRecord) -> None:
        self._episodes[episode.episode_id] = episode

    def get(self, episode_id: str) -> EpisodeRecord | None:
        return self._episodes.get(episode_id)

    def all(self) -> list[EpisodeRecord]:
        return list(self._episodes.values())

    def top_k(self, query_embedding: list[float], k: int = 5) -> list[EpisodeRecord]:
        scored = []
        for e in self._episodes.values():
            sim = cosine_similarity(query_embedding, e.embedding)
            score = 0.7 * sim + 0.3 * e.importance
            scored.append((score, e))
            e.usage_count += 1
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:k]]

    def apply_decay(self, k: float = 0.08) -> None:
        now = datetime.now(timezone.utc)
        for ep in self._episodes.values():
            age_days = max((now - ep.end_ts).total_seconds(), 0.0) / 86400.0
            ep.importance *= math.exp(-k * age_days)

    def prune_below(self, min_importance: float) -> int:
        to_delete = [eid for eid, e in self._episodes.items() if e.importance < min_importance]
        for eid in to_delete:
            del self._episodes[eid]
        return len(to_delete)

    def __len__(self) -> int:
        return len(self._episodes)

