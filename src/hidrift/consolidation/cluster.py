from __future__ import annotations

from collections import defaultdict

from hidrift.schemas import EpisodeRecord


def cluster_episodes(episodes: list[EpisodeRecord]) -> dict[str, list[EpisodeRecord]]:
    # Lightweight deterministic grouping by goal for reproducible baseline behavior.
    clusters: dict[str, list[EpisodeRecord]] = defaultdict(list)
    for ep in episodes:
        clusters[ep.goal].append(ep)
    return dict(clusters)

