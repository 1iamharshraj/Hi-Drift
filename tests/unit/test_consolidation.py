from __future__ import annotations

from datetime import timedelta

from hidrift.consolidation.prune import apply_exponential_decay
from hidrift.memory.service import MemoryService
from hidrift.schemas import EpisodeRecord
from hidrift.utils import utc_now


def test_decay_reduces_importance() -> None:
    now = utc_now()
    episode = EpisodeRecord(
        episode_id="e-1",
        session_id="s-1",
        start_ts=now - timedelta(days=5),
        end_ts=now - timedelta(days=5),
        goal="calendar",
        actions=[],
        outcomes=[],
        reward_sum=1.0,
        embedding=[0.1, 0.2],
        importance=1.0,
        drift_context={},
    )
    apply_exponential_decay([episode], k=0.08)
    assert episode.importance < 1.0


def test_semantic_dedup() -> None:
    memory = MemoryService(dedup_threshold=0.88)
    # Identical statements produce identical embeddings with deterministic encoder.
    now = utc_now()
    from hidrift.schemas import SemanticMemoryItem
    from hidrift.utils import embed_text

    statement = "User prefers concise answers."
    i1 = SemanticMemoryItem(
        memory_id="m1",
        statement=statement,
        evidence_episode_ids=["e1"],
        confidence=0.8,
        stability=0.7,
        created_at=now,
        last_validated_at=now,
        tags=["pref"],
        embedding=embed_text(statement),
    )
    i2 = SemanticMemoryItem(
        memory_id="m2",
        statement=statement,
        evidence_episode_ids=["e2"],
        confidence=0.8,
        stability=0.7,
        created_at=now,
        last_validated_at=now,
        tags=["pref"],
        embedding=embed_text(statement),
    )
    assert memory.semantic.add(i1) is True
    assert memory.semantic.add(i2) is False

