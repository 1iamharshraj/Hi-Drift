from __future__ import annotations

from datetime import timedelta

from hidrift.memory.semantic import SemanticMemory
from hidrift.schemas import SemanticFact
from hidrift.utils import embed_text, utc_now


def _fact(fact_id: str, subject: str, relation: str, obj: str, confidence: float, version: int) -> SemanticFact:
    now = utc_now()
    return SemanticFact(
        fact_id=fact_id,
        statement=f"{subject} {relation} {obj}",
        fact_type="rule",
        subject=subject,
        relation=relation,
        object=obj,
        confidence=confidence,
        stability=0.7,
        evidence_episode_ids=["ep-1"],
        drift_event_ids=[],
        embedding=embed_text(f"{subject} {relation} {obj}"),
        is_active=True,
        version=version,
        valid_from=now - timedelta(days=1),
        valid_to=None,
        created_at=now,
        last_validated_at=now,
        tags=["test"],
    )


def test_graph_upsert_and_subgraph() -> None:
    sem = SemanticMemory(graph_persistence_path="artifacts/test_sem_graph.json")
    fact = _fact("f1", "calendar", "RULE_FOR", "high_success_strategy", 0.7, 1)
    sem.upsert_fact(fact)
    graph = sem.get_subgraph("calendar", hops=1)
    assert len(graph["nodes"]) >= 2
    assert len(graph["edges"]) >= 1


def test_conflict_resolution_keeps_best_fact() -> None:
    sem = SemanticMemory(graph_persistence_path="artifacts/test_sem_graph2.json")
    old_fact = _fact("f-old", "calendar", "RULE_FOR", "old", 0.6, 1)
    new_fact = _fact("f-new", "calendar", "RULE_FOR", "new", 0.9, 2)
    sem.upsert_fact(old_fact)
    sem.upsert_fact(new_fact)
    sem.resolve_conflicts()
    active = [f for f in sem.all_facts() if f.is_active]
    assert any(f.fact_id == "f-new" for f in active)
    assert all(f.fact_id != "f-old" for f in active)


def test_hybrid_retrieve_shape() -> None:
    sem = SemanticMemory(graph_persistence_path="artifacts/test_sem_graph3.json")
    sem.upsert_fact(_fact("f1", "travel", "PREFERS", "detailed", 0.8, 1))
    sem.resolve_conflicts()
    result = sem.hybrid_retrieve(embed_text("travel detailed preference"), k=3)
    assert "hard_constraints" in result
    assert "supporting_context" in result

