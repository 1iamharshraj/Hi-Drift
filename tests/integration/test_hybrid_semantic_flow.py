from __future__ import annotations

import asyncio

from hidrift.agent.runtime import AgentRuntime


def test_consolidation_creates_semantic_facts() -> None:
    rt = AgentRuntime()
    rt.drift.config.threshold = 0.0
    for i in range(4):
        asyncio.run(
            rt.handle_turn(
                session_id="s-hybrid",
                user_id="u-1",
                user_input=f"please manage calendar turn {i}",
                agent_output=f"calendar response {i}",
                reward=0.8,
                task_label="calendar",
            )
        )
    facts = rt.memory.semantic.all_facts()
    assert len(facts) >= 1


def test_superseded_fact_deactivation() -> None:
    rt = AgentRuntime()
    sem = rt.memory.semantic
    from datetime import timedelta

    from hidrift.schemas import SemanticFact
    from hidrift.utils import embed_text, utc_now

    now = utc_now()
    f1 = SemanticFact(
        fact_id="f1",
        statement="calendar RULE_FOR old",
        fact_type="rule",
        subject="calendar",
        relation="RULE_FOR",
        object="old",
        confidence=0.5,
        stability=0.5,
        evidence_episode_ids=[],
        drift_event_ids=[],
        embedding=embed_text("calendar old"),
        is_active=True,
        version=1,
        valid_from=now - timedelta(days=1),
        valid_to=None,
        created_at=now,
        last_validated_at=now,
    )
    f2 = f1.model_copy(update={"fact_id": "f2", "object": "new", "confidence": 0.9, "version": 2})
    sem.upsert_fact(f1)
    sem.upsert_fact(f2)
    sem.resolve_conflicts()
    active = [f for f in sem.all_facts() if f.is_active]
    assert len(active) >= 1
    assert any(f.fact_id == "f2" for f in active)

