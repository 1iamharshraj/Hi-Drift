from __future__ import annotations

from hidrift.schemas import SemanticFact


def resolve_conflicts(facts: list[SemanticFact]) -> list[SemanticFact]:
    # Highest confidence then latest version survives for same (subject, relation).
    grouped: dict[tuple[str, str], list[SemanticFact]] = {}
    for fact in facts:
        grouped.setdefault((fact.subject, fact.relation), []).append(fact)
    for key, bucket in grouped.items():
        bucket.sort(key=lambda f: (f.confidence, f.version, f.created_at), reverse=True)
        winner = bucket[0]
        for fact in bucket:
            fact.is_active = fact.fact_id == winner.fact_id
    return facts

