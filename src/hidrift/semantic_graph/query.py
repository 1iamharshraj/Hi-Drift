from __future__ import annotations

from hidrift.schemas import SemanticFact


def by_subject(facts: list[SemanticFact], subject: str) -> list[SemanticFact]:
    return [f for f in facts if f.subject == subject and f.is_active]


def by_relation(facts: list[SemanticFact], relation: str) -> list[SemanticFact]:
    return [f for f in facts if f.relation == relation and f.is_active]


def by_entity(facts: list[SemanticFact], entity: str) -> list[SemanticFact]:
    return [f for f in facts if (f.subject == entity or f.object == entity) and f.is_active]

