from __future__ import annotations

import json
import uuid
from pathlib import Path

from hidrift.schemas import GraphEdge, GraphNode, SemanticFact, SemanticMemoryItem
from hidrift.semantic_graph.networkx_store import NetworkxSemanticGraphStore
from hidrift.semantic_graph.reasoning import resolve_conflicts
from hidrift.utils import cosine_similarity


class SemanticMemory:
    def __init__(
        self,
        dedup_threshold: float = 0.88,
        graph_persistence_path: str = "artifacts/semantic_graph.json",
        fusion_weights: dict[str, float] | None = None,
    ) -> None:
        self._items: dict[str, SemanticMemoryItem] = {}
        self._facts: dict[str, SemanticFact] = {}
        self._dedup_threshold = dedup_threshold
        self.graph = NetworkxSemanticGraphStore()
        self.graph_persistence_path = Path(graph_persistence_path)
        self.fusion_weights = fusion_weights or {
            "graph": 0.45,
            "vector": 0.30,
            "confidence": 0.15,
            "stability": 0.10,
        }
        self.load_graph()

    def add(self, item: SemanticMemoryItem) -> bool:
        for existing in self._items.values():
            sim = cosine_similarity(existing.embedding, item.embedding)
            if sim >= self._dedup_threshold:
                return False
        self._items[item.memory_id] = item
        return True

    def _find_fact_id_by_triple(self, subject: str, relation: str, object_value: str) -> str | None:
        for fact_id, fact in self._facts.items():
            if fact.subject == subject and fact.relation == relation and fact.object == object_value:
                return fact_id
        return None

    def upsert_fact(self, fact: SemanticFact) -> None:
        existing_id = self._find_fact_id_by_triple(fact.subject, fact.relation, fact.object)
        if existing_id is not None and existing_id != fact.fact_id:
            existing = self._facts[existing_id]
            merged_evidence = sorted(set(existing.evidence_episode_ids) | set(fact.evidence_episode_ids))
            merged_drift = sorted(set(existing.drift_event_ids) | set(fact.drift_event_ids))
            merged_tags = sorted(set(existing.tags) | set(fact.tags))
            updated = existing.model_copy(
                update={
                    "statement": fact.statement if len(fact.statement) > len(existing.statement) else existing.statement,
                    "confidence": max(existing.confidence, fact.confidence),
                    "stability": max(existing.stability, fact.stability),
                    "embedding": fact.embedding or existing.embedding,
                    "version": max(existing.version, fact.version) + 1,
                    "last_validated_at": fact.last_validated_at,
                    "evidence_episode_ids": merged_evidence,
                    "drift_event_ids": merged_drift,
                    "tags": merged_tags,
                    "is_active": True,
                }
            )
            self._facts[existing_id] = updated
            fact = updated
        else:
            self._facts[fact.fact_id] = fact
        self.graph.upsert_node(
            GraphNode(
                node_id=fact.fact_id,
                node_type="SemanticFact",
                label=fact.statement,
                properties=fact.model_dump(mode="json"),
            )
        )
        self.graph.upsert_node(
            GraphNode(node_id=fact.subject, node_type="Entity", label=fact.subject, properties={"role": "subject"})
        )
        self.graph.upsert_node(
            GraphNode(node_id=fact.object, node_type="Entity", label=fact.object, properties={"role": "object"})
        )
        self.graph.upsert_edge(
            GraphEdge(
                edge_id=str(uuid.uuid4()),
                source_id=fact.subject,
                target_id=fact.fact_id,
                edge_type="HAS_FACT",
                properties={"relation": fact.relation},
            )
        )
        self.graph.upsert_edge(
            GraphEdge(
                edge_id=str(uuid.uuid4()),
                source_id=fact.fact_id,
                target_id=fact.object,
                edge_type=fact.relation,
                properties={"confidence": fact.confidence},
            )
        )
        for ep_id in fact.evidence_episode_ids:
            self.graph.upsert_node(GraphNode(node_id=ep_id, node_type="Episode", label=ep_id, properties={}))
            self.graph.upsert_edge(
                GraphEdge(
                    edge_id=str(uuid.uuid4()),
                    source_id=fact.fact_id,
                    target_id=ep_id,
                    edge_type="OBSERVED_IN",
                    properties={},
                )
            )
        self.persist_graph()

    def link_evidence(self, fact_id: str, episode_ids: list[str]) -> None:
        fact = self._facts.get(fact_id)
        if fact is None:
            return
        merged = set(fact.evidence_episode_ids) | set(episode_ids)
        fact.evidence_episode_ids = sorted(merged)
        self.upsert_fact(fact)

    def resolve_conflicts(self) -> None:
        resolved = resolve_conflicts(list(self._facts.values()))
        self._facts = {f.fact_id: f for f in resolved}
        self.persist_graph()

    def all_facts(self) -> list[SemanticFact]:
        return list(self._facts.values())

    def active_facts(self) -> list[SemanticFact]:
        return [f for f in self._facts.values() if f.is_active]

    def all(self) -> list[SemanticMemoryItem]:
        return list(self._items.values())

    def top_k(self, query_embedding: list[float], k: int = 5) -> list[SemanticMemoryItem]:
        scored = [(cosine_similarity(query_embedding, i.embedding), i) for i in self._items.values()]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:k]]

    def hybrid_retrieve(self, query_embedding: list[float], k: int = 5) -> dict[str, list[SemanticFact]]:
        active = self.active_facts()
        ranked = []
        for fact in active:
            vec = cosine_similarity(query_embedding, fact.embedding)
            graph_relevance = 1.0 if fact.relation in {"PREFERS", "RULE_FOR", "SUPERSEDES"} else 0.6
            score = (
                self.fusion_weights["graph"] * graph_relevance
                + self.fusion_weights["vector"] * vec
                + self.fusion_weights["confidence"] * fact.confidence
                + self.fusion_weights["stability"] * fact.stability
            )
            ranked.append((score, fact))
        ranked.sort(key=lambda x: x[0], reverse=True)
        hard_constraints = [f for _, f in ranked if f.relation in {"PREFERS", "RULE_FOR", "SUPERSEDES"}][:k]
        supporting_context = [f for _, f in ranked[:k]]
        return {"hard_constraints": hard_constraints, "supporting_context": supporting_context}

    def get_subgraph(self, entity_id: str, hops: int = 2) -> dict:
        return self.graph.get_subgraph(entity_id, hops=hops)

    def persist_graph(self) -> None:
        payload = {
            "graph": self.graph.to_dict(),
            "facts": [f.model_dump(mode="json") for f in self._facts.values()],
        }
        self.graph_persistence_path.parent.mkdir(parents=True, exist_ok=True)
        self.graph_persistence_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_graph(self) -> None:
        if not self.graph_persistence_path.exists():
            return
        payload = json.loads(self.graph_persistence_path.read_text(encoding="utf-8"))
        self.graph.load_dict(payload.get("graph", {}))
        facts: dict[str, SemanticFact] = {}
        for raw in payload.get("facts", []):
            fact = SemanticFact.model_validate(raw)
            facts[fact.fact_id] = fact
        self._facts = facts

    def __len__(self) -> int:
        return len(self._items)
