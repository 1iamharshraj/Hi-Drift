from __future__ import annotations

from typing import Protocol

from hidrift.schemas import GraphEdge, GraphNode


class SemanticGraphStore(Protocol):
    def upsert_node(self, node: GraphNode) -> None:
        ...

    def upsert_edge(self, edge: GraphEdge) -> None:
        ...

    def get_subgraph(self, entity_id: str, hops: int = 2) -> dict:
        ...

    def get_neighbors(self, node_id: str) -> list[str]:
        ...

    def to_dict(self) -> dict:
        ...

    def load_dict(self, payload: dict) -> None:
        ...

