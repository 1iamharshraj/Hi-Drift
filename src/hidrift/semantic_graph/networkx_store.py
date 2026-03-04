from __future__ import annotations

import networkx as nx

from hidrift.schemas import GraphEdge, GraphNode


class NetworkxSemanticGraphStore:
    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()

    def upsert_node(self, node: GraphNode) -> None:
        self.graph.add_node(node.node_id, node_type=node.node_type, label=node.label, **node.properties)

    def upsert_edge(self, edge: GraphEdge) -> None:
        self.graph.add_edge(
            edge.source_id,
            edge.target_id,
            key=edge.edge_id,
            edge_type=edge.edge_type,
            **edge.properties,
        )

    def get_neighbors(self, node_id: str) -> list[str]:
        if node_id not in self.graph:
            return []
        return list(set(self.graph.successors(node_id)) | set(self.graph.predecessors(node_id)))

    def get_subgraph(self, entity_id: str, hops: int = 2) -> dict:
        if entity_id not in self.graph:
            return {"nodes": [], "edges": []}
        visited = {entity_id}
        frontier = {entity_id}
        for _ in range(max(hops, 0)):
            nxt = set()
            for node in frontier:
                nxt.update(self.get_neighbors(node))
            frontier = nxt - visited
            visited.update(frontier)
        nodes = [{"node_id": n, **self.graph.nodes[n]} for n in visited]
        edges = []
        for u, v, k, data in self.graph.edges(keys=True, data=True):
            if u in visited and v in visited:
                edges.append({"edge_id": k, "source_id": u, "target_id": v, **data})
        return {"nodes": nodes, "edges": edges}

    def to_dict(self) -> dict:
        nodes = [{"node_id": n, **data} for n, data in self.graph.nodes(data=True)]
        edges = []
        for u, v, k, data in self.graph.edges(keys=True, data=True):
            edges.append({"edge_id": k, "source_id": u, "target_id": v, **data})
        return {"nodes": nodes, "edges": edges}

    def load_dict(self, payload: dict) -> None:
        self.graph.clear()
        for node in payload.get("nodes", []):
            node_id = node["node_id"]
            props = {k: v for k, v in node.items() if k != "node_id"}
            self.graph.add_node(node_id, **props)
        for edge in payload.get("edges", []):
            source = edge["source_id"]
            target = edge["target_id"]
            edge_id = edge["edge_id"]
            props = {k: v for k, v in edge.items() if k not in {"source_id", "target_id", "edge_id"}}
            self.graph.add_edge(source, target, key=edge_id, **props)

