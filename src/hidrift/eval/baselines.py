from __future__ import annotations

import uuid
from dataclasses import dataclass

from hidrift.agent.runtime import AgentRuntime, RuntimeConfig
from hidrift.drift.service import DriftService, TriggerConfig
from hidrift.memory.service import MemoryService


@dataclass
class BaselineConfig:
    name: str
    drift_enabled: bool
    fixed_consolidation_interval: int | None
    use_graph_semantic: bool = True
    use_vector_semantic: bool = True
    use_conflict_resolution: bool = True


def default_systems() -> list[str]:
    return [
        "RAG-only",
        "HierMemory-noDrift",
        "VectorOnly-noGraph",
        "GraphOnly-noVector",
        "MemGPT-style",
        "GenerativeAgents-style",
        "FlatMem-TopK",
        "HiDrift-noConflict",
        "HiDrift-noDriftSignal",
        "HiDrift-full",
    ]


def build_baseline(name: str) -> tuple[AgentRuntime, BaselineConfig]:
    base_cfg = RuntimeConfig(llm_provider="fallback", llm_model="fallback-template", require_llm=False)
    graph_path = f"artifacts/eval_semantic_graph_{uuid.uuid4()}.json"
    if name == "RAG-only":
        runtime = AgentRuntime(
            memory_service=MemoryService(graph_persistence_path=graph_path),
            drift_service=DriftService(TriggerConfig(threshold=9999.0)),
            config=base_cfg,
        )
        return runtime, BaselineConfig(
            name=name,
            drift_enabled=False,
            fixed_consolidation_interval=None,
            use_graph_semantic=False,
            use_vector_semantic=False,
            use_conflict_resolution=False,
        )
    if name == "HierMemory-noDrift":
        runtime = AgentRuntime(
            memory_service=MemoryService(graph_persistence_path=graph_path),
            drift_service=DriftService(TriggerConfig(threshold=9999.0)),
            config=base_cfg,
        )
        return runtime, BaselineConfig(
            name=name,
            drift_enabled=False,
            fixed_consolidation_interval=999,
            use_graph_semantic=True,
            use_vector_semantic=True,
            use_conflict_resolution=True,
        )
    if name == "HiDrift-full":
        runtime = AgentRuntime(
            memory_service=MemoryService(graph_persistence_path=graph_path),
            drift_service=DriftService(TriggerConfig(threshold=0.35, hysteresis_turns=2, cooldown_turns=1)),
            config=base_cfg,
        )
        return runtime, BaselineConfig(
            name=name,
            drift_enabled=True,
            fixed_consolidation_interval=None,
            use_graph_semantic=True,
            use_vector_semantic=True,
            use_conflict_resolution=True,
        )
    if name == "VectorOnly-noGraph":
        runtime = AgentRuntime(
            memory_service=MemoryService(graph_persistence_path=graph_path),
            drift_service=DriftService(TriggerConfig(threshold=0.35, hysteresis_turns=2, cooldown_turns=1)),
            config=base_cfg,
        )
        return runtime, BaselineConfig(
            name=name,
            drift_enabled=True,
            fixed_consolidation_interval=None,
            use_graph_semantic=False,
            use_vector_semantic=True,
            use_conflict_resolution=False,
        )
    if name == "GraphOnly-noVector":
        runtime = AgentRuntime(
            memory_service=MemoryService(graph_persistence_path=graph_path),
            drift_service=DriftService(TriggerConfig(threshold=0.35, hysteresis_turns=2, cooldown_turns=1)),
            config=base_cfg,
        )
        return runtime, BaselineConfig(
            name=name,
            drift_enabled=True,
            fixed_consolidation_interval=None,
            use_graph_semantic=True,
            use_vector_semantic=False,
            use_conflict_resolution=True,
        )
    if name == "HiDrift-noConflict":
        runtime = AgentRuntime(
            memory_service=MemoryService(graph_persistence_path=graph_path),
            drift_service=DriftService(TriggerConfig(threshold=0.35, hysteresis_turns=2, cooldown_turns=1)),
            config=base_cfg,
        )
        return runtime, BaselineConfig(
            name=name,
            drift_enabled=True,
            fixed_consolidation_interval=None,
            use_graph_semantic=True,
            use_vector_semantic=True,
            use_conflict_resolution=False,
        )
    if name == "HiDrift-noDriftSignal":
        runtime = AgentRuntime(
            memory_service=MemoryService(graph_persistence_path=graph_path),
            drift_service=DriftService(TriggerConfig(threshold=9999.0)),
            config=base_cfg,
        )
        return runtime, BaselineConfig(
            name=name,
            drift_enabled=False,
            fixed_consolidation_interval=20,
            use_graph_semantic=True,
            use_vector_semantic=True,
            use_conflict_resolution=True,
        )
    if name == "MemGPT-style":
        runtime = AgentRuntime(
            memory_service=MemoryService(graph_persistence_path=graph_path),
            drift_service=DriftService(TriggerConfig(threshold=9999.0)),
            config=base_cfg,
        )
        return runtime, BaselineConfig(
            name=name,
            drift_enabled=False,
            fixed_consolidation_interval=12,
            use_graph_semantic=False,
            use_vector_semantic=True,
            use_conflict_resolution=False,
        )
    if name == "GenerativeAgents-style":
        runtime = AgentRuntime(
            memory_service=MemoryService(graph_persistence_path=graph_path),
            drift_service=DriftService(TriggerConfig(threshold=9999.0)),
            config=base_cfg,
        )
        return runtime, BaselineConfig(
            name=name,
            drift_enabled=False,
            fixed_consolidation_interval=10,
            use_graph_semantic=True,
            use_vector_semantic=True,
            use_conflict_resolution=False,
        )
    if name == "FlatMem-TopK":
        runtime = AgentRuntime(
            memory_service=MemoryService(graph_persistence_path=graph_path),
            drift_service=DriftService(TriggerConfig(threshold=9999.0)),
            config=base_cfg,
        )
        return runtime, BaselineConfig(
            name=name,
            drift_enabled=False,
            fixed_consolidation_interval=None,
            use_graph_semantic=False,
            use_vector_semantic=True,
            use_conflict_resolution=False,
        )
    raise ValueError(f"Unknown baseline: {name}")
