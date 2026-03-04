from __future__ import annotations

from dataclasses import dataclass

from hidrift.agent.runtime import AgentRuntime, RuntimeConfig
from hidrift.drift.service import DriftService, TriggerConfig
from hidrift.memory.service import MemoryService


@dataclass
class BaselineConfig:
    name: str
    drift_enabled: bool
    fixed_consolidation_interval: int | None


def build_baseline(name: str) -> tuple[AgentRuntime, BaselineConfig]:
    base_cfg = RuntimeConfig(llm_provider="fallback", llm_model="fallback-template", require_llm=False)
    if name == "RAG-only":
        runtime = AgentRuntime(
            memory_service=MemoryService(),
            drift_service=DriftService(TriggerConfig(threshold=9999.0)),
            config=base_cfg,
        )
        return runtime, BaselineConfig(name=name, drift_enabled=False, fixed_consolidation_interval=None)
    if name == "HierMemory-noDrift":
        runtime = AgentRuntime(
            memory_service=MemoryService(),
            drift_service=DriftService(TriggerConfig(threshold=9999.0)),
            config=base_cfg,
        )
        return runtime, BaselineConfig(name=name, drift_enabled=False, fixed_consolidation_interval=15)
    if name == "HiDrift-full":
        runtime = AgentRuntime(
            memory_service=MemoryService(),
            drift_service=DriftService(TriggerConfig(threshold=0.35)),
            config=base_cfg,
        )
        return runtime, BaselineConfig(name=name, drift_enabled=True, fixed_consolidation_interval=None)
    raise ValueError(f"Unknown baseline: {name}")
