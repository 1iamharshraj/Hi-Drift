from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass

from hidrift.consolidation.worker import ConsolidationWorker
from hidrift.drift.service import DriftService
from hidrift.llm.factory import build_llm_client
from hidrift.memory.service import MemoryService
from hidrift.schemas import InteractionEvent
from hidrift.utils import utc_now


@dataclass
class RuntimeConfig:
    k_working: int = 5
    k_episodic: int = 5
    k_semantic: int = 5
    llm_provider: str = "gemini"
    llm_model: str = "gemini-2.5-flash"
    require_llm: bool = False


class AgentRuntime:
    def __init__(
        self,
        memory_service: MemoryService | None = None,
        drift_service: DriftService | None = None,
        config: RuntimeConfig | None = None,
        skip_conflict_resolution: bool = False,
    ) -> None:
        self.config = config or RuntimeConfig()
        self.memory = memory_service or MemoryService()
        self.drift = drift_service or DriftService()
        self.llm = build_llm_client(
            provider=self.config.llm_provider,
            model_name=self.config.llm_model,
            fail_if_unconfigured=self.config.require_llm,
        )
        self.consolidation = ConsolidationWorker(
            self.memory, llm_client=self.llm,
            skip_conflict_resolution=skip_conflict_resolution,
        )
        self._last_consolidation_stats: dict[str, int] | None = None

    async def handle_turn(
        self,
        session_id: str,
        user_id: str,
        user_input: str,
        agent_output: str | None = None,
        reward: float | None = None,
        task_label: str | None = None,
    ) -> dict:
        if agent_output is None:
            prompt = (
                "Respond to the user's request in a helpful assistant style.\n"
                f"User request: {user_input}"
            )
            agent_output = self.llm.generate(
                prompt=prompt,
                system_prompt="You are HiDrift assistant. Be accurate and concise.",
            )
        event = InteractionEvent(
            event_id=str(uuid.uuid4()),
            session_id=session_id,
            user_id=user_id,
            timestamp=utc_now(),
            user_input=user_input,
            agent_output=agent_output,
            tool_calls=[],
            reward=reward,
            task_label=task_label,
        )
        signal = self.drift.process(event)
        episode = self.memory.ingest_interaction(
            event=event,
            drift_context=signal.model_dump(),
            recency=1.0,
            usage=0.0,
            stability=max(1.0 - signal.total_score, 0.0),
        )
        if signal.triggered:
            self._last_consolidation_stats = await self.consolidation.run_once()
        retrieval = self.memory.retrieve(
            query=user_input,
            k_working=self.config.k_working,
            k_episodic=self.config.k_episodic,
            k_semantic=self.config.k_semantic,
        )
        await asyncio.sleep(0)
        return {
            "event": event.model_dump(),
            "drift_signal": signal.model_dump(),
            "episode_id": episode.episode_id,
            "retrieval": {
                "working_count": len(retrieval.working),
                "episodic_count": len(retrieval.episodic),
                "semantic_count": len(retrieval.semantic),
                "hard_constraints_count": len(retrieval.hard_constraints),
                "supporting_context_count": len(retrieval.supporting_context),
            },
            "consolidation": self._last_consolidation_stats,
        }

    def current_drift(self) -> dict | None:
        signal = self.drift.current()
        return signal.model_dump() if signal else None
