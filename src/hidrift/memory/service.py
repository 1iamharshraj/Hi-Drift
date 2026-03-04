from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import timezone

from hidrift.memory.episodic import EpisodicMemory
from hidrift.memory.router import MemoryRouter
from hidrift.memory.semantic import SemanticMemory
from hidrift.memory.working import WorkingMemory
from hidrift.schemas import EpisodeRecord, InteractionEvent
from hidrift.utils import embed_text


@dataclass
class RetrievalBundle:
    working: list[InteractionEvent]
    episodic: list[EpisodeRecord]
    semantic: list[dict]
    hard_constraints: list[dict]
    supporting_context: list[dict]


class MemoryService:
    def __init__(
        self,
        working_maxlen: int = 20,
        dedup_threshold: float = 0.88,
    ) -> None:
        self.working = WorkingMemory(maxlen=working_maxlen)
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory(dedup_threshold=dedup_threshold)
        self.router = MemoryRouter()

    def ingest_interaction(
        self,
        event: InteractionEvent,
        drift_context: dict,
        recency: float = 1.0,
        usage: float = 0.0,
        stability: float = 1.0,
    ) -> EpisodeRecord:
        self.working.add(event)
        reward = event.reward if event.reward is not None else 0.0
        importance = self.router.importance_score(
            recency=recency,
            usage=usage,
            reward=max(min((reward + 1.0) / 2.0, 1.0), 0.0),
            stability=stability,
        )
        combined_text = f"{event.user_input}\n{event.agent_output}"
        episode = EpisodeRecord(
            episode_id=str(uuid.uuid4()),
            session_id=event.session_id,
            start_ts=event.timestamp.astimezone(timezone.utc),
            end_ts=event.timestamp.astimezone(timezone.utc),
            goal=event.task_label or "general_assistance",
            actions=[{"agent_output": event.agent_output, "tool_calls": event.tool_calls}],
            outcomes=[{"reward": event.reward}],
            reward_sum=reward,
            embedding=embed_text(combined_text),
            importance=importance,
            drift_context=drift_context,
        )
        self.episodic.add(episode)
        return episode

    def retrieve(self, query: str, k_working: int = 5, k_episodic: int = 5, k_semantic: int = 5) -> RetrievalBundle:
        query_embedding = embed_text(query)
        recent = self.working.get_recent(k_working)
        episodic = self.episodic.top_k(query_embedding, k=k_episodic)
        semantic = [item.model_dump() for item in self.semantic.top_k(query_embedding, k=k_semantic)]
        hybrid = self.semantic.hybrid_retrieve(query_embedding, k=k_semantic)
        return RetrievalBundle(
            working=recent,
            episodic=episodic,
            semantic=semantic,
            hard_constraints=[f.model_dump() for f in hybrid["hard_constraints"]],
            supporting_context=[f.model_dump() for f in hybrid["supporting_context"]],
        )
