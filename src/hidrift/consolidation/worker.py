from __future__ import annotations

import asyncio
import uuid

from hidrift.consolidation.cluster import cluster_episodes
from hidrift.consolidation.prune import apply_exponential_decay, prune_low_importance
from hidrift.consolidation.summarize import extract_semantic_facts, summarize_cluster
from hidrift.memory.service import MemoryService
from hidrift.schemas import SemanticFact, SemanticMemoryItem
from hidrift.utils import embed_text, utc_now


class ConsolidationWorker:
    def __init__(
        self,
        memory_service: MemoryService,
        llm_client: object | None = None,
        dedup_threshold: float = 0.88,
        skip_conflict_resolution: bool = False,
    ) -> None:
        self.memory_service = memory_service
        self.llm_client = llm_client
        self.dedup_threshold = dedup_threshold
        self.skip_conflict_resolution = skip_conflict_resolution

    async def run_once(self, min_importance: float = 0.1, decay_k: float = 0.08) -> dict[str, int]:
        self.memory_service.episodic.apply_decay(k=decay_k)
        pruned_low = self.memory_service.episodic.prune_below(min_importance=min_importance)
        # Keep episodic memory bounded after consolidation to realize compaction gains.
        pruned_cap = self.memory_service.episodic.prune_to_top_n(40)
        episodes = self.memory_service.episodic.all()
        apply_exponential_decay(episodes, k=0.0)
        kept = prune_low_importance(episodes, min_importance=min_importance)
        clusters = cluster_episodes(kept)

        created = 0
        facts_created = 0
        for goal, group in clusters.items():
            statement = summarize_cluster(goal, group, llm_client=self.llm_client)
            confidence = min(1.0, 0.5 + 0.05 * len(group))
            item = SemanticMemoryItem(
                memory_id=str(uuid.uuid4()),
                statement=statement,
                evidence_episode_ids=[ep.episode_id for ep in group],
                confidence=confidence,
                stability=sum(max(ep.importance, 0.0) for ep in group) / max(len(group), 1),
                created_at=utc_now(),
                last_validated_at=utc_now(),
                tags=[goal, "consolidated"],
                embedding=embed_text(statement),
            )
            if self.memory_service.semantic.add(item):
                created += 1
            candidates = extract_semantic_facts(goal, group, llm_client=self.llm_client)
            for cand in candidates:
                fact = SemanticFact(
                    fact_id=str(uuid.uuid4()),
                    statement=cand["statement"],
                    fact_type=cand["fact_type"],
                    subject=cand["subject"],
                    relation=cand["relation"],
                    object=cand["object"],
                    confidence=cand["confidence"],
                    stability=cand["stability"],
                    evidence_episode_ids=[ep.episode_id for ep in group],
                    drift_event_ids=[],
                    embedding=embed_text(cand["statement"]),
                    is_active=True,
                    version=1,
                    valid_from=cand["valid_from"],
                    valid_to=cand["valid_to"],
                    created_at=utc_now(),
                    last_validated_at=utc_now(),
                    tags=cand["tags"],
                )
                self.memory_service.semantic.upsert_fact(fact)
                facts_created += 1
        if not self.skip_conflict_resolution:
            self.memory_service.semantic.resolve_conflicts()

        await asyncio.sleep(0)
        return {
            "clusters": len(clusters),
            "semantic_created": created,
            "facts_created": facts_created,
            "episodes_seen": len(episodes),
            "episodes_pruned_low_importance": pruned_low,
            "episodes_pruned_capacity": pruned_cap,
        }
