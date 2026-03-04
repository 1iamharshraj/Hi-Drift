from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class InteractionEvent(BaseModel):
    event_id: str
    session_id: str
    user_id: str
    timestamp: datetime
    user_input: str
    agent_output: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    reward: float | None = None
    task_label: str | None = None


class EpisodeRecord(BaseModel):
    episode_id: str
    session_id: str
    start_ts: datetime
    end_ts: datetime
    goal: str
    actions: list[dict[str, Any]]
    outcomes: list[dict[str, Any]]
    reward_sum: float
    embedding: list[float]
    importance: float
    drift_context: dict[str, Any] = Field(default_factory=dict)
    usage_count: int = 0


class SemanticMemoryItem(BaseModel):
    memory_id: str
    statement: str
    evidence_episode_ids: list[str]
    confidence: float
    stability: float
    created_at: datetime
    last_validated_at: datetime
    tags: list[str] = Field(default_factory=list)
    embedding: list[float] = Field(default_factory=list)


class SemanticFact(BaseModel):
    fact_id: str
    statement: str
    fact_type: str = "other"
    subject: str
    relation: str
    object: str
    confidence: float
    stability: float
    evidence_episode_ids: list[str] = Field(default_factory=list)
    drift_event_ids: list[str] = Field(default_factory=list)
    embedding: list[float] = Field(default_factory=list)
    is_active: bool = True
    version: int = 1
    valid_from: datetime
    valid_to: datetime | None = None
    created_at: datetime
    last_validated_at: datetime
    tags: list[str] = Field(default_factory=list)


class SemanticRelation(BaseModel):
    relation_id: str
    source_id: str
    target_id: str
    relation_type: str
    confidence: float = 1.0
    created_at: datetime


class GraphNode(BaseModel):
    node_id: str
    node_type: str
    label: str
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    edge_id: str
    source_id: str
    target_id: str
    edge_type: str
    properties: dict[str, Any] = Field(default_factory=dict)


class DriftSignal(BaseModel):
    ts: datetime
    behavioral_shift: float
    task_shift: float
    performance_drop: float
    total_score: float
    threshold: float
    triggered: bool
