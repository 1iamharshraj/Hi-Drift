from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from hidrift.agent.runtime import AgentRuntime, RuntimeConfig
from hidrift.schemas import SemanticFact

try:
    from fastapi import FastAPI
except Exception:  # pragma: no cover
    FastAPI = None  # type: ignore[assignment]


class IngestRequest(BaseModel):
    session_id: str
    user_id: str
    user_input: str
    agent_output: str | None = None
    reward: float | None = None
    task_label: str | None = None


class RetrieveRequest(BaseModel):
    query: str
    k_working: int = 5
    k_episodic: int = 5
    k_semantic: int = 5


class ConsolidationRequest(BaseModel):
    min_importance: float = Field(default=0.1, ge=0.0, le=1.0)
    decay_k: float = Field(default=0.08, ge=0.0, le=1.0)


class SemanticUpsertRequest(BaseModel):
    fact: SemanticFact


@dataclass
class EvalRun:
    run_id: str
    metrics: dict[str, Any]
    artifacts: dict[str, Any]


_RUNTIME: AgentRuntime | None = None
_EVAL_RUNS: dict[str, EvalRun] = {}


def _runtime() -> AgentRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = AgentRuntime(
            config=RuntimeConfig(
                llm_provider="gemini",
                llm_model="gemini-2.5-flash",
                require_llm=True,
            )
        )
    return _RUNTIME


def create_app() -> Any:
    if FastAPI is None:
        raise RuntimeError("fastapi is not installed; install with `uv pip install fastapi uvicorn`")

    app = FastAPI(title="HiDrift API", version="0.1.0")

    @app.post("/v1/memory/ingest")
    async def memory_ingest(payload: IngestRequest) -> dict:
        rt = _runtime()
        return await rt.handle_turn(
            session_id=payload.session_id,
            user_id=payload.user_id,
            user_input=payload.user_input,
            agent_output=payload.agent_output,
            reward=payload.reward,
            task_label=payload.task_label,
        )

    @app.post("/v1/memory/retrieve")
    async def memory_retrieve(payload: RetrieveRequest) -> dict:
        rt = _runtime()
        bundle = rt.memory.retrieve(
            query=payload.query,
            k_working=payload.k_working,
            k_episodic=payload.k_episodic,
            k_semantic=payload.k_semantic,
        )
        return {
            "working": [e.model_dump() for e in bundle.working],
            "episodic": [e.model_dump() for e in bundle.episodic],
            "semantic": bundle.semantic,
            "hard_constraints": bundle.hard_constraints,
            "supporting_context": bundle.supporting_context,
        }

    @app.get("/v1/drift/current")
    async def drift_current() -> dict:
        return {"drift_signal": _runtime().current_drift()}

    @app.post("/v1/consolidation/run")
    async def consolidation_run(payload: ConsolidationRequest) -> dict:
        return await _runtime().consolidation.run_once(
            min_importance=payload.min_importance,
            decay_k=payload.decay_k,
        )

    @app.get("/v1/semantic/facts")
    async def semantic_facts(query: str = "", k: int = 5) -> dict:
        rt = _runtime()
        bundle = rt.memory.retrieve(query=query, k_semantic=k)
        return {
            "query": query,
            "hard_constraints": bundle.hard_constraints,
            "supporting_context": bundle.supporting_context,
            "semantic": bundle.semantic,
        }

    @app.get("/v1/semantic/graph/subgraph")
    async def semantic_subgraph(entity_id: str, hops: int = 2) -> dict:
        rt = _runtime()
        return rt.memory.semantic.get_subgraph(entity_id=entity_id, hops=hops)

    @app.post("/v1/semantic/facts/upsert")
    async def semantic_fact_upsert(payload: SemanticUpsertRequest) -> dict:
        rt = _runtime()
        rt.memory.semantic.upsert_fact(payload.fact)
        rt.memory.semantic.resolve_conflicts()
        return {"status": "ok", "fact_id": payload.fact.fact_id}

    @app.get("/v1/semantic/conflicts")
    async def semantic_conflicts(entity_id: str = "") -> dict:
        rt = _runtime()
        facts = rt.memory.semantic.all_facts()
        conflicts = []
        grouped: dict[tuple[str, str], list] = {}
        for fact in facts:
            grouped.setdefault((fact.subject, fact.relation), []).append(fact)
        for (subject, relation), bucket in grouped.items():
            if entity_id and subject != entity_id and all(f.object != entity_id for f in bucket):
                continue
            if len(bucket) > 1:
                conflicts.append(
                    {
                        "subject": subject,
                        "relation": relation,
                        "fact_ids": [f.fact_id for f in bucket],
                        "active_fact_ids": [f.fact_id for f in bucket if f.is_active],
                    }
                )
        return {"conflicts": conflicts}

    @app.get("/v1/eval/run/{run_id}")
    async def eval_run(run_id: str) -> dict:
        run = _EVAL_RUNS.get(run_id)
        if run is None:
            return {"error": "run_not_found", "run_id": run_id}
        return {"run_id": run.run_id, "metrics": run.metrics, "artifacts": run.artifacts}

    return app


def run_consolidation_sync() -> dict:
    return asyncio.run(_runtime().consolidation.run_once())
