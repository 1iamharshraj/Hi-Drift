from __future__ import annotations

import asyncio

from hidrift.agent.runtime import AgentRuntime


def test_end_to_end_ingest_retrieve_act_loop() -> None:
    runtime = AgentRuntime()
    result = asyncio.run(
        runtime.handle_turn(
            session_id="s-1",
            user_id="u-1",
            user_input="Please schedule my meeting tomorrow.",
            agent_output="Meeting scheduled for tomorrow at 10:00.",
            reward=1.0,
            task_label="calendar",
        )
    )
    assert result["episode_id"]
    assert result["retrieval"]["working_count"] >= 1
    assert result["retrieval"]["episodic_count"] >= 1


def test_drift_trigger_can_invoke_consolidation() -> None:
    runtime = AgentRuntime()
    # Force low threshold by mutating config object for deterministic integration behavior.
    runtime.drift.config.threshold = 0.0
    for i in range(4):
        asyncio.run(
            runtime.handle_turn(
                session_id="s-2",
                user_id="u-1",
                user_input=f"Turn {i} with changing task",
                agent_output=f"Output {i}",
                reward=0.2,
                task_label=f"task_{i}",
            )
        )
    assert len(runtime.memory.semantic) >= 1

