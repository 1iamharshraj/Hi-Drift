from __future__ import annotations

from datetime import datetime, timezone

from hidrift.schemas import EpisodeRecord


def _deterministic_summary(goal: str, episodes: list[EpisodeRecord]) -> str:
    rewards = [e.reward_sum for e in episodes]
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    sample_actions = []
    for ep in episodes[:3]:
        if ep.actions:
            sample_actions.append(str(ep.actions[0]))
    action_hint = " | ".join(sample_actions) if sample_actions else "no_actions"
    return (
        f"For goal '{goal}', {len(episodes)} episodes indicate average reward "
        f"{avg_reward:.3f}; representative actions: {action_hint}"
    )


def summarize_cluster(goal: str, episodes: list[EpisodeRecord], llm_client: object | None = None) -> str:
    if not episodes:
        return f"No episodes available for goal '{goal}'."
    if llm_client is None:
        return _deterministic_summary(goal, episodes)
    context_lines = []
    for ep in episodes[:8]:
        action = ep.actions[0] if ep.actions else {}
        outcome = ep.outcomes[0] if ep.outcomes else {}
        context_lines.append(
            f"- episode_id={ep.episode_id}, reward={ep.reward_sum:.3f}, action={action}, outcome={outcome}"
        )
    prompt = (
        f"Summarize stable long-term semantic memory for goal '{goal}'.\n"
        "Return exactly 1 concise factual statement useful for future retrieval.\n"
        "Evidence:\n"
        + "\n".join(context_lines)
    )
    try:
        text = llm_client.generate(
            prompt=prompt,
            system_prompt=(
                "You are a memory consolidation model. Produce a concise, non-speculative memory statement."
            ),
        )
        return text.strip()
    except Exception:
        return _deterministic_summary(goal, episodes)


def extract_semantic_facts(goal: str, episodes: list[EpisodeRecord], llm_client: object | None = None) -> list[dict]:
    statement = summarize_cluster(goal, episodes, llm_client=llm_client)
    avg_reward = sum(e.reward_sum for e in episodes) / max(len(episodes), 1)
    obj = "high_success_strategy" if avg_reward >= 0.6 else "needs_adjustment"
    relation = "RULE_FOR"
    facts = [
        {
            "statement": statement,
            "fact_type": "task_policy",
            "subject": goal,
            "relation": relation,
            "object": obj,
            "confidence": max(0.3, min(0.95, 0.55 + 0.05 * len(episodes))),
            "stability": max(0.0, min(1.0, avg_reward if avg_reward >= 0 else 0.0)),
            "valid_from": datetime.now(timezone.utc),
            "valid_to": None,
            "tags": [goal, "consolidated"],
        }
    ]
    style_counts = {"concise": 0, "detailed": 0, "bullet": 0}
    schema_counts = {"v1": 0, "v2": 0}
    for ep in episodes:
        if not ep.actions:
            continue
        user_input = str(ep.actions[0].get("user_input", "")).lower()
        for style in style_counts:
            if style in user_input:
                style_counts[style] += 1
        for version in schema_counts:
            token = f"api {version}"
            if token in user_input or f"schema {version}" in user_input:
                schema_counts[version] += 1

    dominant_style, style_votes = max(style_counts.items(), key=lambda kv: kv[1])
    if style_votes > 0:
        facts.append(
            {
                "statement": f"For {goal}, preferred response style is {dominant_style}.",
                "fact_type": "user_preference",
                "subject": goal,
                "relation": "PREFERS",
                "object": dominant_style,
                "confidence": max(0.4, min(0.95, 0.5 + 0.05 * style_votes)),
                "stability": max(0.0, min(1.0, avg_reward if avg_reward >= 0 else 0.0)),
                "valid_from": datetime.now(timezone.utc),
                "valid_to": None,
                "tags": [goal, "style", dominant_style],
            }
        )

    dominant_schema, schema_votes = max(schema_counts.items(), key=lambda kv: kv[1])
    if schema_votes > 0:
        facts.append(
            {
                "statement": f"For {goal}, active tool schema is {dominant_schema}.",
                "fact_type": "environment_state",
                "subject": goal,
                "relation": "USES_SCHEMA",
                "object": dominant_schema,
                "confidence": max(0.4, min(0.95, 0.5 + 0.05 * schema_votes)),
                "stability": max(0.0, min(1.0, avg_reward if avg_reward >= 0 else 0.0)),
                "valid_from": datetime.now(timezone.utc),
                "valid_to": None,
                "tags": [goal, "schema", dominant_schema],
            }
        )
    return facts
