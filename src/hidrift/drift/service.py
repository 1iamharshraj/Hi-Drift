from __future__ import annotations

from dataclasses import dataclass

from hidrift.drift.features import OnlineDriftState
from hidrift.drift.scoring import DriftScorer
from hidrift.schemas import DriftSignal, InteractionEvent
from hidrift.utils import embed_text, utc_now


@dataclass
class TriggerConfig:
    threshold: float = 0.35
    hysteresis_turns: int = 3
    cooldown_turns: int = 2


class DriftService:
    def __init__(self, config: TriggerConfig | None = None) -> None:
        self.config = config or TriggerConfig()
        self.state = OnlineDriftState()
        self.scorer = DriftScorer()
        self._consecutive_above = 0
        self._cooldown = 0
        self._last_signal: DriftSignal | None = None

    def process(self, event: InteractionEvent) -> DriftSignal:
        emb = embed_text(f"{event.user_input}\n{event.agent_output}")
        behavioral_shift = self.state.update_embedding(emb)
        _, _, task_shift = self.state.update_task_distribution(event.task_label)
        performance_drop = self.state.update_performance(event.reward)
        total = self.scorer.score(behavioral_shift, task_shift, performance_drop)

        if self._cooldown > 0:
            self._cooldown -= 1
            triggered = False
            self._consecutive_above = 0
        else:
            if total > self.config.threshold:
                self._consecutive_above += 1
            else:
                self._consecutive_above = 0
            triggered = self._consecutive_above >= self.config.hysteresis_turns
            if triggered:
                self._cooldown = self.config.cooldown_turns
                self._consecutive_above = 0

        signal = DriftSignal(
            ts=utc_now(),
            behavioral_shift=behavioral_shift,
            task_shift=task_shift,
            performance_drop=performance_drop,
            total_score=total,
            threshold=self.config.threshold,
            triggered=triggered,
        )
        self._last_signal = signal
        return signal

    def current(self) -> DriftSignal | None:
        return self._last_signal

