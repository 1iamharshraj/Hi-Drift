from __future__ import annotations

from collections import deque

from hidrift.schemas import InteractionEvent


class WorkingMemory:
    def __init__(self, maxlen: int = 20) -> None:
        self._buffer: deque[InteractionEvent] = deque(maxlen=maxlen)

    def add(self, event: InteractionEvent) -> None:
        self._buffer.append(event)

    def get_recent(self, k: int = 5) -> list[InteractionEvent]:
        if k <= 0:
            return []
        return list(self._buffer)[-k:]

    def __len__(self) -> int:
        return len(self._buffer)

