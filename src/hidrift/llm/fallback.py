from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FallbackLLMClient:
    model_name: str = "fallback-template"

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        head = prompt.strip().replace("\n", " ")
        if len(head) > 220:
            head = head[:220] + "..."
        if system_prompt:
            return f"[{self.model_name}] {head}"
        return f"[{self.model_name}] {head}"

