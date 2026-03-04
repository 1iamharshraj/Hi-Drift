from __future__ import annotations

import os
from dataclasses import dataclass


def normalize_model_name(name: str) -> str:
    # Accept common typo form "gemini-2,5-flash".
    cleaned = name.strip().replace(",", ".")
    if cleaned.lower() == "gemini-2.5-flash":
        return "gemini-2.5-flash"
    return cleaned


@dataclass
class GeminiClient:
    api_key: str
    model_name: str = "gemini-2.5-flash"

    def __post_init__(self) -> None:
        self.model_name = normalize_model_name(self.model_name)
        try:
            from google import genai  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "google-genai is not installed. Install with `uv pip install google-genai`."
            ) from exc
        self._genai = genai
        self._client = genai.Client(api_key=self.api_key)

    @classmethod
    def from_env(cls, model_name: str = "gemini-2.5-flash") -> "GeminiClient":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        return cls(api_key=api_key, model_name=model_name)

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        full_prompt = prompt if not system_prompt else f"System: {system_prompt}\n\nUser: {prompt}"
        resp = self._client.models.generate_content(
            model=self.model_name,
            contents=full_prompt,
        )
        text = getattr(resp, "text", None)
        if text:
            return text.strip()
        return str(resp).strip()

