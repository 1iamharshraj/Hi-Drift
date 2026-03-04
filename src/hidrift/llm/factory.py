from __future__ import annotations

import os

from hidrift.env import load_dotenv
from hidrift.llm.fallback import FallbackLLMClient
from hidrift.llm.gemini import GeminiClient, normalize_model_name
from hidrift.llm.types import LLMClient


def build_llm_client(
    provider: str | None = None,
    model_name: str | None = None,
    fail_if_unconfigured: bool = False,
) -> LLMClient:
    load_dotenv()
    provider = (provider or os.getenv("HIDRIFT_LLM_PROVIDER", "gemini")).lower()
    model_name = normalize_model_name(model_name or os.getenv("HIDRIFT_LLM_MODEL", "gemini-2.5-flash"))
    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            return GeminiClient(api_key=api_key, model_name=model_name)
        if fail_if_unconfigured:
            raise RuntimeError("GEMINI_API_KEY is required for Gemini provider.")
    return FallbackLLMClient(model_name="fallback-template")
