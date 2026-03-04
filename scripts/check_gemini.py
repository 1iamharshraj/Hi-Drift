from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hidrift.env import load_dotenv
from hidrift.llm.factory import build_llm_client


def main() -> None:
    load_dotenv(ROOT / ".env")
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not set.")
    client = build_llm_client(provider="gemini", model_name="gemini-2.5-flash", fail_if_unconfigured=True)
    response = client.generate(
        prompt="Reply with exactly: GEMINI_OK",
        system_prompt="You are a strict validator.",
    )
    print(response)


if __name__ == "__main__":
    main()
