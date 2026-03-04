from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(dotenv_path: str | Path | None = None) -> None:
    """
    Minimal .env loader to avoid extra dependency.
    Only sets keys that are not already present in process env.
    """
    if dotenv_path is None:
        dotenv_path = Path.cwd() / ".env"
    path = Path(dotenv_path)
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

