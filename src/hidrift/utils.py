from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _chunk_hash(text: str, idx: int) -> float:
    h = hashlib.sha256(f"{idx}:{text}".encode("utf-8")).digest()
    value = int.from_bytes(h[:8], "big") / (2**64 - 1)
    return (value * 2.0) - 1.0


def embed_text(text: str, dim: int = 64) -> list[float]:
    vec = [_chunk_hash(text, i) for i in range(dim)]
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def l2_distance(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 1.0
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def kl_divergence(p: dict[str, float], q: dict[str, float], eps: float = 1e-8) -> float:
    keys = set(p.keys()) | set(q.keys())
    kl = 0.0
    for key in keys:
        pi = max(p.get(key, 0.0), eps)
        qi = max(q.get(key, 0.0), eps)
        kl += pi * math.log(pi / qi)
    return kl


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)

