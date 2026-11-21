"""OpenAI text embeddings, the inputs for the two trained models (19th and the local baseline).

19th trains on table columns, not raw text, so each text is turned into a vector first.
Training sets are embedded in batches and cached under .tmp/embeddings/; scored items are
embedded one call at a time inside the timed endpoint call, as an application would.
"""
import hashlib
import time

import numpy as np

from bench.tasks.base import ROOT

MODEL = "text-embedding-3-small"
DIMENSIONS = 1536
PRICE_PER_MILLION = 0.02  # US dollars per million input tokens, read on 2026-11-20
CACHE = ROOT / ".tmp" / "embeddings"
BATCH_CHARS = 400_000  # well under the 300,000-token request limit
BATCH_ITEMS = 512


def embed(client, texts: list[str]) -> tuple[np.ndarray, int | None]:
    """One embeddings call. Returns a (len(texts), DIMENSIONS) float32 array and the input token count."""
    response = client.embeddings.create(model=MODEL, input=texts)
    vectors = np.array([row.embedding for row in sorted(response.data, key=lambda row: row.index)], dtype=np.float32)
    if vectors.shape != (len(texts), DIMENSIONS):
        raise ValueError(f"expected {len(texts)} embeddings of {DIMENSIONS} numbers, got {vectors.shape}")
    usage = getattr(response, "usage", None)
    return vectors, getattr(usage, "prompt_tokens", None)


def embed_timed(client, texts: list[str]) -> tuple[np.ndarray, int, int | None]:
    started = time.perf_counter()
    vectors, tokens = embed(client, texts)
    return vectors, round((time.perf_counter() - started) * 1000), tokens


def _key(text: str) -> str:
    return hashlib.sha256(f"{MODEL}:{DIMENSIONS}:{text}".encode()).hexdigest()


def batches(texts: list[str]) -> list[list[str]]:
    groups, current, size = [], [], 0
    for text in texts:
        if current and (size + len(text) > BATCH_CHARS or len(current) >= BATCH_ITEMS):
            groups.append(current)
            current, size = [], 0
        current.append(text)
        size += len(text)
    if current:
        groups.append(current)
    return groups


def embed_cached(client, texts: list[str], cache=CACHE, on_batch=None) -> tuple[np.ndarray, int]:
    """Embed every text, reusing cached vectors. Returns the vectors and the tokens paid for now."""
    cache.mkdir(parents=True, exist_ok=True)
    missing = list(dict.fromkeys(text for text in texts if not (cache / f"{_key(text)}.npy").exists()))
    paid = 0
    for group in batches(missing):
        vectors, tokens = embed(client, group)
        for text, vector in zip(group, vectors):
            np.save(cache / f"{_key(text)}.npy", vector)
        paid += tokens or 0
        if on_batch:
            on_batch(len(group))
    return np.stack([np.load(cache / f"{_key(text)}.npy") for text in texts]), paid
