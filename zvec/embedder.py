#!/usr/bin/env python3.10
"""
Embedder module for QMDZvec.
Extracts embeddings from OpenClaw's SQLite memory DB (which already has them
computed by node-llama-cpp with embeddinggemma-300m, 768-dim).
For new text without pre-computed embeddings, uses the SQLite DB as a lookup
or falls back to a simple TF-IDF approach for testing.
"""
import json
import os
import sqlite3
import numpy as np

SQLITE_PATH = os.environ.get("SQLITE_PATH", os.path.expanduser("~/.openclaw/memory/main.sqlite"))
DIM = 768  # embeddinggemma-300m


def get_embedding_from_sqlite(text: str) -> list | None:
    """Look up an embedding for exact text match in SQLite."""
    if not os.path.exists(SQLITE_PATH):
        return None
    conn = sqlite3.connect(SQLITE_PATH)
    row = conn.execute(
        "SELECT embedding FROM chunks WHERE text = ? AND embedding IS NOT NULL LIMIT 1",
        (text,)
    ).fetchone()
    conn.close()
    if row and row[0]:
        return json.loads(row[0]) if isinstance(row[0], str) else list(row[0])
    return None


def get_all_embeddings(limit=None):
    """Yield (id, text, path, embedding) tuples from SQLite."""
    if not os.path.exists(SQLITE_PATH):
        return
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    q = "SELECT id, text, path, embedding FROM chunks WHERE embedding IS NOT NULL ORDER BY id"
    if limit:
        q += f" LIMIT {int(limit)}"
    for row in conn.execute(q):
        emb = json.loads(row["embedding"]) if isinstance(row["embedding"], str) else list(row["embedding"])
        yield row["id"], row["text"], row["path"], emb
    conn.close()


def random_embedding(dim=DIM) -> list:
    """Generate a random unit embedding (for testing only)."""
    v = np.random.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-9
    return v.tolist()


def text_to_embedding(text: str, dim=DIM) -> list:
    """
    Best-effort embedding for arbitrary text.
    First tries SQLite lookup; falls back to deterministic hash-based embedding.
    """
    emb = get_embedding_from_sqlite(text)
    if emb:
        return emb
    # Deterministic hash-based embedding (for testing/demo - not semantic)
    import hashlib
    h = hashlib.sha256(text.encode()).digest()
    np.random.seed(int.from_bytes(h[:4], 'big'))
    v = np.random.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-9
    return v.tolist()
