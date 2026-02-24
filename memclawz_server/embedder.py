#!/usr/bin/env python3
"""
Embedder module for memclawz — 100% LOCAL, zero external APIs.
Uses sentence-transformers (all-mpnet-base-v2, 768-dim).
"""
import json
import os
import sqlite3
from typing import Optional, List

import numpy as np

SQLITE_PATH = os.environ.get("SQLITE_PATH", os.path.expanduser("~/.openclaw/memory/main.sqlite"))
DIM = 768

# Lazy-loaded local model
_st_model = None
LOCAL_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "all-mpnet-base-v2")


def _get_local_model():
    """Load sentence-transformers model (singleton, lazy). 100% local after first download."""
    global _st_model
    if _st_model is not None:
        return _st_model
    try:
        from sentence_transformers import SentenceTransformer
        print(f"Loading local embedding model: {LOCAL_MODEL_NAME}")
        _st_model = SentenceTransformer(LOCAL_MODEL_NAME)
        print(f"✅ Local embedding model loaded ({LOCAL_MODEL_NAME}, dim={_st_model.get_sentence_embedding_dimension()})")
        return _st_model
    except Exception as e:
        print(f"⚠️  Failed to load local model: {e}")
        return None


def embed_local(text: str) -> Optional[list]:
    """Generate embedding using local sentence-transformers model."""
    model = _get_local_model()
    if model is None:
        return None
    try:
        emb = model.encode(text, normalize_embeddings=True)
        return emb.tolist()
    except Exception as e:
        print(f"⚠️  Local embed failed: {e}")
        return None


def get_embedding_from_sqlite(text: str) -> Optional[list]:
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


def text_to_embedding(text: str, dim: int = DIM) -> list:
    """Generate embedding — 100% local. SQLite cache → local model → hash fallback."""
    emb = get_embedding_from_sqlite(text)
    if emb:
        return emb
    emb = embed_local(text)
    if emb:
        return emb
    # Last resort
    import hashlib
    h = hashlib.sha256(text.encode()).digest()
    np.random.seed(int.from_bytes(h[:4], 'big'))
    v = np.random.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-9
    return v.tolist()
