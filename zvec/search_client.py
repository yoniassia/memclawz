#!/usr/bin/env python3.10
"""Simple Python client for querying the Zvec memory server."""
import json
import urllib.request

ZVEC_URL = "http://localhost:4010"


def search_with_embedding(embedding: list, topk: int = 5, url: str = ZVEC_URL) -> list[dict]:
    """Search Zvec with a pre-computed embedding vector."""
    data = json.dumps({"embedding": embedding, "topk": topk}).encode()
    req = urllib.request.Request(
        f"{url}/search", data=data,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    resp = urllib.request.urlopen(req, timeout=10)
    result = json.loads(resp.read())
    return result.get("results", [])


def search(query_text: str, topk: int = 5, url: str = ZVEC_URL) -> list[dict]:
    """Search by text — uses hash-based embedding (for real semantic search,
    integrate with the actual embedding model)."""
    from zvec.embedder import text_to_embedding
    emb = text_to_embedding(query_text)
    return search_with_embedding(emb, topk=topk, url=url)


def index_docs(docs: list[dict], url: str = ZVEC_URL) -> int:
    """Index documents. Each doc needs: id, embedding, text, path."""
    data = json.dumps({"docs": docs}).encode()
    req = urllib.request.Request(
        f"{url}/index", data=data,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    resp = urllib.request.urlopen(req, timeout=30)
    result = json.loads(resp.read())
    return result.get("indexed", 0)


def health(url: str = ZVEC_URL) -> dict:
    """Check server health."""
    resp = urllib.request.urlopen(f"{url}/health", timeout=5)
    return json.loads(resp.read())


def stats(url: str = ZVEC_URL) -> dict:
    """Get collection stats."""
    resp = urllib.request.urlopen(f"{url}/stats", timeout=5)
    return json.loads(resp.read())
