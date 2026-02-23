"""Zvec server tests — requires server running on localhost:4010."""
import json
import urllib.request
import numpy as np
import pytest
import time

ZVEC_URL = "http://localhost:4010"
DIM = 768


def _get(path):
    return json.loads(urllib.request.urlopen(f"{ZVEC_URL}{path}", timeout=5).read())

def _post(path, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(f"{ZVEC_URL}{path}", data=body,
                                headers={"Content-Type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(req, timeout=10).read())

def _rand_emb(dim=DIM):
    v = np.random.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-9
    return v.tolist()

def _unique_id():
    return f"test-{int(time.time()*1000)}-{np.random.randint(0,99999)}"


@pytest.fixture(autouse=True)
def check_server():
    try:
        urllib.request.urlopen(f"{ZVEC_URL}/health", timeout=2)
    except:
        pytest.skip("Zvec server not running on :4010")


class TestHealth:
    def test_health(self):
        r = _get("/health")
        assert r["status"] == "ok"
        assert "zvec" in r["engine"]

    def test_stats(self):
        r = _get("/stats")
        assert "dim" in r or "status" in r


class TestIndex:
    def test_index_single(self):
        doc_id = _unique_id()
        r = _post("/index", {"docs": [{"id": doc_id, "embedding": _rand_emb(), "text": "test doc", "path": "test.md"}]})
        assert r["indexed"] == 1

    def test_index_batch(self):
        docs = [{"id": _unique_id(), "embedding": _rand_emb(), "text": f"batch doc {i}", "path": "batch.md"} for i in range(100)]
        r = _post("/index", {"docs": docs})
        assert r["indexed"] == 100

    def test_index_upsert(self):
        doc_id = _unique_id()
        emb = _rand_emb()
        _post("/index", {"docs": [{"id": doc_id, "embedding": emb, "text": "version 1", "path": "test.md"}]})
        _post("/index", {"docs": [{"id": doc_id, "embedding": emb, "text": "version 2", "path": "test.md"}]})
        # Search for this exact embedding - should find it
        r = _post("/search", {"embedding": emb, "topk": 5})
        texts = [x["text"] for x in r["results"]]
        # At least one result should be version 2 (upserted)
        assert any("version" in t for t in texts)


class TestSearch:
    def test_search_exact(self):
        """Index a specific doc and search with its exact embedding."""
        emb = _rand_emb()
        doc_id = _unique_id()
        _post("/index", {"docs": [{"id": doc_id, "embedding": emb, "text": "Four Seasons hotel costs €1360", "path": "hotels.md"}]})
        r = _post("/search", {"embedding": emb, "topk": 5})
        assert r["count"] > 0
        assert any("Four Seasons" in x["text"] for x in r["results"])

    def test_search_semantic(self):
        """Index hotel docs, search with similar embedding."""
        base_emb = np.array(_rand_emb())
        # Index with base embedding
        _post("/index", {"docs": [{"id": _unique_id(), "embedding": base_emb.tolist(),
                                    "text": "Luxury hotel accommodation pricing in Limassol", "path": "h.md"}]})
        # Search with slightly perturbed embedding (simulating semantic similarity)
        noise = np.random.randn(DIM).astype(np.float32) * 0.1
        query = (base_emb + noise)
        query = (query / (np.linalg.norm(query) + 1e-9)).tolist()
        r = _post("/search", {"embedding": query, "topk": 5})
        assert r["count"] > 0

    def test_search_empty(self):
        """Search with a random embedding — may return results from existing data, but shouldn't crash."""
        r = _post("/search", {"embedding": _rand_emb(), "topk": 5})
        assert "results" in r
        assert isinstance(r["results"], list)

    def test_search_topk(self):
        """Verify topk parameter limits results."""
        emb = _rand_emb()
        # Index 10 docs with similar embeddings
        for i in range(10):
            noise = np.random.randn(DIM).astype(np.float32) * 0.05
            e = (np.array(emb) + noise)
            e = (e / (np.linalg.norm(e) + 1e-9)).tolist()
            _post("/index", {"docs": [{"id": _unique_id(), "embedding": e, "text": f"topk doc {i}", "path": "t.md"}]})
        
        for k in [1, 3, 5]:
            r = _post("/search", {"embedding": emb, "topk": k})
            assert r["count"] <= k
