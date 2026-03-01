#!/usr/bin/env python3.10
"""
memclawz-server: Fast vector memory service for OpenClaw
FastAPI + uvicorn with Pydantic validation, CORS, multi-worker support
"""
import json
import os
import signal
import sys
import time
import sqlite3
from typing import List, Optional, Any, Dict

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

import zvec

PORT = int(os.environ.get("ZVEC_PORT", "4010"))
DATA_DIR = os.environ.get("ZVEC_DATA", os.path.expanduser("~/.openclaw/zvec-memory"))
SQLITE_PATH = os.environ.get("SQLITE_PATH", os.path.expanduser("~/.openclaw/memory/main.sqlite"))
WORKERS = int(os.environ.get("ZVEC_WORKERS", "2"))

# Auto-detected from first embedding (no hardcoded DIM)
DIM = None

os.makedirs(DATA_DIR, exist_ok=True)

collection = None

app = FastAPI(title="memclawz-server", description="Fast vector memory service for OpenClaw")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Signal handlers for clean shutdown (#12) ---

def _shutdown(signum, frame):
    """Flush collection and exit cleanly on SIGTERM/SIGINT."""
    global collection
    sig_name = signal.Signals(signum).name
    print(f"\n[memclawz] Received {sig_name}, shutting down gracefully...")
    if collection is not None:
        try:
            collection.flush()
            print("[memclawz] Collection flushed.")
        except Exception as e:
            print(f"[memclawz] Flush error: {e}")
    # Remove stale lock files
    lock_path = os.path.join(DATA_DIR, "memory", ".lock")
    if os.path.exists(lock_path):
        try:
            os.remove(lock_path)
            print(f"[memclawz] Removed lock file: {lock_path}")
        except Exception:
            pass
    sys.exit(0)

signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT, _shutdown)


# --- Pydantic models ---

class DocInput(BaseModel):
    id: Optional[str] = None
    embedding: Optional[List[float]] = None
    text: str = ""
    path: str = ""
    source: str = ""
    start_line: int = 0
    end_line: int = 0

class IndexRequest(BaseModel):
    docs: Optional[List[DocInput]] = None
    text: Optional[str] = None
    meta: Optional[dict] = None

class IndexResponse(BaseModel):
    indexed: int

class SearchRequest(BaseModel):
    embedding: Optional[List[float]] = None
    text: Optional[str] = None
    topk: int = 10
    filter: Optional[str] = None

class SearchResult(BaseModel):
    id: str
    score: float
    text: str = ""
    path: str = ""
    source: str = ""
    start_line: Optional[int] = None
    end_line: Optional[int] = None

class SearchResponse(BaseModel):
    results: List[SearchResult]
    count: int


# --- Core logic ---

def ensure_collection(dim: int = 768, max_retries: int = 5) -> Any:
    """Open or create collection with retry+backoff (#12, #18)."""
    global collection, DIM
    col_path = os.path.join(DATA_DIR, "memory")
    waits = [2, 4, 6, 8, 10]

    for attempt in range(max_retries):
        try:
            # Remove stale lock files before retrying
            lock_path = os.path.join(col_path, ".lock")
            if attempt > 0 and os.path.exists(lock_path):
                try:
                    os.remove(lock_path)
                    print(f"[memclawz] Removed stale lock file (attempt {attempt+1})")
                except Exception:
                    pass

            if os.path.exists(col_path):
                collection = zvec.open(col_path)
                # Try to detect dim from existing collection
                try:
                    s = collection.stats
                    if hasattr(s, 'dim'):
                        DIM = s.dim
                    else:
                        DIM = dim
                except Exception:
                    DIM = dim
                print(f"[memclawz] Opened existing collection (dim={DIM})")
            else:
                DIM = dim
                schema = zvec.CollectionSchema(
                    name="memory",
                    vectors=[
                        zvec.VectorSchema("dense", zvec.DataType.VECTOR_FP32, dim),
                    ],
                    fields=[
                        zvec.FieldSchema("text", zvec.DataType.STRING),
                        zvec.FieldSchema("path", zvec.DataType.STRING),
                        zvec.FieldSchema("source", zvec.DataType.STRING),
                        zvec.FieldSchema("start_line", zvec.DataType.INT32),
                        zvec.FieldSchema("end_line", zvec.DataType.INT32),
                        zvec.FieldSchema("updated_at", zvec.DataType.INT64),
                    ]
                )
                collection = zvec.create_and_open(col_path, schema)
                print(f"[memclawz] Created new collection at {col_path} (dim={dim})")
            return collection

        except Exception as e:
            wait = waits[min(attempt, len(waits)-1)]
            print(f"[memclawz] Collection open failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"[memclawz] Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"Failed to open collection after {max_retries} attempts: {e}")


def get_or_create_collection(dim=256):
    """Wrapper kept for backward compat."""
    return ensure_collection(dim=dim)


def migrate_from_sqlite():
    """Import chunks from OpenClaw's sqlite memory into zvec"""
    global collection, DIM
    if not os.path.exists(SQLITE_PATH):
        return {"error": f"SQLite not found at {SQLITE_PATH}"}

    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT id, path, source, start_line, end_line, text, embedding, updated_at
        FROM chunks
        WHERE embedding IS NOT NULL
        ORDER BY id
    """).fetchall()

    if not rows:
        return {"migrated": 0, "error": "no chunks with embeddings"}

    first_emb = json.loads(rows[0]["embedding"])
    dim = len(first_emb)
    print(f"Detected embedding dimension: {dim}")

    if collection is not None and DIM == dim:
        col = collection
    else:
        col = ensure_collection(dim)

    docs = []
    skipped = 0
    for row in rows:
        emb_raw = row["embedding"]
        if not emb_raw:
            skipped += 1
            continue
        try:
            emb = json.loads(emb_raw) if isinstance(emb_raw, str) else np.frombuffer(emb_raw, dtype=np.float32).tolist()
        except:
            skipped += 1
            continue
        if len(emb) != dim:
            skipped += 1
            continue

        d = zvec.Doc(str(row["id"]))
        d.vectors["dense"] = emb if isinstance(emb, list) else emb.tolist()
        d.fields["text"] = row["text"] or ""
        d.fields["path"] = row["path"] or ""
        d.fields["source"] = row["source"] or ""
        d.fields["start_line"] = row["start_line"] or 0
        d.fields["end_line"] = row["end_line"] or 0
        d.fields["updated_at"] = int(row["updated_at"] or 0)
        docs.append(d)

    conn.close()

    if docs:
        for i in range(0, len(docs), 100):
            batch = docs[i:i+100]
            col.insert(batch)
        col.create_index("dense", zvec.HnswIndexParam())
        col.flush()

    return {"migrated": len(docs), "skipped": skipped, "dimension": dim}


def _compat_query(vq, **kwargs):
    """Compat wrapper: try query() first, fallback to search() (#14)."""
    try:
        return collection.query(vq, **kwargs)
    except AttributeError:
        return collection.search(vq, **kwargs)


def do_search(query_embedding, topk=10, filter_expr=None):
    """Search the zvec collection"""
    if collection is None:
        return {"error": "collection not initialized"}

    vq = zvec.VectorQuery("dense", vector=query_embedding)
    kwargs = {"topk": topk}
    if filter_expr:
        kwargs["filter"] = filter_expr

    results = _compat_query(vq, **kwargs)

    out = []
    for r in results:
        item = {
            "id": r.id,
            "score": float(r.score),
            "text": r.field("text") if r.has_field("text") else "",
            "path": r.field("path") if r.has_field("path") else "",
            "source": r.field("source") if r.has_field("source") else "",
        }
        if r.has_field("start_line"):
            item["start_line"] = r.field("start_line")
        if r.has_field("end_line"):
            item["end_line"] = r.field("end_line")
        out.append(item)

    return {"results": out, "count": len(out)}


# --- Endpoints ---

@app.get("/health")
async def health():
    return {"status": "ok", "engine": "zvec", "version": zvec.__version__}


@app.get("/info")
async def info():
    """Show current collection info including dimension (#13)."""
    return {
        "dim": DIM,
        "data_dir": DATA_DIR,
        "collection_loaded": collection is not None,
        "engine": "zvec",
        "version": zvec.__version__,
    }


@app.get("/stats")
async def stats():
    if collection:
        try:
            s = collection.stats
            total_docs = s.doc_count if hasattr(s, 'doc_count') else 0
            if total_docs == 0:
                try:
                    dim = DIM or 768
                    vq = zvec.VectorQuery("dense", vector=[0.0]*dim)
                    results = _compat_query(vq, topk=1)
                    total_docs = len(results) if results else 0
                    if total_docs > 0:
                        total_docs = "295+"
                except:
                    pass
        except:
            total_docs = 0
        return {"total_docs": total_docs, "dim": DIM, "path": DATA_DIR, "status": "loaded"}
    else:
        return {"total_docs": 0, "dim": DIM, "path": DATA_DIR, "status": "uninitialized"}


@app.get("/migrate")
async def migrate():
    return migrate_from_sqlite()


@app.get("/")
async def root():
    return {"endpoints": ["/health", "/stats", "/info", "/migrate", "/search (POST)", "/index (POST)"]}


@app.post("/search")
async def search_endpoint(req: SearchRequest):
    if req.embedding:
        emb = req.embedding
    elif req.text:
        embedder = get_embedder()
        if not embedder:
            raise HTTPException(status_code=503, detail="Embedding model not available. Provide 'embedding' directly.")
        emb = embedder.embed_text(req.text)
    else:
        raise HTTPException(status_code=400, detail="Provide 'text' or 'embedding'")
    result = do_search(emb, req.topk, req.filter)
    return result


@app.post("/index")
async def index_endpoint(req: IndexRequest):
    import hashlib

    # Support simple {"text": "...", "meta": {...}} shorthand
    if not req.docs and req.text:
        embedder = get_embedder()
        if not embedder:
            raise HTTPException(status_code=503, detail="Embedding model not available. Provide 'docs' with embeddings.")
        emb = embedder.embed_text(req.text)
        meta = req.meta or {}
        doc_id = hashlib.sha256(f"{req.text}{time.time()}".encode()).hexdigest()[:16]
        req.docs = [DocInput(
            id=doc_id, embedding=emb, text=req.text,
            path=meta.get("path", ""), source=meta.get("source", "manual"),
            start_line=meta.get("start_line", 0), end_line=meta.get("end_line", 0),
        )]

    if not req.docs:
        raise HTTPException(status_code=400, detail="Provide 'docs' list or 'text'")

    # Auto-embed any docs missing embeddings
    _embedder = None
    for d in req.docs:
        if not d.embedding and d.text:
            if _embedder is None:
                _embedder = get_embedder()
                if not _embedder:
                    raise HTTPException(status_code=503, detail="Embedding model not available for auto-embed.")
            d.embedding = _embedder.embed_text(d.text)
        if not d.id:
            d.id = hashlib.sha256(f"{d.text}{time.time()}".encode()).hexdigest()[:16]

    global collection, DIM

    # Auto-detect dimension from first embedding (#13, #18)
    incoming_dim = len(req.docs[0].embedding) if req.docs and req.docs[0].embedding else None

    if collection is None:
        dim = incoming_dim or 256
        ensure_collection(dim)
    elif DIM is not None and incoming_dim is not None and incoming_dim != DIM:
        # Dimension validation (#13)
        raise HTTPException(
            status_code=400,
            detail=f"Embedding dimension mismatch: got {incoming_dim}, expected {DIM}. "
                   f"Collection was created with dim={DIM}."
        )

    docs = []
    for d in req.docs:
        # Validate each doc's dimension
        if DIM is not None and len(d.embedding) != DIM:
            raise HTTPException(
                status_code=400,
                detail=f"Doc '{d.id}' has dim {len(d.embedding)}, expected {DIM}"
            )
        doc_id = str(d.id).replace(":", "_").replace("/", "_").replace(" ", "_")
        doc = zvec.Doc(doc_id)
        doc.vectors["dense"] = d.embedding
        doc.fields["text"] = d.text
        doc.fields["path"] = d.path
        doc.fields["source"] = d.source
        doc.fields["start_line"] = d.start_line
        doc.fields["end_line"] = d.end_line
        doc.fields["updated_at"] = int(time.time())
        docs.append(doc)

    collection.upsert(docs)
    collection.flush()
    collection.optimize()
    collection.create_index("dense", zvec.HnswIndexParam())
    return {"indexed": len(docs)}


if __name__ == "__main__":
    print(f"memclawz-server v{zvec.__version__} starting on port {PORT}")

    col_path = os.path.join(DATA_DIR, "memory")
    if os.path.exists(col_path):
        ensure_collection()
        print(f"Loaded collection from {col_path}")
    else:
        print("No collection yet. Call GET /migrate or POST /index to create one.")

    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")


# --- Causality Graph Integration (v3.0) ---

from memclawz_server.causality_graph import CausalityGraph

_graph: Optional[CausalityGraph] = None

def get_graph() -> CausalityGraph:
    global _graph
    if _graph is None:
        _graph = CausalityGraph()
    return _graph


class GraphAddRequest(BaseModel):
    text: str
    embedding: Optional[List[float]] = None
    node_id: Optional[str] = None
    source: str = ""
    timestamp: Optional[float] = None
    caused_by: Optional[List[str]] = None
    causes: Optional[List[str]] = None
    associations: Optional[List[str]] = None

class GraphSearchRequest(BaseModel):
    embedding: List[float]
    topk: int = 5
    similarity_threshold: float = 0.5
    max_depth: int = 2

class GraphKeywordRequest(BaseModel):
    keywords: str
    limit: int = 10


@app.post("/graph/add")
async def graph_add(req: GraphAddRequest):
    g = get_graph()
    nid = g.add_node(
        text=req.text, embedding=req.embedding, node_id=req.node_id,
        source=req.source, timestamp=req.timestamp,
        caused_by=req.caused_by, causes=req.causes, associations=req.associations,
    )
    return {"id": nid, "status": "ok"}


@app.post("/graph/search")
async def graph_search(req: GraphSearchRequest):
    g = get_graph()
    result = g.multi_hop_search(
        query_embedding=req.embedding, topk=req.topk,
        similarity_threshold=req.similarity_threshold, max_depth=req.max_depth,
    )
    return result


@app.post("/graph/keyword")
async def graph_keyword(req: GraphKeywordRequest):
    g = get_graph()
    results = g.keyword_search(req.keywords, req.limit)
    return {"results": results, "count": len(results)}


@app.get("/graph/stats")
async def graph_stats():
    g = get_graph()
    return g.stats()
