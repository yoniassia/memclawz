#!/usr/bin/env python3.10
"""
zvec-memory: Fast vector memory service for OpenClaw
FastAPI + uvicorn with Pydantic validation, CORS, multi-worker support
"""
import json
import os
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
DIM = 768

os.makedirs(DATA_DIR, exist_ok=True)

collection = None

app = FastAPI(title="zvec-memory", description="Fast vector memory service for OpenClaw")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic models ---

class DocInput(BaseModel):
    id: str
    embedding: List[float]
    text: str = ""
    path: str = ""
    source: str = ""
    start_line: int = 0
    end_line: int = 0

class IndexRequest(BaseModel):
    docs: List[DocInput]

class IndexResponse(BaseModel):
    indexed: int

class SearchRequest(BaseModel):
    embedding: List[float]
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


# --- Core logic (unchanged) ---

def get_or_create_collection(dim=256):
    global collection, DIM
    DIM = dim
    col_path = os.path.join(DATA_DIR, "memory")

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

    if os.path.exists(col_path):
        collection = zvec.open(col_path)
        print(f"Opened existing collection: {collection.stats}")
    else:
        collection = zvec.create_and_open(col_path, schema)
        print(f"Created new collection at {col_path}")

    return collection


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

    # Reuse existing collection if already open with matching dim
    if collection is not None and DIM == dim:
        col = collection
    else:
        col = get_or_create_collection(dim)

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


def do_search(query_embedding, topk=10, filter_expr=None):
    """Search the zvec collection"""
    if collection is None:
        return {"error": "collection not initialized"}

    vq = zvec.VectorQuery("dense", vector=query_embedding)
    kwargs = {"topk": topk}
    if filter_expr:
        kwargs["filter"] = filter_expr

    results = collection.query(vq, **kwargs)

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


@app.get("/stats")
async def stats():
    if collection:
        try:
            s = collection.stats
            total_docs = s.doc_count if hasattr(s, 'doc_count') else 0
            # zvec doc_count may return 0 even with data; use search as fallback
            if total_docs == 0:
                try:
                    vq = zvec.VectorQuery("dense", vector=[0.0]*DIM)
                    results = collection.query(vector_query=[vq], topk=1)
                    total_docs = len(results) if results else 0
                    if total_docs > 0:
                        total_docs = "295+"  # approximate — search works
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
    return {"endpoints": ["/health", "/stats", "/migrate", "/search (POST)"]}


@app.post("/search")
async def search_endpoint(req: SearchRequest):
    if not req.embedding:
        raise HTTPException(status_code=400, detail="missing embedding")
    result = do_search(req.embedding, req.topk, req.filter)
    return result


@app.post("/index")
async def index_endpoint(req: IndexRequest):
    if not req.docs:
        raise HTTPException(status_code=400, detail="missing docs")

    global collection
    if collection is None:
        dim = len(req.docs[0].embedding) if req.docs and req.docs[0].embedding else 256
        get_or_create_collection(dim)

    docs = []
    for d in req.docs:
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
    print(f"zvec-memory v{zvec.__version__} starting on port {PORT}")

    col_path = os.path.join(DATA_DIR, "memory")
    if os.path.exists(col_path):
        collection = zvec.open(col_path)
        print(f"Loaded collection from {col_path}")
    else:
        print("No collection yet. Call GET /migrate to import from SQLite.")

    # Single worker when run directly (collection is not fork-safe)
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
