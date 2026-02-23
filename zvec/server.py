#!/usr/bin/env python3.10
"""
zvec-memory: Fast vector memory service for OpenClaw
Replaces sqlite-vec with Alibaba Zvec (HNSW) + BM25 hybrid search
"""
import json
import os
import sys
import time
import sqlite3
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import zvec

PORT = int(os.environ.get("ZVEC_PORT", "4010"))
DATA_DIR = os.environ.get("ZVEC_DATA", os.path.expanduser("~/.openclaw/zvec-memory"))
SQLITE_PATH = os.environ.get("SQLITE_PATH", os.path.expanduser("~/.openclaw/memory/main.sqlite"))
DIM = 256  # embeddinggemma-300m output dim — will detect from data

os.makedirs(DATA_DIR, exist_ok=True)

collection = None

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
        print(f"Opened existing collection: {collection.stats()}")
    else:
        collection = zvec.create_and_open(col_path, schema)
        print(f"Created new collection at {col_path}")
    
    return collection


def migrate_from_sqlite():
    """Import chunks from OpenClaw's sqlite memory into zvec"""
    global DIM
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
    
    # Detect embedding dimension from first row
    first_emb = json.loads(rows[0]["embedding"])
    dim = len(first_emb)
    print(f"Detected embedding dimension: {dim}")
    
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
        # Insert in batches of 100
        for i in range(0, len(docs), 100):
            batch = docs[i:i+100]
            col.insert(batch)
        col.create_index("dense", zvec.HnswIndexParam())
        col.flush()
    
    return {"migrated": len(docs), "skipped": skipped, "dimension": dim}


def search(query_embedding, topk=10, filter_expr=None):
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


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Silence default logging
    
    def _json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def do_GET(self):
        url = urlparse(self.path)
        
        if url.path == "/health":
            self._json({"status": "ok", "engine": "zvec", "version": zvec.__version__})
        
        elif url.path == "/stats":
            if collection:
                self._json({"dim": DIM, "path": DATA_DIR, "status": "loaded"})
            else:
                self._json({"error": "no collection"}, 500)
        
        elif url.path == "/migrate":
            result = migrate_from_sqlite()
            self._json(result)
        
        else:
            self._json({"endpoints": ["/health", "/stats", "/migrate", "/search (POST)"]})
    
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        url = urlparse(self.path)
        
        if url.path == "/search":
            embedding = body.get("embedding", [])
            topk = body.get("topk", 10)
            filter_expr = body.get("filter")
            if not embedding:
                self._json({"error": "missing embedding"}, 400)
                return
            result = search(embedding, topk, filter_expr)
            self._json(result)
        
        elif url.path == "/index":
            docs_data = body.get("docs", [])
            if not docs_data:
                self._json({"error": "missing docs"}, 400)
                return
            docs = []
            for d in docs_data:
                doc = zvec.Doc(str(d["id"]))
                doc.vectors["dense"] = d["embedding"]
                doc.fields["text"] = d.get("text", "")
                doc.fields["path"] = d.get("path", "")
                doc.fields["source"] = d.get("source", "")
                doc.fields["start_line"] = d.get("start_line", 0)
                doc.fields["end_line"] = d.get("end_line", 0)
                doc.fields["updated_at"] = int(time.time())
                docs.append(doc)
            collection.upsert(docs)
            collection.flush()
            self._json({"indexed": len(docs)})
        
        else:
            self._json({"error": "unknown endpoint"}, 404)


if __name__ == "__main__":
    print(f"zvec-memory v{zvec.__version__} starting on port {PORT}")
    
    # Try to open existing collection or wait for migration
    col_path = os.path.join(DATA_DIR, "memory")
    if os.path.exists(col_path):
        collection = zvec.open(col_path)
        print(f"Loaded collection from {col_path}")
    else:
        print("No collection yet. Call GET /migrate to import from SQLite.")
    
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Listening on http://127.0.0.1:{PORT}")
    server.serve_forever()
