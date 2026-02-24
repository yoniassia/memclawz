#!/usr/bin/env python3.10
"""
Fleet Memory Server — Multi-tenant Zvec for OpenClaw agent fleets.

Each agent gets its own namespace (collection). Supports:
- Namespaced index/search (per-agent or fleet-wide)
- Simple API key authentication
- Namespace listing and stats

Usage:
    python3.10 fleet_server.py --port 4011 --data ~/.openclaw/fleet-memory
    python3.10 fleet_server.py --port 4011 --api-key my-secret-key
"""
import argparse
import json
import os
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

import zvec

DIM = 768
DEFAULT_PORT = 4011
DEFAULT_DATA = os.path.expanduser("~/.openclaw/fleet-memory")


class FleetMemory:
    """Manages multiple namespaced Zvec collections."""

    def __init__(self, data_dir: str, dim: int = DIM):
        self.data_dir = data_dir
        self.dim = dim
        self.collections: dict[str, zvec.Collection] = {}
        os.makedirs(data_dir, exist_ok=True)
        self._load_existing()

    def _load_existing(self):
        """Load all existing namespace collections."""
        for name in os.listdir(self.data_dir):
            path = os.path.join(self.data_dir, name)
            if os.path.isdir(path):
                try:
                    self.collections[name] = zvec.open(path)
                    print(f"  Loaded namespace: {name}")
                except Exception as e:
                    print(f"  Warning: Failed to load {name}: {e}", file=sys.stderr)

    def _create_collection(self, namespace: str) -> zvec.Collection:
        """Create a new namespace collection."""
        path = os.path.join(self.data_dir, namespace)
        schema = zvec.CollectionSchema(
            name=namespace,
            vectors=[zvec.VectorSchema("dense", zvec.DataType.VECTOR_FP32, self.dim)],
            fields=[
                zvec.FieldSchema("text", zvec.DataType.STRING),
                zvec.FieldSchema("path", zvec.DataType.STRING),
                zvec.FieldSchema("source", zvec.DataType.STRING),
                zvec.FieldSchema("agent", zvec.DataType.STRING),
                zvec.FieldSchema("start_line", zvec.DataType.INT32),
                zvec.FieldSchema("end_line", zvec.DataType.INT32),
                zvec.FieldSchema("updated_at", zvec.DataType.INT64),
                zvec.FieldSchema("shared", zvec.DataType.INT32),  # 1=shared, 0=private
            ]
        )
        col = zvec.create_and_open(path, schema)
        self.collections[namespace] = col
        print(f"  Created namespace: {namespace}")
        return col

    def get(self, namespace: str) -> zvec.Collection:
        """Get or create a namespace collection."""
        if namespace not in self.collections:
            return self._create_collection(namespace)
        return self.collections[namespace]

    def namespaces(self) -> list[dict]:
        """List all namespaces with stats."""
        result = []
        for name, col in self.collections.items():
            result.append({"namespace": name, "stats": str(col.stats())})
        return result

    def index(self, namespace: str, docs_data: list[dict]) -> dict:
        """Index documents into a namespace."""
        col = self.get(namespace)
        docs = []
        for d in docs_data:
            doc = zvec.Doc(str(d["id"]))
            doc.vectors["dense"] = d["embedding"]
            doc.fields["text"] = d.get("text", "")
            doc.fields["path"] = d.get("path", "")
            doc.fields["source"] = d.get("source", "")
            doc.fields["agent"] = namespace
            doc.fields["start_line"] = d.get("start_line", 0)
            doc.fields["end_line"] = d.get("end_line", 0)
            doc.fields["updated_at"] = int(time.time())
            doc.fields["shared"] = 1 if d.get("shared", True) else 0
            docs.append(doc)
        col.upsert(docs)
        col.flush()
        col.optimize()
        col.create_index("dense", zvec.HnswIndexParam())
        return {"indexed": len(docs), "namespace": namespace}

    def search(self, embedding: list[float], topk: int = 10,
               namespace: str = "all", shared_only: bool = False) -> dict:
        """Search one namespace or all namespaces."""
        results = []

        targets = (
            list(self.collections.keys()) if namespace == "all"
            else [namespace]
        )

        for ns in targets:
            if ns not in self.collections:
                continue
            col = self.collections[ns]
            try:
                vq = zvec.VectorQuery("dense", vector=embedding)
                hits = col.query(vq, topk=topk)
                for r in hits:
                    item = {
                        "id": r.id,
                        "score": float(r.score),
                        "namespace": ns,
                        "text": r.field("text") if r.has_field("text") else "",
                        "path": r.field("path") if r.has_field("path") else "",
                        "agent": r.field("agent") if r.has_field("agent") else ns,
                    }
                    if shared_only and r.has_field("shared") and r.field("shared") == 0:
                        continue
                    results.append(item)
            except Exception as e:
                print(f"  Search error in {ns}: {e}", file=sys.stderr)

        # Sort by score descending, take top-k
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:topk]

        return {"results": results, "count": len(results), "namespaces_searched": targets}


class FleetHandler(BaseHTTPRequestHandler):
    fleet: FleetMemory = None  # Set by main()
    api_key: str = None

    def log_message(self, fmt, *args):
        pass

    def _check_auth(self) -> bool:
        if not self.api_key:
            return True
        key = self.headers.get("X-API-Key", "")
        if key != self.api_key:
            self._json({"error": "unauthorized"}, 401)
            return False
        return True

    def _json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_GET(self):
        if not self._check_auth():
            return
        url = urlparse(self.path)

        if url.path == "/health":
            self._json({"status": "ok", "engine": "zvec-fleet", "version": "1.0.0"})
        elif url.path == "/namespaces":
            self._json({"namespaces": self.fleet.namespaces()})
        else:
            self._json({"endpoints": ["/health", "/namespaces", "/search (POST)", "/index (POST)"]})

    def do_POST(self):
        if not self._check_auth():
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        url = urlparse(self.path)

        if url.path == "/search":
            embedding = body.get("embedding", [])
            if not embedding:
                self._json({"error": "missing embedding"}, 400)
                return
            result = self.fleet.search(
                embedding=embedding,
                topk=body.get("topk", 10),
                namespace=body.get("namespace", "all"),
                shared_only=body.get("shared_only", False),
            )
            self._json(result)

        elif url.path == "/index":
            namespace = body.get("namespace")
            if not namespace:
                self._json({"error": "missing namespace"}, 400)
                return
            docs = body.get("docs", [])
            if not docs:
                self._json({"error": "missing docs"}, 400)
                return
            result = self.fleet.index(namespace, docs)
            self._json(result)

        else:
            self._json({"error": "unknown endpoint"}, 404)


def main():
    parser = argparse.ArgumentParser(description="QMDZvec Fleet Memory Server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--api-key", default=os.environ.get("FLEET_API_KEY"))
    parser.add_argument("--dim", type=int, default=DIM)
    args = parser.parse_args()

    print(f"🚀 Fleet Memory Server starting on port {args.port}")
    print(f"   Data: {args.data}, Dim: {args.dim}")
    print(f"   Auth: {'API key required' if args.api_key else 'open (no auth)'}")

    fleet = FleetMemory(data_dir=args.data, dim=args.dim)
    FleetHandler.fleet = fleet
    FleetHandler.api_key = args.api_key

    server = HTTPServer(("0.0.0.0", args.port), FleetHandler)
    print(f"   Listening on http://0.0.0.0:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
