#!/usr/bin/env python3.10
"""
Causality Graph — Lightweight directed graph for causal memory.
Replaces Mem0 in memclawz v3.0.

Architecture informed by AMA-Bench (Zhao et al., 2026).
Storage: SQLite. Embeddings: numpy. No heavy deps.
"""
import os
import sqlite3
import time
import uuid
from typing import List, Optional, Dict, Any

import numpy as np

DB_PATH = os.environ.get(
    "CAUSALITY_DB",
    os.path.expanduser("~/.openclaw/zvec-memory/causality.db"),
)


def _cosine_sim(a: List[float], b: List[float]) -> float:
    a_np = np.array(a, dtype=np.float32)
    b_np = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(a_np) * np.linalg.norm(b_np)
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a_np, b_np) / denom)


class CausalityGraph:
    """SQLite-backed causality graph with embedding similarity + edge traversal."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DB_PATH
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                embedding BLOB,
                timestamp REAL,
                source TEXT DEFAULT '',
                created_at REAL
            );
            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                src TEXT NOT NULL,
                dst TEXT NOT NULL,
                edge_type TEXT NOT NULL CHECK(edge_type IN ('causality','association','similarity')),
                weight REAL DEFAULT 1.0,
                created_at REAL,
                FOREIGN KEY(src) REFERENCES nodes(id),
                FOREIGN KEY(dst) REFERENCES nodes(id)
            );
            CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
            CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
            CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type);
        """)
        self.conn.commit()

    def add_node(
        self,
        text: str,
        embedding: Optional[List[float]] = None,
        node_id: Optional[str] = None,
        source: str = "",
        timestamp: Optional[float] = None,
        caused_by: Optional[List[str]] = None,
        causes: Optional[List[str]] = None,
        associations: Optional[List[str]] = None,
    ) -> str:
        nid = node_id or str(uuid.uuid4())[:12]
        ts = timestamp or time.time()
        emb_blob = np.array(embedding, dtype=np.float32).tobytes() if embedding else None

        self.conn.execute(
            "INSERT OR REPLACE INTO nodes (id, text, embedding, timestamp, source, created_at) VALUES (?,?,?,?,?,?)",
            (nid, text, emb_blob, ts, source, time.time()),
        )

        for src_id in (caused_by or []):
            self.conn.execute(
                "INSERT INTO edges (src, dst, edge_type, weight, created_at) VALUES (?,?,?,?,?)",
                (src_id, nid, "causality", 1.0, time.time()),
            )
        for dst_id in (causes or []):
            self.conn.execute(
                "INSERT INTO edges (src, dst, edge_type, weight, created_at) VALUES (?,?,?,?,?)",
                (nid, dst_id, "causality", 1.0, time.time()),
            )
        for assoc_id in (associations or []):
            self.conn.execute(
                "INSERT INTO edges (src, dst, edge_type, weight, created_at) VALUES (?,?,?,?,?)",
                (nid, assoc_id, "association", 1.0, time.time()),
            )
            self.conn.execute(
                "INSERT INTO edges (src, dst, edge_type, weight, created_at) VALUES (?,?,?,?,?)",
                (assoc_id, nid, "association", 1.0, time.time()),
            )

        self.conn.commit()
        return nid

    def _get_node(self, node_id: str) -> Optional[Dict]:
        row = self.conn.execute("SELECT id, text, timestamp, source FROM nodes WHERE id=?", (node_id,)).fetchone()
        if not row:
            return None
        return {"id": row["id"], "text": row["text"], "timestamp": row["timestamp"], "source": row["source"]}

    def similarity_search(self, query_embedding: List[float], topk: int = 5) -> List[Dict]:
        rows = self.conn.execute("SELECT id, text, embedding, timestamp, source FROM nodes WHERE embedding IS NOT NULL").fetchall()
        scored = []
        for r in rows:
            emb = np.frombuffer(r["embedding"], dtype=np.float32).tolist()
            score = _cosine_sim(query_embedding, emb)
            scored.append({
                "id": r["id"], "text": r["text"], "score": score,
                "timestamp": r["timestamp"], "source": r["source"],
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:topk]

    def traverse_edges(self, node_ids: List[str], max_depth: int = 2, edge_types: Optional[List[str]] = None) -> List[Dict]:
        visited = set(node_ids)
        frontier = list(node_ids)
        results = []

        for depth in range(max_depth):
            if not frontier:
                break
            next_frontier = []
            placeholders = ",".join("?" * len(frontier))
            type_filter = ""
            params = list(frontier)
            if edge_types:
                type_filter = f" AND edge_type IN ({','.join('?' * len(edge_types))})"
                params.extend(edge_types)

            for direction, col_from, col_to in [("out", "src", "dst"), ("in", "dst", "src")]:
                rows = self.conn.execute(
                    f"SELECT {col_to} as neighbor, edge_type, weight FROM edges WHERE {col_from} IN ({placeholders}){type_filter}",
                    params,
                ).fetchall()
                for r in rows:
                    if r["neighbor"] not in visited:
                        visited.add(r["neighbor"])
                        next_frontier.append(r["neighbor"])
                        node = self._get_node(r["neighbor"])
                        if node:
                            node["edge_type"] = r["edge_type"]
                            node["depth"] = depth + 1
                            results.append(node)

            frontier = next_frontier
        return results

    def multi_hop_search(
        self, query_embedding: List[float], topk: int = 5,
        similarity_threshold: float = 0.5, max_depth: int = 2,
    ) -> Dict[str, Any]:
        sim_results = self.similarity_search(query_embedding, topk=topk)
        confidence = self._compute_confidence(sim_results, topk)
        all_results = list(sim_results)

        if confidence < similarity_threshold and sim_results:
            seed_ids = [r["id"] for r in sim_results[:3]]
            traversed = self.traverse_edges(seed_ids, max_depth=max_depth)
            seen_ids = {r["id"] for r in all_results}
            for t in traversed:
                if t["id"] not in seen_ids:
                    t["score"] = 0.0
                    all_results.append(t)
                    seen_ids.add(t["id"])

        return {
            "results": all_results[:topk * 2],
            "confidence": confidence,
            "similarity_count": len(sim_results),
            "traversal_count": len(all_results) - len(sim_results),
        }

    def keyword_search(self, keywords: str, limit: int = 10) -> List[Dict]:
        terms = keywords.lower().split()
        if not terms:
            return []
        rows = self.conn.execute("SELECT id, text, timestamp, source FROM nodes").fetchall()
        results = []
        for r in rows:
            text_lower = r["text"].lower()
            if all(t in text_lower for t in terms):
                results.append({
                    "id": r["id"], "text": r["text"],
                    "timestamp": r["timestamp"], "source": r["source"],
                    "score": sum(text_lower.count(t) for t in terms) / max(len(text_lower.split()), 1),
                })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def _compute_confidence(self, results: List[Dict], expected_k: int) -> float:
        if not results:
            return 0.0
        top_score = results[0]["score"]
        count_ratio = min(len(results) / max(expected_k, 1), 1.0)
        avg_score = sum(r["score"] for r in results) / len(results)
        return round(min(max(0.5 * top_score + 0.3 * avg_score + 0.2 * count_ratio, 0.0), 1.0), 4)

    def stats(self) -> Dict[str, Any]:
        node_count = self.conn.execute("SELECT COUNT(*) as c FROM nodes").fetchone()["c"]
        edge_count = self.conn.execute("SELECT COUNT(*) as c FROM edges").fetchone()["c"]
        edge_types = self.conn.execute("SELECT edge_type, COUNT(*) as c FROM edges GROUP BY edge_type").fetchall()
        return {"nodes": node_count, "edges": edge_count, "edge_types": {r["edge_type"]: r["c"] for r in edge_types}}

    def close(self):
        self.conn.close()
