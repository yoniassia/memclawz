#!/usr/bin/env python3.10
"""
MemClawz v2.0 Gateway - Unified API for Zvec + Mem0
Hybrid memory system combining speed of Zvec (~8ms) with intelligence of Mem0 (~100ms)

Architecture:
- QMD Layer (<1ms): Hot tasks from memory/qmd/current.json  
- Zvec Layer (~8ms): HNSW index for existing memories
- Mem0 Layer (~100ms): Smart memory with auto-extraction
"""

import json
import os
import time
import traceback
from typing import List, Optional, Dict, Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import uvicorn

from mem0_config import create_mem0_memory

PORT = int(os.environ.get("MEMCLAWZ_V2_PORT", "4011"))
ZVEC_PORT = int(os.environ.get("ZVEC_PORT", "4010"))
ZVEC_URL = f"http://localhost:{ZVEC_PORT}"
QMD_PATH = os.path.expanduser("~/.openclaw/workspace/memory/qmd/current.json")

app = FastAPI(
    title="MemClawz v2.0", 
    description="Hybrid memory system: Zvec speed + Mem0 intelligence"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global memory instance
mem0_memory = None

def init_mem0():
    """Initialize Mem0 memory instance"""
    global mem0_memory
    if mem0_memory is None:
        try:
            mem0_memory = create_mem0_memory()
            print("✓ Mem0 initialized successfully")
        except Exception as e:
            print(f"✗ Mem0 initialization failed: {e}")
            mem0_memory = None
    return mem0_memory

# Pydantic models
class SearchRequest(BaseModel):
    text: str
    topk: int = 10
    user_id: str = "default"

class SearchResult(BaseModel):
    id: str
    score: float
    text: str
    source: str
    metadata: Optional[Dict] = {}

class SearchResponse(BaseModel):
    results: List[SearchResult]
    sources: List[str]
    total_time_ms: float

class IndexRequest(BaseModel):
    text: str
    metadata: Optional[Dict] = {}
    user_id: str = "default"

class ExtractRequest(BaseModel):
    conversation: str
    user_id: str = "default"

class IngestRequest(BaseModel):
    conversation: str
    user_id: str = "default"
    auto_index: bool = True

def load_qmd() -> List[Dict]:
    """Load QMD (Quick Memory Data) for <1ms hot tasks"""
    try:
        if os.path.exists(QMD_PATH):
            with open(QMD_PATH, 'r') as f:
                qmd_data = json.load(f)
                # Extract tasks as searchable text
                if isinstance(qmd_data, dict):
                    tasks = []
                    for key, value in qmd_data.items():
                        if isinstance(value, dict):
                            tasks.append({
                                "id": f"qmd:{key}",
                                "text": json.dumps(value, indent=2),
                                "source": "qmd",
                                "metadata": {"type": "task", "key": key}
                            })
                    return tasks
        return []
    except Exception as e:
        print(f"QMD load error: {e}")
        return []

def search_qmd(query: str, topk: int = 3) -> List[Dict]:
    """Search QMD data - simple text matching"""
    qmd_tasks = load_qmd()
    query_lower = query.lower()
    
    matches = []
    for task in qmd_tasks:
        text_lower = task["text"].lower()
        if query_lower in text_lower:
            # Simple relevance scoring
            score = query_lower.count(query_lower) / len(text_lower.split())
            matches.append({
                "id": task["id"],
                "score": min(score, 0.95),  # Cap at 0.95 to distinguish from exact matches
                "text": task["text"][:500] + "..." if len(task["text"]) > 500 else task["text"],
                "source": "qmd",
                "metadata": task.get("metadata", {})
            })
    
    # Sort by score and return top results
    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches[:topk]

def search_zvec(query: str, embedding: List[float] = None, topk: int = 10) -> List[Dict]:
    """Search Zvec layer (~8ms)"""
    try:
        if not embedding:
            # If no embedding provided, we'd need to generate it
            # For now, skip Zvec search if no embedding
            return []
        
        response = requests.post(
            f"{ZVEC_URL}/search",
            json={"embedding": embedding, "topk": topk},
            timeout=2.0
        )
        
        if response.status_code == 200:
            data = response.json()
            results = []
            for item in data.get("results", []):
                results.append({
                    "id": item.get("id", ""),
                    "score": item.get("score", 0.0),
                    "text": item.get("text", ""),
                    "source": "zvec",
                    "metadata": {
                        "path": item.get("path", ""),
                        "source_type": item.get("source", ""),
                        "start_line": item.get("start_line"),
                        "end_line": item.get("end_line")
                    }
                })
            return results
        else:
            print(f"Zvec search failed: {response.status_code}")
            return []
    except Exception as e:
        print(f"Zvec search error: {e}")
        return []

def search_mem0(query: str, user_id: str = "default", topk: int = 10) -> List[Dict]:
    """Search Mem0 layer (~100ms)"""
    try:
        mem0 = init_mem0()
        if not mem0:
            return []
        
        search_results = mem0.search(query, user_id=user_id, limit=topk)
        
        results = []
        for i, item in enumerate(search_results.get("results", [])):
            results.append({
                "id": item.get("id", f"mem0:{i}"),
                "score": item.get("score", 0.0),
                "text": item.get("memory", item.get("text", "")),
                "source": "mem0",
                "metadata": item.get("metadata", {})
            })
        return results
    except Exception as e:
        print(f"Mem0 search error: {e}")
        return []

def dedupe_and_rank(results: List[Dict]) -> List[Dict]:
    """Dedupe and rank results from multiple sources"""
    # Simple deduplication by text similarity
    unique_results = []
    seen_texts = set()
    
    for result in results:
        text = result["text"][:100].lower().strip()
        if text not in seen_texts:
            seen_texts.add(text)
            unique_results.append(result)
    
    # Sort by score (highest first)
    unique_results.sort(key=lambda x: x["score"], reverse=True)
    return unique_results

@app.get("/")
async def root():
    return {
        "service": "MemClawz v2.0",
        "architecture": "QMD (<1ms) + Zvec (~8ms) + Mem0 (~100ms)",
        "endpoints": ["/health", "/stats", "/search", "/index", "/extract", "/ingest", "/migrate"]
    }

@app.get("/health")
async def health():
    """Health check for all layers"""
    start_time = time.time()
    
    # Check QMD
    qmd_status = "ok" if os.path.exists(QMD_PATH) else "missing"
    
    # Check Zvec
    zvec_status = "unknown"
    try:
        resp = requests.get(f"{ZVEC_URL}/health", timeout=1.0)
        zvec_status = "ok" if resp.status_code == 200 else "error"
    except:
        zvec_status = "down"
    
    # Check Mem0
    mem0_status = "ok" if init_mem0() else "error"
    
    total_time = (time.time() - start_time) * 1000
    
    return {
        "status": "ok" if all(s in ["ok", "missing"] for s in [qmd_status, zvec_status, mem0_status]) else "partial",
        "layers": {
            "qmd": qmd_status,
            "zvec": zvec_status, 
            "mem0": mem0_status
        },
        "check_time_ms": round(total_time, 2)
    }

@app.get("/stats")
async def stats():
    """Combined stats from all layers"""
    stats_data = {
        "qmd": {"count": len(load_qmd())},
        "zvec": {"count": 0},
        "mem0": {"count": 0}
    }
    
    # Get Zvec stats
    try:
        resp = requests.get(f"{ZVEC_URL}/stats", timeout=1.0)
        if resp.status_code == 200:
            zvec_data = resp.json()
            stats_data["zvec"] = zvec_data
    except:
        pass
    
    # Mem0 stats would require custom implementation
    # For now, just indicate if it's available
    stats_data["mem0"]["available"] = init_mem0() is not None
    
    return stats_data

@app.post("/search")
async def unified_search(req: SearchRequest):
    """Unified search across QMD + Zvec + Mem0"""
    start_time = time.time()
    
    all_results = []
    sources_used = []
    
    # 1. Search QMD (fastest <1ms)
    qmd_results = search_qmd(req.text, topk=3)
    if qmd_results:
        all_results.extend(qmd_results)
        sources_used.append("qmd")
    
    # 2. Search Mem0 (smart memory ~100ms) 
    mem0_results = search_mem0(req.text, req.user_id, topk=req.topk)
    if mem0_results:
        all_results.extend(mem0_results)
        sources_used.append("mem0")
    
    # 3. Search Zvec (fast vector search ~8ms) - would need embedding generation
    # For now skip Zvec in unified search unless we have embeddings
    # zvec_results = search_zvec(req.text, topk=req.topk)
    
    # Dedupe and rank
    final_results = dedupe_and_rank(all_results)[:req.topk]
    
    total_time = (time.time() - start_time) * 1000
    
    return SearchResponse(
        results=[SearchResult(**r) for r in final_results],
        sources=sources_used,
        total_time_ms=round(total_time, 2)
    )

@app.post("/index")
async def unified_index(req: IndexRequest):
    """Write to both Zvec and Mem0"""
    results = {"zvec": None, "mem0": None}
    
    # Index to Mem0
    try:
        mem0 = init_mem0()
        if mem0:
            mem0_result = mem0.add(req.text, user_id=req.user_id, metadata=req.metadata)
            results["mem0"] = mem0_result
    except Exception as e:
        print(f"Mem0 index error: {e}")
        results["mem0"] = {"error": str(e)}
    
    # Index to Zvec would require embedding generation
    # For now, just indicate the intent
    results["zvec"] = {"status": "would_need_embedding"}
    
    return {"indexed": results}

@app.post("/extract")
async def extract_facts(req: ExtractRequest):
    """Mem0-only: Auto-extract facts from conversation"""
    try:
        mem0 = init_mem0()
        if not mem0:
            raise HTTPException(status_code=503, detail="Mem0 not available")
        
        # Mem0 auto-extracts facts when adding memories
        result = mem0.add(req.conversation, user_id=req.user_id)
        
        return {
            "extracted": True,
            "result": result,
            "user_id": req.user_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

@app.post("/ingest")
async def ingest_conversation(req: IngestRequest):
    """Auto-extract facts and optionally index key facts to Zvec"""
    try:
        mem0 = init_mem0()
        if not mem0:
            raise HTTPException(status_code=503, detail="Mem0 not available")
        
        # Extract facts with Mem0
        mem0_result = mem0.add(req.conversation, user_id=req.user_id)
        
        results = {
            "mem0_extraction": mem0_result,
            "zvec_index": None
        }
        
        # If auto_index is True, we'd also index key facts to Zvec
        # For now, just indicate the intent
        if req.auto_index:
            results["zvec_index"] = {"status": "would_index_key_facts"}
        
        return {
            "ingested": True,
            "results": results,
            "user_id": req.user_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.get("/migrate") 
async def migrate_zvec_to_mem0():
    """Migrate existing Zvec memories into Mem0"""
    try:
        # Get all memories from Zvec
        resp = requests.get(f"{ZVEC_URL}/stats", timeout=2.0)
        if resp.status_code != 200:
            raise HTTPException(status_code=503, detail="Zvec not available")
        
        mem0 = init_mem0()
        if not mem0:
            raise HTTPException(status_code=503, detail="Mem0 not available")
        
        # For now, just return migration plan
        # Full implementation would require iterating through Zvec docs
        return {
            "migration_plan": "Copy Zvec documents to Mem0 while preserving metadata",
            "status": "not_implemented",
            "note": "Would migrate existing memories from Zvec to Mem0 layer"
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print(f"🔨 MemClawz v2.0 starting on port {PORT}")
    print(f"   QMD Path: {QMD_PATH}")
    print(f"   Zvec URL: {ZVEC_URL}")
    
    # Initialize Mem0 on startup
    init_mem0()
    
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")