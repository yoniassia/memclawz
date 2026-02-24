#!/usr/bin/env python3.10
"""
Standalone FastAPI embedding server for memclawz (#17).
Uses _embed_node.mjs via subprocess to generate embeddings.
Runs on port 4020.

Usage:
    python3.10 -m uvicorn memclawz_server.embed_server:app --host 127.0.0.1 --port 4020
"""
import json
import os
import subprocess
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

EMBED_SCRIPT = os.path.join(os.path.dirname(__file__), "_embed_node.mjs")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "")

# Auto-detect model
if not EMBED_MODEL:
    import glob
    search_paths = [
        os.path.expanduser("~/.node-llama-cpp/models/"),
        os.path.expanduser("~/.cache/qmd/models/"),
        os.path.expanduser("~/.cache/node-llama-cpp/"),
    ]
    for sp in search_paths:
        if os.path.isdir(sp):
            gguf_files = glob.glob(os.path.join(sp, "*embedding*.gguf"))
            if gguf_files:
                EMBED_MODEL = gguf_files[0]
                break

PORT = int(os.environ.get("EMBED_PORT", "4020"))

app = FastAPI(title="memclawz-embed", description="Local embedding server using node-llama-cpp")

# Persistent subprocess
_proc = None


class EmbedRequest(BaseModel):
    texts: List[str]


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]


def _get_proc():
    global _proc
    if _proc and _proc.poll() is None:
        return _proc
    if not EMBED_MODEL:
        raise RuntimeError("No GGUF embedding model found. Set EMBED_MODEL env var.")
    if not os.path.exists(EMBED_SCRIPT):
        raise RuntimeError(f"Missing {EMBED_SCRIPT}")

    _proc = subprocess.Popen(
        ["node", EMBED_SCRIPT, EMBED_MODEL, "--stream"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1
    )
    line = _proc.stdout.readline().strip()
    if line != "READY":
        raise RuntimeError(f"Embed process failed: {line}")
    return _proc


@app.get("/health")
async def health():
    return {"status": "ok", "model": os.path.basename(EMBED_MODEL) if EMBED_MODEL else "none"}


@app.post("/embed")
async def embed(req: EmbedRequest):
    """Generate embeddings for a list of texts."""
    if not req.texts:
        raise HTTPException(status_code=400, detail="empty texts list")
    try:
        proc = _get_proc()
        embeddings = []
        for text in req.texts:
            clean = text.replace("\n", " ").replace("\r", " ").strip() or "empty"
            proc.stdin.write(clean + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline().strip()
            if not line:
                raise RuntimeError("Empty response from embedding process")
            embeddings.append(json.loads(line))
        return {"embeddings": embeddings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print(f"memclawz-embed starting on port {PORT}")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
