#!/usr/bin/env python3
"""
Embedding bridge: Uses QMD's built-in vsearch to extract embeddings,
or calls QMD's node-llama-cpp engine via subprocess for embedding generation.

Usage:
  python3 embed_bridge.py --search "Mor wife name"
  python3 embed_bridge.py --reindex
  python3 embed_bridge.py --serve --port 4011
"""
import json
import os
import sys
import time
import glob
import subprocess
import hashlib
from pathlib import Path

ZVEC_URL = "http://localhost:4010"
WORKSPACE = os.environ.get("OPENCLAW_WORKSPACE", os.path.expanduser("~/.openclaw/workspace"))

def embed_via_qmd(text: str) -> list:
    """Use QMD's node-llama-cpp to generate embeddings via a helper script."""
    # QMD uses node-llama-cpp internally. We call a small Node.js script
    # that loads the same GGUF model and returns the embedding vector.
    script = os.path.join(os.path.dirname(__file__), "_embed_node.mjs")
    if not os.path.exists(script):
        _create_node_embed_script(script)
    
    result = subprocess.run(
        ["node", "--experimental-modules", script, text[:500]],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        raise RuntimeError(f"Embedding failed: {result.stderr[:200]}")
    return json.loads(result.stdout)

def _create_node_embed_script(path):
    """Create a Node.js script that uses node-llama-cpp for embeddings."""
    script = '''
import { getLlama } from "node-llama-cpp";
import { fileURLToPath } from "url";
import path from "path";
import os from "os";

const text = process.argv[2] || "";
const modelPath = path.join(os.homedir(), ".cache/qmd/models/hf_ggml-org_embeddinggemma-300M-Q8_0.gguf");

const llama = await getLlama();
const model = await llama.loadModel({ modelPath });
const ctx = await model.createEmbeddingContext();
const embedding = await ctx.getEmbeddingFor(text);
console.log(JSON.stringify(Array.from(embedding.vector)));
process.exit(0);
'''
    Path(path).write_text(script)

def search_zvec(query: str, top_k: int = 10):
    """Embed query locally via QMD's model, then search Zvec."""
    import urllib.request
    
    embedding = embed_via_qmd(query)
    
    payload = json.dumps({
        "embedding": embedding,
        "top_k": top_k
    }).encode()
    
    req = urllib.request.Request(
        f"{ZVEC_URL}/search",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}

def chunk_text(text: str, path: str, chunk_size: int = 20, overlap: int = 3):
    """Split text into overlapping line-based chunks."""
    lines = text.split('\n')
    chunks = []
    i = 0
    while i < len(lines):
        end = min(i + chunk_size, len(lines))
        chunk_lines = lines[i:end]
        chunk_text = '\n'.join(chunk_lines)
        if chunk_text.strip():
            chunk_id = f"{path}_{i}_{end}"
            chunk_id = chunk_id.replace(":", "_").replace("/", "_").replace(" ", "_")
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "path": path,
                "source": "memory",
                "start_line": i + 1,
                "end_line": end
            })
        i += chunk_size - overlap
    return chunks

def reindex_all():
    """Re-index all memory files with local GGUF embeddings via node-llama-cpp."""
    import urllib.request
    
    memory_dir = os.path.join(WORKSPACE, "memory")
    patterns = [
        os.path.join(memory_dir, "*.md"),
        os.path.join(WORKSPACE, "MEMORY.md"),
        os.path.join(WORKSPACE, "USER.md"),
        os.path.join(WORKSPACE, "AGENTS.md"),
        os.path.join(WORKSPACE, "IDENTITY.md"),
        os.path.join(WORKSPACE, "notes/people/*.md"),
    ]
    
    all_files = set()
    for pattern in patterns:
        all_files.update(glob.glob(pattern, recursive=True))
    
    print(f"Found {len(all_files)} files to index")
    
    all_chunks = []
    for fpath in sorted(all_files):
        try:
            text = Path(fpath).read_text(encoding='utf-8')
            rel = os.path.relpath(fpath, WORKSPACE)
            chunks = chunk_text(text, rel)
            all_chunks.extend(chunks)
            print(f"  {rel}: {len(chunks)} chunks")
        except Exception as e:
            print(f"  SKIP {fpath}: {e}")
    
    print(f"\nTotal chunks: {len(all_chunks)}")
    print("Generating embeddings via node-llama-cpp (local GGUF)...")
    
    # Embed one at a time (model load is cached after first call)
    indexed = 0
    batch = []
    for i, chunk in enumerate(all_chunks):
        try:
            emb = embed_via_qmd(chunk["text"][:500])
            chunk["embedding"] = emb
            batch.append(chunk)
            
            # Send in batches of 10
            if len(batch) >= 10 or i == len(all_chunks) - 1:
                payload = json.dumps({"docs": batch}).encode()
                req = urllib.request.Request(
                    f"{ZVEC_URL}/index",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                resp = urllib.request.urlopen(req, timeout=30)
                result = json.loads(resp.read())
                indexed += result.get("indexed", 0)
                print(f"  [{i+1}/{len(all_chunks)}] Indexed batch of {len(batch)}")
                batch = []
        except Exception as e:
            print(f"  [{i+1}] FAILED: {e}")
            batch = []
    
    print(f"\n✅ Indexed {indexed} total chunks with local embeddings")

if __name__ == "__main__":
    if "--search" in sys.argv:
        idx = sys.argv.index("--search")
        query = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else ""
        results = search_zvec(query)
        print(json.dumps(results, indent=2))
    
    elif "--reindex" in sys.argv:
        reindex_all()
    
    else:
        print(__doc__)
