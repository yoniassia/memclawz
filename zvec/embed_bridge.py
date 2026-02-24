#!/usr/bin/env python3
"""
Local Embedding Bridge for memclawz

Generates embeddings using QMD's locally-downloaded GGUF model (embeddinggemma-300M)
via node-llama-cpp, eliminating the need for OpenAI API keys.

Usage:
  # Index all memory files with local embeddings
  python3 zvec/embed_bridge.py --reindex

  # Search using local embeddings
  python3 zvec/embed_bridge.py --search "what is memclawz"

  # Generate embedding for text (outputs JSON array)
  python3 zvec/embed_bridge.py --embed "hello world"

Environment:
  ZVEC_PORT     — Zvec server port (default: 4010)
  EMBED_MODEL   — Path to GGUF model (auto-detected if not set)
  EMBED_DIM     — Embedding dimension (default: 256)
"""

import os
import sys
import json
import glob
import subprocess
import urllib.request
from pathlib import Path

ZVEC_PORT = int(os.environ.get("ZVEC_PORT", 4010))
ZVEC_URL = f"http://localhost:{ZVEC_PORT}"

# Auto-detect GGUF model path
EMBED_MODEL = os.environ.get("EMBED_MODEL", "")
if not EMBED_MODEL:
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

EMBED_SCRIPT = os.path.join(os.path.dirname(__file__), "_embed_node.mjs")

# Persistent embedding process
_embed_proc = None
_embed_ready = False


def _get_embed_proc():
    """Get or start persistent node embedding process."""
    global _embed_proc, _embed_ready
    if _embed_proc and _embed_proc.poll() is None:
        return _embed_proc

    if not EMBED_MODEL:
        raise RuntimeError("No GGUF embedding model found. Set EMBED_MODEL or install embeddinggemma.")
    if not os.path.exists(EMBED_SCRIPT):
        raise RuntimeError(f"Missing {EMBED_SCRIPT} — run from memclawz root")

    _embed_proc = subprocess.Popen(
        ["node", EMBED_SCRIPT, EMBED_MODEL, "--stream"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1
    )
    # Wait for READY signal
    line = _embed_proc.stdout.readline().strip()
    if line != "READY":
        raise RuntimeError(f"Embedding process failed to start: {line}")
    _embed_ready = True
    return _embed_proc


def _shutdown_embed():
    global _embed_proc, _embed_ready
    if _embed_proc and _embed_proc.poll() is None:
        try:
            _embed_proc.stdin.write("EXIT\n")
            _embed_proc.stdin.flush()
            _embed_proc.wait(timeout=5)
        except:
            _embed_proc.kill()
    _embed_proc = None
    _embed_ready = False

import atexit
atexit.register(_shutdown_embed)


def embed_text(text: str) -> list:
    """Generate embedding vector for text using persistent local GGUF model."""
    proc = _get_embed_proc()
    # Replace newlines with spaces for single-line protocol
    clean = text.replace("\n", " ").replace("\r", " ").strip()
    if not clean:
        clean = "empty"
    proc.stdin.write(clean + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline().strip()
    if not line:
        raise RuntimeError("Empty response from embedding process")
    return json.loads(line)


def embed_batch(texts: list) -> list:
    """Generate embeddings for multiple texts via persistent process."""
    proc = _get_embed_proc()
    results = []
    for text in texts:
        clean = text.replace("\n", " ").replace("\r", " ").strip()
        if not clean:
            clean = "empty"
        proc.stdin.write(clean + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline().strip()
        if not line:
            raise RuntimeError("Empty response from embedding process")
        results.append(json.loads(line))
    return results


def zvec_request(path: str, data=None):
    """Make a request to the Zvec server."""
    url = f"{ZVEC_URL}{path}"
    if data:
        req = urllib.request.Request(url, data=json.dumps(data).encode(),
                                     headers={"Content-Type": "application/json"})
    else:
        req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def reindex(workspace=None):
    """Reindex all memory files using local embeddings."""
    workspace = workspace or os.environ.get("OPENCLAW_WORKSPACE",
        os.path.expanduser("~/.openclaw/workspace"))

    # Find all markdown files
    md_patterns = [
        os.path.join(workspace, "*.md"),
        os.path.join(workspace, "memory", "*.md"),
        os.path.join(workspace, "memory", "**", "*.md"),
        os.path.join(workspace, "knowledge", "**", "*.md"),
    ]

    files = []
    for pattern in md_patterns:
        files.extend(glob.glob(pattern, recursive=True))
    files = sorted(set(files))

    if not files:
        print("No markdown files found to index.")
        return

    print(f"📄 Found {len(files)} files to index")
    print(f"🧠 Model: {os.path.basename(EMBED_MODEL)}")

    # Chunk files
    sys.path.insert(0, os.path.dirname(__file__))
    try:
        from chunker import chunk_file
    except ImportError:
        # Simple fallback chunker
        def chunk_file(path):
            text = open(path).read()
            chunks = []
            lines = text.split("\n")
            chunk_size = 20
            for i in range(0, len(lines), chunk_size):
                chunk_lines = lines[i:i + chunk_size]
                chunk_text = "\n".join(chunk_lines)
                if chunk_text.strip():
                    chunks.append({
                        "text": chunk_text,
                        "start_line": i + 1,
                        "end_line": min(i + chunk_size, len(lines)),
                    })
            return chunks

    total_chunks = 0
    total_indexed = 0
    batch_docs = []
    batch_texts = []

    for f_path in files:
        rel_path = os.path.relpath(f_path, workspace)
        try:
            chunks = chunk_file(f_path)
        except Exception as e:
            print(f"  ⚠️ Skip {rel_path}: {e}")
            continue

        for chunk in chunks:
            # Support both dict and dataclass Chunk objects
            if hasattr(chunk, "start_line"):
                start_line = chunk.start_line
                end_line = getattr(chunk, "end_line", 0)
                text = chunk.text
            else:
                start_line = chunk.get("start_line", 0)
                end_line = chunk.get("end_line", 0)
                text = chunk.get("text", "")

            doc_id = f"md_{rel_path}_{start_line}"
            doc_id = doc_id.replace(":", "_").replace("/", "_").replace(" ", "_")

            batch_docs.append({
                "id": doc_id,
                "text": text,
                "path": rel_path,
                "source": "embed_bridge",
                "start_line": start_line,
                "end_line": end_line,
            })
            batch_texts.append(text)
            total_chunks += 1

        # Process in batches of 10
        if len(batch_texts) >= 10:
            try:
                embeddings = embed_batch(batch_texts)
                for doc, emb in zip(batch_docs, embeddings):
                    doc["embedding"] = emb
                zvec_request("/index", {"docs": batch_docs})
                total_indexed += len(batch_docs)
                print(f"  ✅ Indexed batch ({total_indexed}/{total_chunks})")
            except Exception as e:
                print(f"  ❌ Batch failed: {e}")
            batch_docs = []
            batch_texts = []

    # Final batch
    if batch_texts:
        try:
            embeddings = embed_batch(batch_texts)
            for doc, emb in zip(batch_docs, embeddings):
                doc["embedding"] = emb
            zvec_request("/index", {"docs": batch_docs})
            total_indexed += len(batch_docs)
        except Exception as e:
            print(f"  ❌ Final batch failed: {e}")

    print(f"\n✅ Indexed {total_indexed}/{total_chunks} chunks from {len(files)} files")
    stats = zvec_request("/stats")
    print(f"📊 Zvec stats: {json.dumps(stats)}")


def search(query: str, topk: int = 5):
    """Search Zvec using locally-generated embedding."""
    print(f"🔍 Query: {query}")
    embedding = embed_text(query)
    results = zvec_request("/search", {"embedding": embedding, "topk": topk})
    
    if not results.get("results"):
        print("No results found.")
        return results

    for i, r in enumerate(results["results"]):
        score = r.get("score", 0)
        path = r.get("fields", {}).get("path", "?")
        text = r.get("fields", {}).get("text", "")[:120]
        print(f"  {i+1}. [{score:.3f}] {path}")
        print(f"     {text}...")
    
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    if sys.argv[1] == "--reindex":
        workspace = sys.argv[2] if len(sys.argv) > 2 else None
        reindex(workspace)
    elif sys.argv[1] == "--search":
        query = " ".join(sys.argv[2:])
        search(query)
    elif sys.argv[1] == "--embed":
        text = " ".join(sys.argv[2:])
        emb = embed_text(text)
        print(json.dumps(emb))
    else:
        print(f"Unknown flag: {sys.argv[1]}")
        print("Use --reindex, --search, or --embed")
        sys.exit(1)
