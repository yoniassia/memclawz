#!/usr/bin/env python3
"""Migrate all workspace markdown files into the local HNSW index."""
import os, sys, time, json, numpy as np

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
print = lambda *a, **k: (__builtins__['print'] if isinstance(__builtins__, dict) else __builtins__.print)(*a, **k, flush=True)
import builtins
_print = builtins.print
def print(*a, **k):
    k['flush'] = True
    _print(*a, **k)

from embedder import embed_local, _get_local_model
import hnswlib

WORKSPACE = os.path.expanduser("~/.openclaw/workspace")
DATA_DIR = os.path.expanduser("~/.openclaw/zvec-memory")
INDEX_PATH = os.path.join(DATA_DIR, "hnsw.index")
META_PATH = os.path.join(DATA_DIR, "meta.json")
DIM = 768

os.makedirs(DATA_DIR, exist_ok=True)

def chunk_file(path, max_chunk=500):
    try:
        with open(path, 'r', errors='ignore') as f:
            content = f.read()
    except:
        return []
    if not content.strip():
        return []
    sections = content.split('\n\n')
    chunks = []
    current = ''
    line_num = 1
    start_line = 1
    for section in sections:
        lines_in_section = section.count('\n') + 1
        if len(current) + len(section) > max_chunk and current:
            chunks.append({'text': current.strip(), 'start_line': start_line, 'end_line': line_num - 1})
            current = section
            start_line = line_num
        else:
            current += '\n\n' + section if current else section
        line_num += lines_in_section
    if current.strip():
        chunks.append({'text': current.strip(), 'start_line': start_line, 'end_line': line_num})
    return chunks

# 1. Collect files
print("Collecting markdown files...")
md_files = []
skip_dirs = {'node_modules', '.git', '.next', 'dist', 'archive', '__pycache__'}
for root, dirs, files in os.walk(WORKSPACE):
    dirs[:] = [d for d in dirs if d not in skip_dirs]
    for f in files:
        if f.endswith('.md'):
            md_files.append(os.path.join(root, f))
print(f"Found {len(md_files)} files")

# 2. Chunk
print("Chunking...")
all_docs = []
for path in md_files:
    rel = os.path.relpath(path, WORKSPACE)
    for c in chunk_file(path):
        if len(c['text']) >= 20:
            all_docs.append({**c, 'path': rel, 'source': 'workspace'})
print(f"Total chunks: {len(all_docs)}")

# 3. Embed
print("Loading embedding model...")
model = _get_local_model()
print("Embedding chunks...")
t0 = time.time()
BATCH = 64
for i in range(0, len(all_docs), BATCH):
    batch_texts = [d['text'] for d in all_docs[i:i+BATCH]]
    embs = model.encode(batch_texts, normalize_embeddings=True, show_progress_bar=False)
    for j, emb in enumerate(embs):
        all_docs[i+j]['embedding'] = emb.tolist()
    done = min(i+BATCH, len(all_docs))
    elapsed = time.time() - t0
    rate = done / elapsed if elapsed > 0 else 0
    print(f"  {done}/{len(all_docs)} ({rate:.0f}/sec)")

t1 = time.time()
print(f"Embedding done: {t1-t0:.1f}s")

# 4. Build HNSW index
print("Building HNSW index...")
idx = hnswlib.Index(space='cosine', dim=DIM)
idx.init_index(max_elements=max(len(all_docs)*2, 10000), ef_construction=200, M=32)
idx.set_ef(128)

embeddings = np.array([d['embedding'] for d in all_docs], dtype=np.float32)
ids = np.arange(len(all_docs))
idx.add_items(embeddings, ids)
idx.save_index(INDEX_PATH)

# Save metadata
doc_meta = {}
for i, d in enumerate(all_docs):
    doc_meta[str(i)] = {
        'text': d['text'], 'path': d['path'], 'source': d['source'],
        'start_line': d['start_line'], 'end_line': d['end_line']
    }
with open(META_PATH, 'w') as f:
    json.dump({'docs': doc_meta, 'next_id': len(all_docs), 'dim': DIM}, f)

print(f"\n✅ Indexed {len(all_docs)} chunks from {len(md_files)} files")
print(f"   Index: {INDEX_PATH} ({os.path.getsize(INDEX_PATH)/1024/1024:.1f}MB)")
print(f"   Meta: {META_PATH} ({os.path.getsize(META_PATH)/1024/1024:.1f}MB)")

# 5. Test search
print("\n--- Test Searches ---")
queries = ['eToro brand guidelines', 'compliance disclaimer FCA', 'Yoni family']
for q in queries:
    emb = np.array([embed_local(q)], dtype=np.float32)
    labels, dists = idx.knn_query(emb, k=3)
    print(f"\nQ: \"{q}\"")
    for l, d in zip(labels[0], dists[0]):
        m = doc_meta[str(l)]
        print(f"  ({1-d:.3f}) [{m['path']}:{m['start_line']}] {m['text'][:80]}...")

print("\n🏠 100% LOCAL — zero external APIs")
