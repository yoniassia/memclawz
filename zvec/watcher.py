#!/usr/bin/env python3.10
"""
Zvec Auto-Indexing Watcher
Syncs new chunks from OpenClaw's SQLite memory DB into Zvec HNSW index.
Runs as a loop, checking every 60 seconds for new/updated chunks.
"""
import json
import os
import sqlite3
import time
import urllib.request

SQLITE_PATH = os.path.expanduser("~/.openclaw/memory/main.sqlite")
STATE_FILE = os.path.expanduser("~/.openclaw/workspace/zvec-memory/sync-state.json")
ZVEC_URL = "http://localhost:4010"
POLL_INTERVAL = 60  # seconds

def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except:
        return {"last_sync_id": 0, "total_synced": 0}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def get_new_chunks(last_id):
    """Get chunks from SQLite that haven't been synced yet."""
    if not os.path.exists(SQLITE_PATH):
        return []
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, path, source, start_line, end_line, text, embedding, updated_at
        FROM chunks
        WHERE embedding IS NOT NULL AND rowid > ?
        ORDER BY rowid
        LIMIT 500
    """, (last_id,)).fetchall()
    conn.close()
    return rows

def get_max_rowid():
    if not os.path.exists(SQLITE_PATH):
        return 0
    conn = sqlite3.connect(SQLITE_PATH)
    row = conn.execute("SELECT MAX(rowid) FROM chunks WHERE embedding IS NOT NULL").fetchone()
    conn.close()
    return row[0] or 0

def index_to_zvec(chunks):
    """Send chunks to zvec /index endpoint."""
    docs = []
    for row in chunks:
        emb_raw = row["embedding"]
        if not emb_raw:
            continue
        try:
            emb = json.loads(emb_raw) if isinstance(emb_raw, str) else list(emb_raw)
        except:
            continue
        docs.append({
            "id": str(row["id"]),
            "embedding": emb,
            "text": row["text"] or "",
            "path": row["path"] or "",
            "source": row["source"] or "",
            "start_line": row["start_line"] or 0,
            "end_line": row["end_line"] or 0,
        })
    
    if not docs:
        return 0
    
    data = json.dumps({"docs": docs}).encode()
    req = urllib.request.Request(
        f"{ZVEC_URL}/index",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        result = json.loads(resp.read())
        return result.get("indexed", len(docs))
    except Exception as e:
        print(f"Error indexing to zvec: {e}")
        return 0

def sync_once():
    state = load_state()
    max_id = get_max_rowid()
    
    if max_id <= state["last_sync_id"]:
        return 0  # Nothing new
    
    chunks = get_new_chunks(state["last_sync_id"])
    if not chunks:
        state["last_sync_id"] = max_id
        save_state(state)
        return 0
    
    indexed = index_to_zvec(chunks)
    state["last_sync_id"] = max_id
    state["total_synced"] = state.get("total_synced", 0) + indexed
    state["last_sync_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_state(state)
    return indexed

def main():
    print(f"Zvec watcher started. Polling every {POLL_INTERVAL}s")
    print(f"SQLite: {SQLITE_PATH}")
    print(f"Zvec: {ZVEC_URL}")
    
    # Check zvec health
    try:
        resp = urllib.request.urlopen(f"{ZVEC_URL}/health", timeout=5)
        print(f"Zvec health: {resp.read().decode()}")
    except:
        print("WARNING: Zvec server not reachable. Will retry on each sync.")
    
    while True:
        try:
            n = sync_once()
            if n > 0:
                print(f"Synced {n} new chunks to zvec")
        except Exception as e:
            print(f"Sync error: {e}")
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    if len(os.sys.argv) > 1 and os.sys.argv[1] == "--once":
        n = sync_once()
        print(f"Synced {n} chunks")
    else:
        main()
