#!/bin/bash
# QMDZvec History Bootstrap — Import ALL existing memory into Zvec
# Run once after install, or anytime to re-sync all history.
# Usage: bash scripts/bootstrap-history.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ZVEC_PORT="${ZVEC_PORT:-4010}"
WORKSPACE="${OPENCLAW_WORKSPACE:-$HOME/.openclaw/workspace}"
SQLITE_PATH="${SQLITE_PATH:-$HOME/.openclaw/memory/main.sqlite}"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✅ $1${NC}"; }
warn() { echo -e "${YELLOW}⚠️  $1${NC}"; }

# Find Python
PYTHON=""
for p in python3.10 python3.11 python3.12 python3; do
    if command -v "$p" &>/dev/null; then
        ver=$("$p" -c "import sys; v=sys.version_info; exit(0 if v.major>=3 and v.minor>=10 else 1)" 2>/dev/null) && PYTHON="$p" && break
    fi
done
[ -z "$PYTHON" ] && { echo "❌ Python 3.10+ required"; exit 1; }

# Ensure server is running
if ! curl -sf "http://localhost:$ZVEC_PORT/health" &>/dev/null; then
    echo "❌ Zvec server not running on port $ZVEC_PORT. Start it first."
    exit 1
fi

SQLITE_CHUNKS=0
MD_CHUNKS=0

# ── Step 1: Migrate SQLite chunks ────────────────────────
if [ -f "$SQLITE_PATH" ]; then
    COUNT=$("$PYTHON" -c "
import sqlite3, os
conn = sqlite3.connect('$SQLITE_PATH')
print(conn.execute('SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL').fetchone()[0])
conn.close()
" 2>/dev/null || echo "0")
    echo "📦 Found $COUNT chunks with embeddings in SQLite"
    
    if [ "$COUNT" -gt 0 ]; then
        echo "   Migrating via /migrate endpoint..."
        RESULT=$(curl -sf "http://localhost:$ZVEC_PORT/migrate" 2>/dev/null || echo '{"error":"failed"}')
        MIGRATED=$(echo "$RESULT" | "$PYTHON" -c "import sys,json; d=json.load(sys.stdin); print(d.get('migrated', d.get('total_docs', 0)))" 2>/dev/null || echo "0")
        SQLITE_CHUNKS=$MIGRATED
        ok "Migrated $MIGRATED chunks from SQLite"
    fi
else
    warn "No SQLite memory found at $SQLITE_PATH — skipping"
fi

# ── Step 2: Index markdown files not in SQLite ───────────
echo ""
echo "📄 Scanning workspace for .md files..."

"$PYTHON" << 'PYEOF'
import sys, os, json, glob

sys.path.insert(0, os.environ.get("REPO_DIR", "."))
WORKSPACE = os.environ.get("WORKSPACE", os.path.expanduser("~/.openclaw/workspace"))
ZVEC_PORT = os.environ.get("ZVEC_PORT", "4010")

from zvec.chunker import chunk_file
from zvec.embedder import text_to_embedding

import urllib.request

# Collect all .md files from workspace
md_files = []
for pattern in [
    os.path.join(WORKSPACE, "MEMORY.md"),
    os.path.join(WORKSPACE, "AGENTS.md"),
    os.path.join(WORKSPACE, "memory", "*.md"),
    os.path.join(WORKSPACE, "memory", "**", "*.md"),
    os.path.join(WORKSPACE, "knowledge", "**", "*.md"),
]:
    md_files.extend(glob.glob(pattern, recursive=True))

md_files = list(set(md_files))
print(f"   Found {len(md_files)} markdown files")

# Check what's already indexed via stats
try:
    req = urllib.request.Request(f"http://localhost:{ZVEC_PORT}/stats")
    resp = urllib.request.urlopen(req)
    stats = json.loads(resp.read())
    existing = stats.get("total_docs", 0)
except:
    existing = 0

# Chunk and index
total_new = 0
batch = []
for fp in md_files:
    if not os.path.exists(fp):
        continue
    try:
        chunks = chunk_file(fp)
    except Exception as e:
        print(f"   ⚠️  Failed to chunk {fp}: {e}")
        continue
    
    for c in chunks:
        emb = text_to_embedding(c.text)
        doc_id = f"md:{os.path.basename(fp)}:{c.start_line}"
        batch.append({
            "id": doc_id,
            "embedding": emb,
            "text": c.text[:2000],  # cap text size
            "path": fp,
        })
        
        # Flush in batches of 50
        if len(batch) >= 50:
            data = json.dumps({"docs": batch}).encode()
            req = urllib.request.Request(
                f"http://localhost:{ZVEC_PORT}/index",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            try:
                urllib.request.urlopen(req)
                total_new += len(batch)
            except Exception as e:
                print(f"   ⚠️  Batch index failed: {e}")
            batch = []

# Final batch
if batch:
    data = json.dumps({"docs": batch}).encode()
    req = urllib.request.Request(
        f"http://localhost:{ZVEC_PORT}/index",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        urllib.request.urlopen(req)
        total_new += len(batch)
    except Exception as e:
        print(f"   ⚠️  Final batch index failed: {e}")

print(f"   ✅ Indexed {total_new} new chunks from markdown files")

# Final stats
try:
    req = urllib.request.Request(f"http://localhost:{ZVEC_PORT}/stats")
    resp = urllib.request.urlopen(req)
    stats = json.loads(resp.read())
    total = stats.get("total_docs", 0)
    print(f"\n📊 Total chunks in index: {total}")
except:
    pass
PYEOF

echo ""
ok "History bootstrap complete!"
