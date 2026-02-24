#!/bin/bash
# server-wrapper.sh — Kills stale port processes before starting memclawz server (#12)
set -e

ZVEC_PORT="${ZVEC_PORT:-4010}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-python3.10}"

# Kill any stale process on the port
STALE_PID=$(lsof -ti tcp:$ZVEC_PORT 2>/dev/null || true)
if [ -n "$STALE_PID" ]; then
    echo "[memclawz] Killing stale process on port $ZVEC_PORT (PID: $STALE_PID)"
    kill -9 $STALE_PID 2>/dev/null || true
    sleep 1
fi

# Remove stale lock files
LOCK_FILE="${ZVEC_DATA:-$HOME/.openclaw/zvec-memory}/memory/.lock"
if [ -f "$LOCK_FILE" ]; then
    echo "[memclawz] Removing stale lock file: $LOCK_FILE"
    rm -f "$LOCK_FILE"
fi

echo "[memclawz] Starting server on port $ZVEC_PORT..."
cd "$REPO_DIR"
exec "$PYTHON" -m uvicorn memclawz_server.server:app --host 127.0.0.1 --port "$ZVEC_PORT" --log-level info
