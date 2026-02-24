#!/bin/bash
# Install QMDZvec into an OpenClaw workspace
set -e

WORKSPACE="${OPENCLAW_WORKSPACE:-$HOME/.openclaw/workspace}"
ZVEC_DATA="${ZVEC_DATA:-$HOME/.openclaw/zvec-memory}"

echo "=== QMDZvec Installer ==="
echo "Workspace: $WORKSPACE"
echo "Zvec data: $ZVEC_DATA"

# Install Python deps
pip install zvec numpy 2>/dev/null || pip3.10 install zvec numpy

# Create directories
mkdir -p "$WORKSPACE/memory/qmd"
mkdir -p "$ZVEC_DATA"

# Copy server if not present
if [ ! -f "$ZVEC_DATA/server.py" ]; then
    cp memclawz_server/server.py "$ZVEC_DATA/server.py"
    echo "Installed server.py to $ZVEC_DATA"
fi

# Initialize QMD if not present
if [ ! -f "$WORKSPACE/memory/qmd/current.json" ]; then
    cat > "$WORKSPACE/memory/qmd/current.json" <<'EOF'
{
  "session_id": "initial",
  "started_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "tasks": [],
  "entities_seen": {},
  "updated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
    echo "Created initial QMD at $WORKSPACE/memory/qmd/current.json"
fi

# Start Zvec server
echo "Starting Zvec server on port 4010..."
cd "$ZVEC_DATA"
nohup python3.10 server.py > server.log 2>&1 &
echo "Zvec server PID: $!"

# Migrate data if collection doesn't exist
sleep 2
if ! [ -d "$ZVEC_DATA/memory" ]; then
    echo "Migrating from SQLite..."
    curl -s http://localhost:4010/migrate
fi

# Start watcher
echo "Starting watcher..."
nohup python3.10 "$ZVEC_DATA/watcher.py" > "$ZVEC_DATA/watcher.log" 2>&1 &
echo "Watcher PID: $!"

echo "=== QMDZvec installed and running ==="
curl -s http://localhost:4010/health
