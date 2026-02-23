#!/bin/bash
# QMDZvec — One-command setup
set -e

echo "🧠 Installing QMDZvec — Three-Speed Memory for OpenClaw"

# Install Python dependencies
pip install zvec numpy 2>/dev/null || pip3 install zvec numpy

# Create QMD directory
mkdir -p memory/qmd

# Initialize QMD if not exists
if [ ! -f memory/qmd/current.json ]; then
    cat > memory/qmd/current.json << 'EOF'
{
  "session_id": "initial",
  "tasks": [],
  "entities_seen": {},
  "updated_at": ""
}
EOF
    echo "✅ Created memory/qmd/current.json"
fi

# Create Zvec data directory
mkdir -p ~/.openclaw/zvec-memory
echo "✅ Created ~/.openclaw/zvec-memory"

# Start Zvec server
echo "🚀 Starting Zvec server on port ${ZVEC_PORT:-4010}..."
cd "$(dirname "$0")/.."
nohup python3.10 zvec/server.py > /tmp/zvec-server.log 2>&1 &
sleep 2

# Check health
if curl -sf http://localhost:${ZVEC_PORT:-4010}/health > /dev/null 2>&1; then
    echo "✅ Zvec server running on port ${ZVEC_PORT:-4010}"
else
    echo "⚠️  Server may still be starting — check /tmp/zvec-server.log"
fi

# Migrate existing memory if SQLite exists
if [ -f ~/.openclaw/memory/main.sqlite ]; then
    echo "📦 Migrating existing OpenClaw memory to Zvec..."
    curl -sf http://localhost:${ZVEC_PORT:-4010}/migrate
    echo ""
    echo "✅ Migration complete"
fi

echo ""
echo "🧠 QMDZvec is ready! Add this to your AGENTS.md:"
echo '  - On session start: Read memory/qmd/current.json'
echo '  - For search: POST to http://localhost:4010/search'
echo '  - For compaction: python3.10 scripts/qmd-compact.py'
