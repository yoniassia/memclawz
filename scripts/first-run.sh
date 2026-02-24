#!/bin/bash
# QMDZvec First Run — Complete Setup
# Usage: cd QMDZvec && bash scripts/first-run.sh
#
# Goes from fresh clone to fully operational memory in one command.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ZVEC_PORT="${ZVEC_PORT:-4010}"
WORKSPACE="${OPENCLAW_WORKSPACE:-$HOME/.openclaw/workspace}"
SQLITE_PATH="${SQLITE_PATH:-$HOME/.openclaw/memory/main.sqlite}"

# Colors
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✅ $1${NC}"; }
warn() { echo -e "${YELLOW}⚠️  $1${NC}"; }
fail() { echo -e "${RED}❌ $1${NC}"; exit 1; }

echo "🧠 QMDZvec First Run — Three-Speed Memory for OpenClaw"
echo "======================================================="
echo ""

# ── 1. Prerequisites ──────────────────────────────────────
echo "📋 Checking prerequisites..."

# Python 3.10+
PYTHON=""
for p in python3.10 python3.11 python3.12 python3; do
    if command -v "$p" &>/dev/null; then
        ver=$("$p" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON="$p"
            break
        fi
    fi
done
[ -z "$PYTHON" ] && fail "Python 3.10+ required. On macOS: brew install python@3.12 | On Ubuntu: apt install python3.10"
ok "Python: $PYTHON ($ver)"

# pip
"$PYTHON" -m pip --version &>/dev/null || fail "pip not available for $PYTHON"
ok "pip available"

# OpenClaw workspace
[ -d "$WORKSPACE" ] || warn "OpenClaw workspace not found at $WORKSPACE (will create QMD locally)"

# ── 2. Install Python Dependencies ───────────────────────
echo ""
echo "📦 Installing Python dependencies..."
"$PYTHON" -m pip install -q zvec numpy 2>&1 | tail -1
ok "zvec + numpy installed"

# ── 3. Create QMD Directory ──────────────────────────────
echo ""
echo "📂 Setting up QMD working memory..."
QMD_DIR="$WORKSPACE/memory/qmd"
mkdir -p "$QMD_DIR"
if [ ! -f "$QMD_DIR/current.json" ]; then
    cat > "$QMD_DIR/current.json" << 'EOF'
{
  "session_id": "initial",
  "tasks": [],
  "entities_seen": {},
  "decisions": [],
  "updated_at": ""
}
EOF
    ok "Created $QMD_DIR/current.json"
else
    ok "QMD already exists"
fi

# ── 4. Create Zvec data directory ────────────────────────
mkdir -p "$HOME/.openclaw/zvec-memory"

# ── 5. Stop existing Zvec server if running ──────────────
if curl -sf "http://localhost:$ZVEC_PORT/health" &>/dev/null; then
    warn "Zvec already running on port $ZVEC_PORT — reusing"
    SERVER_ALREADY_RUNNING=1
else
    SERVER_ALREADY_RUNNING=0
fi

# ── 6. Start Zvec Server ────────────────────────────────
if [ "$SERVER_ALREADY_RUNNING" = "0" ]; then
    echo ""
    echo "🚀 Starting Zvec server on port $ZVEC_PORT..."
    cd "$REPO_DIR"
    nohup "$PYTHON" memclawz_server/server.py > /tmp/zvec-server.log 2>&1 &
    ZVEC_PID=$!
    # Wait for server
    for i in $(seq 1 15); do
        if curl -sf "http://localhost:$ZVEC_PORT/health" &>/dev/null; then
            ok "Zvec server running (PID $ZVEC_PID)"
            break
        fi
        sleep 1
    done
    curl -sf "http://localhost:$ZVEC_PORT/health" &>/dev/null || fail "Server failed to start. Check /tmp/zvec-server.log"
fi

# ── 7. Full History Import ───────────────────────────────
echo ""
echo "📦 Importing ALL existing memory into Zvec..."
bash "$SCRIPT_DIR/bootstrap-history.sh"

# ── 8. Start Watcher ────────────────────────────────────
echo ""
echo "👁️  Starting auto-indexing watcher..."
# Kill existing watcher if any
pkill -f "memclawz_server/watcher.py" 2>/dev/null || true
cd "$REPO_DIR"
nohup "$PYTHON" memclawz_server/watcher.py > /tmp/zvec-watcher.log 2>&1 &
ok "Watcher running (PID $!)"

# ── 9. Verify ───────────────────────────────────────────
echo ""
echo "🔍 Running verification..."
cd "$REPO_DIR"
"$PYTHON" scripts/verify.py

# ── 10. Register as OpenClaw Skill ──────────────────────
echo ""
SKILL_DIR="$WORKSPACE/skills/qmd-zvec"
if [ ! -d "$SKILL_DIR" ]; then
    mkdir -p "$SKILL_DIR"
    ln -sf "$REPO_DIR/skill/SKILL.md" "$SKILL_DIR/SKILL.md"
    ok "Registered as OpenClaw skill at $SKILL_DIR"
else
    ok "Skill already registered"
fi

# ── 11. Summary ─────────────────────────────────────────
echo ""
echo "======================================================="
echo "🎉 QMDZvec is fully operational!"
echo ""
STATS=$(curl -sf "http://localhost:$ZVEC_PORT/stats" 2>/dev/null || echo '{}')
TOTAL=$(echo "$STATS" | "$PYTHON" -c "import sys,json; print(json.load(sys.stdin).get('total_docs',0))" 2>/dev/null || echo "?")
echo "  📊 Indexed chunks: $TOTAL"
echo "  🔍 Search: POST http://localhost:$ZVEC_PORT/search"
echo "  🧠 QMD: $QMD_DIR/current.json"
echo "  📝 Logs: /tmp/zvec-server.log, /tmp/zvec-watcher.log"
echo ""
echo "Next steps:"
echo "  1. Add the QMDZvec protocol to your AGENTS.md (see skill/SKILL.md)"
echo "  2. Restart your OpenClaw gateway to pick up the new skill"
echo "  3. Your agent now has 50x faster memory search! 🚀"
echo ""
