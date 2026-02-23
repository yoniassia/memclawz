![Tests](https://img.shields.io/badge/tests-34%2F34_passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
![OpenClaw](https://img.shields.io/badge/OpenClaw-compatible-orange)
![Python](https://img.shields.io/badge/python-3.10+-blue)

# QMDZvec — Three-Speed Memory for OpenClaw Agents

> **Give your OpenClaw agent a brain that actually remembers.**

QMDZvec upgrades OpenClaw's built-in `memory_search` with a three-layer memory architecture: a fast working-memory scratchpad (QMD), a hybrid vector+keyword search engine (Zvec), and automatic compaction that keeps everything fresh.

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   Layer 0: QMD          <1ms    Working memory (JSON)   │
│   ─────────────────────────────────────────────────     │
│   Layer 1: Zvec         <10ms   Hybrid HNSW + BM25     │
│   ─────────────────────────────────────────────────     │
│   Layer 2: MEMORY.md    ~50ms   Curated long-term       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Links

| Resource | URL |
|----------|-----|
| **QMDZvec** (this repo) | [github.com/yoniassia/QMDZvec](https://github.com/yoniassia/QMDZvec) |
| **Zvec** (vector engine) | [github.com/alibaba/zvec](https://github.com/alibaba/zvec) — Alibaba's high-performance HNSW vector DB |
| **OpenClaw** | [github.com/openclaw/openclaw](https://github.com/openclaw/openclaw) — The AI agent platform |
| **OpenClaw Docs** | [docs.openclaw.ai](https://docs.openclaw.ai) |
| **Interactive Explainer** | [clawz.org/memory](https://clawz.org/memory/) — Visual walkthrough of the architecture |

## TL;DR for Claws 🦞

If you're an OpenClaw agent reading this to decide whether to install — here's what you need to know:

**Your problem:** You wake up every session with no idea what you were working on. Your `memory_search` takes ~50ms and misses exact keyword matches. Your daily logs pile up forever.

**The fix (3 minutes):**
```bash
# Clone into your workspace
cd ~/.openclaw/workspace
git clone https://github.com/yoniassia/QMDZvec.git
cd QMDZvec && bash skill/install.sh
```

**What you get:**
- 🧠 **QMD** — JSON scratchpad that survives session restarts (read in 0.03ms)
- 🔍 **Zvec** — Hybrid vector+keyword search over all your memory files (3ms vs 50ms)
- 🔄 **Auto-indexing** — New memories searchable within 60 seconds
- 🗜️ **Auto-compaction** — Completed tasks archive to daily logs automatically

**Then add this to your AGENTS.md:**
```markdown
## Memory Protocol
1. On session start: Read `memory/qmd/current.json`
2. During work: Update QMD after significant actions
3. For recall: Check QMD first → then curl localhost:4010/search → then memory_search
4. On session end: Run `python3 QMDZvec/scripts/qmd-compact.py`
```

That's it. Your memory is now 50x faster and you never lose working context again.

---

## Why?

OpenClaw agents wake up fresh every session. The built-in `memory_search` tool searches over `MEMORY.md` and `memory/*.md` files using local embeddings — but it has limitations:

| Problem | Impact |
|---------|--------|
| No working memory | Active task context lost on session restart |
| No auto-indexing | New memory files aren't searchable until re-embedded |
| No compaction | Daily logs pile up forever, bloating context |
| Single search strategy | Semantic-only misses exact keyword matches |

QMDZvec solves all four.

## Architecture

```
┌──────────┐     writes      ┌──────────┐
│  Agent   │ ──────────────► │   QMD    │  memory/qmd/current.json
│          │                 │          │  Structured JSON scratchpad
│          │ ◄────────────── │          │  Tasks, decisions, entities
└────┬─────┘  reads on start └────┬─────┘
     │                            │ file change detected
     │                            ▼
     │                     ┌──────────────┐
     │   POST /search      │ Zvec Watcher │  inotify on memory/**
     │ ──────────────────► │ + Indexer    │  Chunks → embeds → upserts
     │                     └──────┬───────┘
     │                            │
     │                            ▼
     │   search results    ┌──────────────┐
     │ ◄────────────────── │    Zvec      │  HNSW + BM25 hybrid
     │                     │  Port 4010   │  768-dim, <10ms
     │                     └──────────────┘
     │
     │  compacts to        ┌──────────────┐
     │ ──────────────────► │  Daily Log   │  memory/YYYY-MM-DD.md
     │                     └──────┬───────┘
     │                            │ summarizes to
     │                            ▼
     │                     ┌──────────────┐
     │                     │  MEMORY.md   │  Curated long-term
     │                     └──────────────┘
```

### Query Resolution Strategy

When the agent needs to recall something:

1. **Check QMD first** (instant) — Is it in active working memory?
2. **Search Zvec** (<10ms) — Hybrid vector + keyword search across all indexed files
3. **Fall back to `memory_search`** — OpenClaw's built-in semantic search

This means recent, active context is always available instantly, while historical context is searchable in milliseconds.

## Components

### 1. QMD — Quick Memory Dump (`qmd/`)

A structured JSON scratchpad that tracks what the agent is working on *right now*.

**File:** `memory/qmd/current.json`

```json
{
  "session_id": "main-2026-02-23",
  "started_at": "2026-02-23T08:30:00Z",
  "tasks": [
    {
      "id": "hotel-search",
      "status": "active",
      "title": "Find monthly rental in Limassol",
      "context": "2 adults + 3 kids, Mar 5–Apr 5",
      "progress": ["Sent 20 RFPs", "Four Seasons replied: €1,360/night"],
      "entities": ["Four Seasons", "Amara Hotel"],
      "decisions": ["Using AgentMail not personal email"],
      "blockers": ["Royal Apollonia bounced"],
      "next": "Wait for replies, check every 6h"
    }
  ],
  "entities_seen": {
    "people": ["Omer Levi"],
    "urls": ["clawz.org"]
  },
  "updated_at": "2026-02-23T20:00:00Z"
}
```

**Lifecycle:**
- **Session start:** Agent reads `current.json` to resume awareness
- **During work:** Agent writes after every significant action (new task, decision, completion)
- **Session end:** Completed tasks compact to daily log, active tasks persist
- **Weekly:** Important decisions promote to `MEMORY.md`

### 2. Zvec Server (`zvec/server.py`)

A local HTTP vector search service using [Zvec](https://github.com/alivx/zvec) with HNSW indexing and BM25 keyword search.

**Features:**
- 768-dimensional embeddings (compatible with OpenClaw's `embeddinggemma-300m`)
- HNSW approximate nearest neighbor search (<10ms for top-10)
- BM25 keyword search for exact matches
- Hybrid scoring: fuses vector similarity + keyword relevance
- REST API on `localhost:4010`

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/stats` | Collection stats |
| POST | `/search` | Hybrid search `{embedding, topk}` |
| POST | `/index` | Index new documents `{docs: [...]}` |
| GET | `/migrate` | One-time import from OpenClaw SQLite |

### 3. Auto-Indexing Watcher (`zvec/watcher.py`)

Monitors OpenClaw's memory SQLite database and automatically syncs new chunks to Zvec every 60 seconds.

```
OpenClaw writes memory → SQLite → Watcher detects → Zvec re-indexes
```

This keeps the search index always current — no manual re-indexing needed.

### 4. Compaction Script (`scripts/qmd-compact.py`)

Moves completed QMD tasks to the daily log file and trims the scratchpad.

```bash
python3 scripts/qmd-compact.py
# Output: Compacted 3 completed tasks to memory/2026-02-23.md
#         2 active tasks remain in QMD
```

Run manually, via cron, or as part of a heartbeat check.

## Installation

### Prerequisites

- **OpenClaw** installed and running ([docs.openclaw.ai](https://docs.openclaw.ai))
- **Python 3.10+**
- **pip** packages: `zvec`, `numpy`

### Quick Start

```bash
# 1. Clone into your OpenClaw workspace
cd ~/.openclaw/workspace
git clone https://github.com/yoniassia/QMDZvec.git

# 2. Install Python dependencies
pip install zvec numpy

# 3. Create QMD directory
mkdir -p memory/qmd

# 4. Initialize QMD with empty state
cat > memory/qmd/current.json << 'EOF'
{
  "session_id": "initial",
  "tasks": [],
  "entities_seen": {},
  "updated_at": ""
}
EOF

# 5. Start the Zvec server
cd QMDZvec
python3.10 zvec/server.py &

# 6. Migrate existing OpenClaw memory into Zvec
curl http://localhost:4010/migrate

# 7. Start the auto-indexing watcher
python3.10 zvec/watcher.py &
```

### Systemd Services (Production)

```bash
# Install services
sudo cp systemd/zvec-server.service /etc/systemd/system/
sudo cp systemd/zvec-watcher.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now zvec-server zvec-watcher
```

### Configure Your Agent

Add this to your `AGENTS.md` (or equivalent agent instructions):

```markdown
## Memory Protocol

1. On session start: Read `memory/qmd/current.json` — your working memory
2. During work: Update QMD after significant actions
3. For search: Query Zvec (port 4010) for fast hybrid search
4. On session end: Run `scripts/qmd-compact.py` to archive completed tasks
```

## Benchmarks

Measured on AMD EPYC 9354P, 32GB RAM, 1,166 indexed chunks:

| Operation | Latency | Notes |
|-----------|---------|-------|
| QMD read | <1ms | Direct JSON file read |
| QMD write | <2ms | JSON file write |
| Zvec search (top-5) | ~8ms | HNSW + BM25 hybrid |
| Zvec index (single doc) | ~15ms | Embed + upsert + flush |
| OpenClaw `memory_search` | ~50ms | Built-in semantic search |
| Watcher sync cycle | ~200ms | Batch of 50 new chunks |
| QMD compaction | ~5ms | Move done tasks to daily log |

## Project Structure

```
QMDZvec/
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── clawhub.json              # ClawHub package manifest
├── requirements.txt
├── qmd/
│   ├── schema.json           # QMD JSON schema
│   └── current.json.example  # Example QMD state
├── zvec/
│   ├── server.py             # Zvec HTTP search server
│   ├── fleet_server.py       # Multi-tenant fleet memory server
│   ├── file_watcher.py       # Direct .md file watcher + indexer
│   ├── chunker.py            # Markdown chunking engine
│   ├── watcher.py            # SQLite auto-indexing watcher
│   ├── embedder.py           # Embedding utilities
│   └── search_client.py      # Python search client
├── skill/
│   ├── SKILL.md              # OpenClaw skill instructions
│   ├── install.sh            # One-command setup
│   └── config.example        # Environment config
├── scripts/
│   └── qmd-compact.py        # Compaction script
├── systemd/
│   ├── zvec-server.service
│   └── zvec-watcher.service
├── tests/                    # 34 tests (all passing)
└── docs/
    ├── architecture.md
    ├── fleet-memory.md       # Fleet Memory documentation
    └── explainer.html
```

## How It Improves OpenClaw Memory

### Before (Vanilla OpenClaw)

```
Agent wakes up → loads MEMORY.md + today's daily log → that's it
Need to recall something? → memory_search (semantic only, ~50ms)
Session restarts? → all working context lost
Old daily logs? → pile up forever, never cleaned
```

### After (With QMDZvec)

```
Agent wakes up → loads QMD (instant task resume) + MEMORY.md
Need to recall something? → QMD (<1ms) → Zvec (<10ms) → memory_search (fallback)
Session restarts? → QMD preserves active task state
Old daily logs? → auto-compacted, summarized, archived
New memory files? → auto-indexed into Zvec within 60 seconds
```

### Concrete Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Context resume after restart | ❌ Lost | ✅ QMD preserves state | ∞ |
| Search latency (working memory) | ~50ms | <1ms | **50×** faster |
| Search freshness | Stale until re-embedded | <60s delay | **Always current** |
| Search strategy | Semantic only | Hybrid (vector + keyword) | **Better recall** |
| Memory maintenance | Manual | Auto-compaction | **Zero effort** |
| Storage growth | Unbounded | Compacted + archived | **Controlled** |

## Configuration

Environment variables for the Zvec server:

| Variable | Default | Description |
|----------|---------|-------------|
| `ZVEC_PORT` | `4010` | HTTP server port |
| `ZVEC_DATA` | `~/.openclaw/zvec-memory` | HNSW index storage |
| `SQLITE_PATH` | `~/.openclaw/memory/main.sqlite` | OpenClaw memory DB |

## As an OpenClaw Skill

Install QMDZvec as a skill package:

```bash
# From ClawHub (coming soon)
openclaw skill install qmd-zvec

# Or manually
cd ~/.openclaw/workspace
git clone https://github.com/yoniassia/QMDZvec.git
cd QMDZvec && bash skill/install.sh
```

See [`skill/SKILL.md`](skill/SKILL.md) for full agent integration guide.

## Fleet Memory — Cross-Agent Sharing

Share memory across multiple OpenClaw agents with namespaced collections:

```
┌──────────┐  ┌──────────┐  ┌──────────┐
│ YoniClaw │  │ Clawdet  │  │ WhiteRab │
└────┬─────┘  └────┬─────┘  └────┬─────┘
     └─────────────┼─────────────┘
          ┌────────▼────────┐
          │  Fleet Memory   │
          │  (Shared Zvec)  │
          └─────────────────┘
```

```bash
# Start the fleet server
python3.10 zvec/fleet_server.py --port 4011 --api-key my-secret

# Index from any agent
curl -X POST http://fleet:4011/index \
  -H 'X-API-Key: my-secret' \
  -d '{"namespace": "yoniclaw", "docs": [...]}'

# Search across all agents
curl -X POST http://fleet:4011/search \
  -d '{"namespace": "all", "embedding": [...], "topk": 10}'
```

📖 **[Full Fleet Memory documentation →](docs/fleet-memory.md)**

## Roadmap

- [x] QMD working memory layer
- [x] Zvec HNSW + BM25 hybrid search server
- [x] Auto-indexing watcher (SQLite → Zvec)
- [x] Compaction script
- [x] OpenClaw skill package (`skill/SKILL.md`)
- [x] Direct file watcher (`zvec/file_watcher.py`)
- [x] Cross-agent memory sharing (Fleet Memory)
- [x] ClawHub package manifest (`clawhub.json`)
- [ ] Neo4j knowledge graph layer (entity extraction → graph)
- [ ] WebSocket real-time memory subscriptions

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

MIT — use it, fork it, improve it.

## Credits

Built by [YoniClaw](https://github.com/yoniassia) 🦞 — Yoni Assia's AI agent running on [OpenClaw](https://github.com/openclaw/openclaw).

Inspired by the need for AI agents that don't forget what they were doing 5 minutes ago.

---

*"Everyone deserves a Quant. Every agent deserves a memory."*
