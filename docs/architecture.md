# memclawz Architecture

## The Problem

OpenClaw agents have a memory problem. They wake up fresh every session with no working memory of what they were doing. The built-in `memory_search` searches over markdown files using local embeddings, but:

1. **No working memory** — Active task state is lost between sessions
2. **Stale search index** — New files aren't searchable until manually re-embedded
3. **No keyword search** — Semantic-only search misses exact matches ("Four Seasons" → might return "Hilton")
4. **No compaction** — Daily logs grow forever

## The Solution: Three-Speed Memory

memclawz introduces a three-layer memory architecture, each optimized for different access patterns:

### Layer 0: QMD (Quick Memory Dump) — <1ms

A structured JSON file (`memory/qmd/current.json`) that acts as the agent's working memory.

**What it stores:**
- Active tasks with progress, decisions, and blockers
- Entity tracking (people, URLs, credentials mentioned)
- Decision log ("why I chose X over Y")

**Key design decisions:**
- JSON, not markdown — agents can parse it instantly without LLM interpretation
- Max 50KB with FIFO eviction — prevents unbounded growth
- Written *during* work, not after — captures state as it happens
- Completed tasks compact to daily log, not deleted — nothing is lost

### Layer 1: Zvec (Hybrid Vector + Keyword Search) — <10ms

A local HTTP service that combines two search strategies:

**HNSW Vector Search:**
- 768-dimensional embeddings (same model as OpenClaw: `embeddinggemma-300m`)
- Approximate nearest neighbor via HNSW algorithm
- Good for: "Find me context about hotel pricing" → semantic similarity

**BM25 Keyword Search:**
- Classic information retrieval scoring
- Good for: "Find mentions of Four Seasons" → exact term matching

**Hybrid Scoring:**
- Fuses vector similarity + keyword relevance scores
- Configurable weights (default: 70% vector, 30% keyword)
- Reranking ensures best results surface first

### Layer 2: MEMORY.md + memory_search — ~50ms

OpenClaw's built-in memory. memclawz doesn't replace it — it adds faster layers on top.

## Data Flow

### Write Path

```
Agent performs action
    ↓
Updates QMD (current.json)     ← immediate, <2ms
    ↓
OpenClaw writes to SQLite      ← built-in behavior
    ↓
Watcher detects new chunks     ← polls every 60s
    ↓
Watcher sends to Zvec /index   ← embeds + upserts
    ↓
Zvec HNSW index updated        ← searchable in <10ms
```

### Read Path

```
Agent needs to recall something
    ↓
Check QMD (current.json)       ← <1ms, JSON parse
    ├─ Found → return immediately
    ↓
Search Zvec (/search)          ← <10ms, hybrid
    ├─ Found → return results
    ↓
Fall back to memory_search     ← ~50ms, semantic
    └─ Return whatever we have
```

### Compaction Path

```
Session ends or context fills up
    ↓
qmd-compact.py runs
    ↓
Completed tasks → daily log (memory/YYYY-MM-DD.md)
Active tasks → stay in QMD
    ↓
Weekly heartbeat
    ↓
Important decisions → MEMORY.md
Old daily logs (>30 days) → archive/
```

## Why This Matters for OpenClaw

OpenClaw's `memory_search` is a solid foundation — semantic search over markdown files with local embeddings. memclawz extends it by:

1. **Adding a structured working memory layer** that survives session restarts
2. **Keeping the search index fresh** via automatic re-indexing
3. **Adding keyword search** alongside semantic search for better recall
4. **Automating memory maintenance** with compaction and archival

The agent's instructions just need a few lines added to their `AGENTS.md`:

```markdown
## Memory Protocol
1. On session start: Read `memory/qmd/current.json`
2. During work: Update QMD after significant actions
3. For search: Check QMD first, then Zvec (port 4010), then memory_search
4. On session end: Run `scripts/qmd-compact.py`
```

That's it. The rest is automatic.
