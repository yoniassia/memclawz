# memclawz — Three-Speed Memory Skill

> Give your OpenClaw agent structured working memory, fast vector search, and automatic compaction.

## Quick Setup (One Command)

```bash
cd ~/.openclaw/workspace
git clone https://github.com/yoniassia/memclawz.git
cd memclawz && bash scripts/first-run.sh
```

This single command will:
1. Install dependencies (zvec, numpy)
2. Create QMD working memory
3. Start the Zvec server
4. Import ALL existing OpenClaw memory (SQLite + markdown files)
5. Start the auto-indexing watcher
6. Verify everything works
7. Register as an OpenClaw skill

**Re-sync history anytime:** `bash scripts/bootstrap-history.sh`
**Verify installation:** `python3 scripts/verify.py`

## What This Gives You

| Layer | Speed | What |
|-------|-------|------|
| QMD | <1ms | Structured JSON working memory — tasks, decisions, entities |
| Zvec | <10ms | HNSW vector + BM25 keyword hybrid search over all indexed memory |
| MEMORY.md | ~50ms | Curated long-term memory (OpenClaw built-in) |

## Agent Protocol

### On Session Start

1. Read working memory:
```bash
cat memory/qmd/current.json
```

2. Resume awareness of active tasks, recent decisions, and entities.

### During Work

After any significant action (new task, decision, completion), update QMD:

```bash
# Write updated QMD
cat > memory/qmd/current.json << 'QMDEOF'
{
  "session_id": "main-$(date +%Y-%m-%d)",
  "tasks": [...],
  "entities_seen": {...},
  "updated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
QMDEOF
```

### Searching Memory (Recommended)

Use the search script for easy querying — it handles embedding + search in one step:

```bash
bash scripts/search.sh "what did we decide about the API design"
```

This generates an embedding locally, queries Zvec, and prints formatted results (path, score, snippet).

Environment variables: `ZVEC_MODEL` (path to .gguf), `ZVEC_URL` (default localhost:4010), `TOPK` (default 5).

#### Raw API Search

For direct API access with pre-computed embeddings:

```bash
curl -s -X POST http://localhost:4010/search \
  -H 'Content-Type: application/json' \
  -d '{"embedding": [0.1, 0.2, ...], "topk": 5}'

# Response:
# {"results": [{"id": "...", "text": "...", "score": 0.95, "path": "..."}], "count": 5}
```

### Indexing New Content

```bash
curl -s -X POST http://localhost:4010/index \
  -H 'Content-Type: application/json' \
  -d '{"docs": [{"id": "unique-id", "embedding": [...], "text": "content", "path": "source.md"}]}'
```

### Running Compaction

Archive completed tasks from QMD to daily log:

```bash
python3.10 scripts/qmd-compact.py
```

### On Session End

Run compaction automatically at the end of each session (or via cron/heartbeat):

```bash
python3.10 scripts/qmd-compact.py --auto
```

The `--auto` flag runs silently — no output unless tasks were actually compacted. Ideal for cron jobs or heartbeat hooks.

### Health Check

```bash
curl -s http://localhost:4010/health
# {"status": "ok", "engine": "zvec", "version": "0.2.0"}
```

## Endpoints Reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/stats` | Collection statistics |
| POST | `/search` | Search `{"embedding": [...], "topk": N}` |
| POST | `/index` | Index `{"docs": [{"id", "embedding", "text", "path"}]}` |
| GET | `/migrate` | One-time SQLite import |

## AGENTS.md Integration

Add this to your agent's `AGENTS.md`:

```markdown
### 🧠 QMD — Quick Memory Dump Protocol

QMD is your Layer 0 working memory. Structured JSON, always loaded first.

- **On session start:** Read `memory/qmd/current.json` to resume awareness
- **During work:** Update QMD after significant actions (new task, decision, completion)
- **For search:** POST to `http://localhost:4010/search` with embedding vector
- **Compaction:** Run `python3.10 scripts/qmd-compact.py` to archive done tasks
```

## File Watcher (Direct Indexing)

For auto-indexing `.md` files without SQLite:

```bash
python3.10 zvec/file_watcher.py --dirs memory/ knowledge/ --watch MEMORY.md
```

This watches directories and files, chunks markdown by heading, and indexes into Zvec automatically.

## Fleet Memory (Multi-Agent)

For sharing memory across multiple OpenClaw agents, see [Fleet Memory docs](../docs/fleet-memory.md) and run:

```bash
python3.10 zvec/fleet_server.py --port 4011
```
