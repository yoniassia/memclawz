# Fleet Memory — Cross-Agent Memory Sharing

> When multiple OpenClaw agents share a brain, magic happens.

## Overview

Fleet Memory lets a "Claw Fleet" — multiple OpenClaw agents — share and search each other's memories through a central Zvec server with namespaced collections.

```
┌──────────┐  ┌──────────┐  ┌──────────┐
│ YoniClaw │  │ Clawdet  │  │ WhiteRab │
│ Agent 1  │  │ Agent 2  │  │ Agent 3  │
└────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │             │
     └─────────────┼─────────────┘
                   │
          ┌────────▼────────┐
          │  Shared Zvec    │
          │  Fleet Memory   │
          │  (Central Hub)  │
          └─────────────────┘
```

Each agent gets its own **namespace** — an isolated collection in Zvec. Agents can:
- Index memories into their own namespace
- Search their own namespace, another agent's namespace, or all namespaces at once
- Mark memories as **shared** (visible to fleet) or **private** (agent-only)

## Architecture Options

### 1. Shared Zvec Server (Recommended)

One Fleet Memory server runs centrally. All agents connect via HTTP.

```bash
# On the central server
python3.10 zvec/fleet_server.py --port 4011 --api-key fleet-secret-123

# From any agent
curl -X POST http://fleet-server:4011/index \
  -H 'X-API-Key: fleet-secret-123' \
  -H 'Content-Type: application/json' \
  -d '{"namespace": "yoniclaw", "docs": [...]}'
```

**Pros:** Simple, single source of truth, real-time search  
**Cons:** Single point of failure, requires network access

### 2. QMD Sync via Git

Agents push their `memory/qmd/current.json` to a shared git repo. Each agent pulls on session start.

```bash
# Agent pushes its QMD
cd shared-memory-repo
cp memory/qmd/current.json agents/yoniclaw/qmd.json
git add -A && git commit -m "YoniClaw QMD update" && git push

# Other agent pulls
git pull
cat agents/yoniclaw/qmd.json  # See YoniClaw's active tasks
```

**Pros:** Works offline, git history, no server needed  
**Cons:** Not real-time, merge conflicts possible

### 3. Namespaced Collections

Each agent has its own namespace in the Fleet Memory server but can query others.

```bash
# YoniClaw indexes into its namespace
curl -X POST http://fleet:4011/index \
  -d '{"namespace": "yoniclaw", "docs": [{"id": "insight-1", "text": "BTC showing bullish divergence", "shared": true, ...}]}'

# Clawdet searches YoniClaw's namespace
curl -X POST http://fleet:4011/search \
  -d '{"namespace": "yoniclaw", "embedding": [...], "topk": 5}'

# Fleet-wide search across ALL agents
curl -X POST http://fleet:4011/search \
  -d '{"namespace": "all", "embedding": [...], "topk": 10}'
```

### 4. Fleet-Wide Search

Query that fans out across all agent memories and returns the best results:

```bash
# "What does anyone know about the Limassol hotel market?"
curl -X POST http://fleet:4011/search \
  -H 'X-API-Key: fleet-secret-123' \
  -d '{
    "namespace": "all",
    "embedding": [... embedding of "Limassol hotel market" ...],
    "topk": 10
  }'

# Response includes results from all agents with namespace attribution:
# {"results": [
#   {"id": "...", "text": "Four Seasons Limassol: €1360/night", "namespace": "yoniclaw", "score": 0.92},
#   {"id": "...", "text": "Amara Hotel availability checked", "namespace": "clawdet", "score": 0.87}
# ]}
```

### 5. Privacy Controls

Each indexed document can be marked `shared` or `private`:

```json
{
  "id": "private-note-1",
  "text": "Personal API key for...",
  "shared": false
}
```

When searching with `shared_only: true`, private memories are excluded:

```bash
curl -X POST http://fleet:4011/search \
  -d '{"namespace": "all", "embedding": [...], "shared_only": true}'
```

## Concrete Examples

### Example 1: Market Insight Sharing

YoniClaw discovers a market insight during research:

```bash
# YoniClaw indexes the insight
curl -X POST http://fleet:4011/index -d '{
  "namespace": "yoniclaw",
  "docs": [{"id": "market-btc-2026-02", "embedding": [...],
    "text": "Bitcoin showing bullish divergence on weekly chart. RSI oversold at 28.",
    "shared": true}]
}'
```

Later, Clawdet is preparing a portfolio review:

```bash
# Clawdet searches fleet memory for crypto insights
curl -X POST http://fleet:4011/search -d '{
  "namespace": "all",
  "embedding": [... embedding of "cryptocurrency market analysis" ...],
  "topk": 5
}'
# → Returns YoniClaw's BTC insight with namespace="yoniclaw"
```

### Example 2: Bug Discovery Sharing

Clawdet finds a bug in a shared codebase:

```bash
# Clawdet indexes the bug finding
curl -X POST http://fleet:4011/index -d '{
  "namespace": "clawdet",
  "docs": [{"id": "bug-auth-race", "embedding": [...],
    "text": "Race condition in auth middleware: token refresh can fail under concurrent requests. Fix: add mutex on refresh.",
    "shared": true}]
}'
```

YoniClaw later encounters a similar issue:

```bash
# YoniClaw searches fleet memory
curl -X POST http://fleet:4011/search -d '{
  "namespace": "all",
  "embedding": [... embedding of "authentication token error concurrent" ...],
  "topk": 3
}'
# → Finds Clawdet's bug report, avoids re-debugging
```

### Example 3: Fleet-Wide Knowledge Query

Any agent can ask "what does anyone know about X?":

```bash
# WhiteRabbit needs to know about a topic
curl -X POST http://fleet:4011/search -d '{
  "namespace": "all",
  "embedding": [... embedding of "Cyprus tax regulations for crypto" ...],
  "topk": 10
}'
# → Returns relevant memories from ALL agents who've researched this
```

## Fleet Server API

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/namespaces` | List all agent namespaces |
| POST | `/index` | Index docs: `{namespace, docs: [...]}` |
| POST | `/search` | Search: `{namespace, embedding, topk, shared_only}` |

### Authentication

Set `--api-key` when starting the server. All requests must include `X-API-Key` header.

### Running the Fleet Server

```bash
# Basic (no auth)
python3.10 zvec/fleet_server.py --port 4011

# Production (with auth)
python3.10 zvec/fleet_server.py --port 4011 --api-key $FLEET_API_KEY --data /var/lib/fleet-memory

# Custom dimensions
python3.10 zvec/fleet_server.py --dim 768 --port 4011
```

## Setup Guide

### 1. Start Fleet Server

```bash
cd memclawz
python3.10 zvec/fleet_server.py --port 4011 --api-key my-secret &
```

### 2. Configure Each Agent

Add to each agent's environment:
```bash
export FLEET_URL=http://fleet-server:4011
export FLEET_API_KEY=my-secret
export FLEET_NAMESPACE=agent-name  # e.g., yoniclaw, clawdet
```

### 3. Add to Agent Instructions

```markdown
## Fleet Memory Protocol

- On session start: Search fleet memory for relevant recent context
- During work: Index significant findings with `shared: true`
- Keep private: API keys, personal notes → `shared: false`
- Fleet search: Use namespace="all" for cross-agent queries
```

## Future Enhancements

- **WebSocket subscriptions** — Real-time notifications when another agent indexes something relevant
- **Automatic cross-referencing** — When Agent A indexes something similar to Agent B's memory, link them
- **Memory consensus** — When multiple agents have conflicting information, surface the conflict
- **Hierarchical namespaces** — `team/agent` for multi-team setups
- **Memory expiry** — Auto-archive memories older than N days
