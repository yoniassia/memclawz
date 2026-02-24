#!/usr/bin/env bash
# search.sh — Agent-friendly memory search via QMD/Zvec
# Usage: bash scripts/search.sh "your query text"
set -euo pipefail

QUERY="${1:?Usage: search.sh <query>}"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${ZVEC_MODEL:-$HOME/.openclaw/models/nomic-embed-text-v1.5.Q8_0.gguf}"
ZVEC_URL="${ZVEC_URL:-http://localhost:4010}"
TOPK="${TOPK:-5}"

# Generate embedding via node
EMBEDDING=$(node "$SCRIPT_DIR/memclawz_server/_embed_node.mjs" "$MODEL" "$QUERY")

if [ -z "$EMBEDDING" ] || [ "$EMBEDDING" = "[]" ]; then
  echo "Error: Failed to generate embedding" >&2
  exit 1
fi

# Search Zvec
RESPONSE=$(curl -sf -X POST "$ZVEC_URL/search" \
  -H 'Content-Type: application/json' \
  -d "{\"embedding\": $EMBEDDING, \"topk\": $TOPK}")

# Format results
echo "$RESPONSE" | python3 -c "
import json, sys
data = json.load(sys.stdin)
results = data.get('results', [])
if not results:
    print('No results found.')
    sys.exit(0)
for i, r in enumerate(results, 1):
    score = r.get('score', 0)
    path = r.get('path', '?')
    text = r.get('text', '')[:200].replace('\n', ' ')
    print(f'{i}. [{score:.3f}] {path}')
    print(f'   {text}')
    print()
"
