#!/bin/bash
# QMDZvec — One-command install & setup
# Delegates to the comprehensive first-run script.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "🧠 Installing QMDZvec — Three-Speed Memory for OpenClaw"
echo ""

# Run the full first-run setup
bash "$REPO_DIR/scripts/first-run.sh"
