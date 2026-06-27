#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/root/tennis_signals"
AGENT_DIR="$BASE_DIR/enrichment_agent"

cd "$AGENT_DIR"
docker compose run --rm enrichment-agent
