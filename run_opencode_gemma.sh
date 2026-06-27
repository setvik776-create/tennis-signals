#!/usr/bin/env bash
set -euo pipefail

cd /root/tennis_signals

MODEL="google/gemma-4-31b-it"
exec /root/.opencode/bin/opencode run --model "$MODEL" --agent build "$@"
