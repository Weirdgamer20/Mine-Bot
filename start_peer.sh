#!/usr/bin/env bash
set -e

PORT="${1:-53141}"
AGENT_ID="${2:-LB-01}"
USERNAME="${3:-LB01}"

export PATH="/home/tg/.local/bin:$PATH"
export MC_PORT="$PORT"
export MC_AGENT_ID="$AGENT_ID"
export MC_USERNAME="$USERNAME"

cd /mnt/d/minecraft_learning_bot/bridge
exec node bridge.js "$PORT"
