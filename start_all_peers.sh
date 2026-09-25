#!/usr/bin/env bash
PORT="${1:-53141}"

pkill -f "node bridge.js" 2>/dev/null || true
sleep 1

export PATH="/home/tg/.local/bin:$PATH"
cd /mnt/d/minecraft_learning_bot/bridge

echo "Starting LB-01..."
MC_PORT="$PORT" MC_AGENT_ID="LB-01" MC_USERNAME="LB01" nohup node bridge.js "$PORT" > /mnt/d/minecraft_learning_bot/lb01.log 2>&1 &

echo "Starting LB-02..."
MC_PORT="$PORT" MC_AGENT_ID="LB-02" MC_USERNAME="LB02" nohup node bridge.js "$PORT" > /mnt/d/minecraft_learning_bot/lb02.log 2>&1 &

echo "Starting LB-03..."
MC_PORT="$PORT" MC_AGENT_ID="LB-03" MC_USERNAME="LB03" nohup node bridge.js "$PORT" > /mnt/d/minecraft_learning_bot/lb03.log 2>&1 &

echo "Starting LB-04..."
MC_PORT="$PORT" MC_AGENT_ID="LB-04" MC_USERNAME="LB04" nohup node bridge.js "$PORT" > /mnt/d/minecraft_learning_bot/lb04.log 2>&1 &

echo "All 4 peers started successfully for LAN port $PORT!"
