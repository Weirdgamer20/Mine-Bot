# Startup Guide

## A. Start the WSL Learning Agent (Persistent Streaming)

From WSL or Windows PowerShell via WSL:

```bash
cd /mnt/d/minecraft_learning_bot/agent
source /mnt/d/minecraft_learning_bot/.venv/bin/activate
PYTHONPATH=. python -m bot.runtime --mode stream --port 9099
```

Expected output:
```text
[StreamServer] INFO: Direct Persistent Streaming Server listening on 0.0.0.0:9099
Learning Agent initialized on device: cuda
```

To run the offline synthetic validation test:
```bash
PYTHONPATH=. python -m bot.runtime --mode synthetic
```

---

## B. Start Minecraft

1. Launch Minecraft Java through TLauncher.
2. Enter your world.
3. Open to LAN (e.g. port `25565` or whatever port Minecraft displays in chat).

---

## C. Configure Bridge

Edit:
`bridge/bridge.js` or set environment variables:
- `MC_HOST`: default `127.0.0.1`
- `MC_PORT`: Minecraft port (e.g. `25565` or LAN port)
- `MC_USERNAME`: `LearningAgent`
- `AGENT_HOST`: `127.0.0.1` (WSL host)
- `AGENT_PORT`: `9099` (persistent TCP stream port)

---

## D. Start the Mineflayer Bridge

Can be run from Windows PowerShell or WSL:

```bash
cd /mnt/d/minecraft_learning_bot/bridge
export PATH=/home/tg/.local/bin:$PATH
node bridge.js
```

Or on Windows:
```powershell
cd d:\minecraft_learning_bot\bridge
node bridge.js
```

The bridge will:
1. Connect directly to the Minecraft Java world.
2. Establish a persistent bidirectional TCP socket stream to the WSL agent (`9099`).
3. Continuously stream numerical observations (`voxels`, `player_state`, `inventory`, `entities`, `affordances`).
4. Receive 12-dimensional continuous/discrete actions and execute them via physical controls.

---

## E. Checkpoints & Persistence

Checkpoints are automatically stored under:
```text
agent/checkpoints/
├── agent_checkpoint.pt
└── replay_buffer.pt
```
All neural network weights, world-model dynamics, RND curiosity parameters, actor-critic policies, and replay sequences survive Minecraft deaths and session restarts.
