# Autonomous Minecraft Learning System

## Objective

```text
PLAY → LEARN → LIVE → GROW
```

An autonomous, API-driven Minecraft Java agent designed to learn survival, exploration, interaction, resource gathering, crafting, combat, and progression through experience rather than hard-coded strategy guides.

---

## Core Constraints & Principles

- **No external AI APIs or LLMs.**
- **No computer-vision or screenshot-based perception.**
- **Purely numerical/structured observation space** (3D local voxels, kinematics, inventory slots, entities, and mechanical affordances).
- **Rulebook, Not Strategy Guide**: Minecraft provides deterministic facts (block IDs, recipes, mechanics, raycasts). The agent discovers the utility, value, and causal consequences of those facts.
- **Persistence Across Deaths & Sessions**: Deaths reset episode context, but neural parameters, world models, episodic replay memory, and skills persist across episodes.
- **Direct Persistent Streaming Transport**: Zero HTTP REST dependency. Uses a raw persistent TCP stream with 4-byte length-prefixed binary framing for sub-millisecond bidirectional communication.

---

## Architecture

```text
       Minecraft (Java via TLauncher)
                     ↕ (game protocol)
         Mineflayer Bridge (Node.js)
  [Pure mechanical adapter — zero AI logic]
                     ↕
       Direct Persistent TCP Stream
          (4-byte framed / port 9099)
                     ↕
         WSL2 Learning Agent (PyTorch)
  ┌────────────────────────────────────────┐
  │ 1. MultiModalObservationEncoder        │
  │    (Voxel 3D-CNN + Entity/Inv/Player)  │
  ├────────────────────────────────────────┤
  │ 2. Recurrent World Model (RSSM)        │
  │    - Dynamics: z_t, a_t -> z_(t+1)     │
  │    - Survival predictor: c_t in [0, 1] │
  │    - Epistemic prediction error        │
  ├────────────────────────────────────────┤
  │ 3. Intrinsic RND Curiosity Engine      │
  │    - Fixed random target vs predictor  │
  ├────────────────────────────────────────┤
  │ 4. Actor-Critic Policy (Imagination)   │
  │    - Rollouts in latent latent space   │
  │    - Generalized Advantage Estimation  │
  ├────────────────────────────────────────┤
  │ 5. Trajectory Replay Buffer            │
  │    - Persists sequences to disk        │
  └────────────────────────────────────────┘
```

---

## Repository Structure

```text
minecraft_learning_bot/
├── agent/
│   ├── bot/
│   │   ├── agent.py          # LearningAgent combining perception, world model, AC
│   │   ├── config.py         # Hyperparameters & network shapes
│   │   ├── learning.py       # RSSM loss, RND loss, latent imagination
│   │   ├── memory.py         # Episodic trajectory replay buffer
│   │   ├── models.py         # MultiModalEncoder, WorldModel, RND, ActorCritic
│   │   ├── runtime.py        # CLI entrypoint (--mode stream / --mode synthetic)
│   │   ├── schemas.py        # Observation, Action, Affordances dataclasses
│   │   └── stream_server.py  # Persistent TCP socket streaming server
│   ├── checkpoints/          # Auto-saved model & buffer weights
│   ├── tests/
│   │   └── test_stream.py    # End-to-end socket & tensor verification test
│   └── requirements.txt
├── bridge/
│   ├── bridge.js             # Mineflayer adapter with direct TCP stream
│   └── package.json
├── config/
│   └── default.json
└── docs/
    ├── ARCHITECTURE.md
    ├── OBJECTIVE.md
    └── STARTUP.md
```

---

## Quickstart

### 1. Verify / Test the Agent in WSL

Run the synthetic offline smoke test (tests perception, world-model forward/backward, RND, latent imagination, and GPU training):

```bash
cd /mnt/d/minecraft_learning_bot/agent
source /mnt/d/minecraft_learning_bot/.venv/bin/activate
PYTHONPATH=. python -m bot.runtime --mode synthetic
```

### 2. Start the Persistent Stream Server

```bash
cd /mnt/d/minecraft_learning_bot/agent
source /mnt/d/minecraft_learning_bot/.venv/bin/activate
PYTHONPATH=. python -m bot.runtime --mode stream --port 9099
```

### 3. Launch Minecraft & Connect Bridge

1. Open your Minecraft Java world (e.g. via TLauncher) and Open to LAN (port `25565`).
2. Run the bridge:
   ```bash
   cd /mnt/d/minecraft_learning_bot/bridge
   export PATH=/home/tg/.local/bin:$PATH
   node bridge.js
   ```
   *(Or run `node bridge.js` on Windows PowerShell).*
