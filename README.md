# Mine-Bot — Autonomous Minecraft Learning System

## Objective

```text
PLAY → LEARN → LIVE → GROW
```

An autonomous, general embodied learning agent operating inside Minecraft Java Edition. The system learns survival, exploration, interaction, resource relationships, crafting, combat, building, and long-term competence through experience rather than hard-coded strategy guides.

---

## The Core Architectural Principle

```text
MECHANICS  →  Provided by Minecraft environment (registry, affordances, consequences)
STRATEGY   →  Discovered & learned by the neural agent (representations, world model, skills, planning)
```

- **No external AI APIs or LLMs.**
- **No computer-vision or screenshot-based perception.**
- **Structured numerical observations**: 3D local voxels ($11 \times 11 \times 11$), kinematics, 41 inventory slots, 16 tracked entities, and mechanical affordances.
- **Rulebook, Not Strategy Guide**: Minecraft provides deterministic facts (block IDs, recipes, mechanics, raycasts). The agent discovers the utility, value, and causal consequences of those facts.
- **Persistence Across Deaths & Sessions**: Deaths terminate an episode ($h_0 \leftarrow 0$), but neural parameters, world models, episodic replay memory, skills, and spatial embeddings persist.
- **Direct 16-Byte Framed TCP Stream**: Zero HTTP/REST dependencies. Uses a persistent framed binary protocol with CRC32 integrity validation on port 9099.

---

## 9-Phase Engineering Specification

| Phase | System Component | Implementation Deliverables |
|---|---|---|
| **Phase 1: Environment Contract v1** | Registry & Protocol | Canonical registry (1,058 blocks, 1,312 items, 126 entities, 64 biomes, 2,409 recipes, universal dictionary), version manifest, 16-byte binary framing protocol with CRC32, HELLO/WELCOME handshake, PING/PONG heartbeats. |
| **Phase 2: Real Minecraft Actuation** | Mechanical Bridge | Continuous motor locomotion (7-dim) + discrete primitives (27 actions across Tiers 1–9), inventory hotbar/equip/swap, recipe crafting, chest/container manipulation, furnace smelting, bed sleep, and switches. Integrated UDP LAN auto-discovery (`224.0.2.0:4445`) for TLauncher. |
| **Phase 3: Experience & Memory** | Persistent Memory | Prioritized sequence replay buffer ($P \propto (|TD| + |WM| + Novelty)^\alpha$), spatial memory mapping chunk embeddings & visit counts, and topological experience graph linking state hashes to observed consequences. |
| **Phase 4: World Model & Latent Dynamics** | Recurrent State-Space Model | RSSM with deterministic GRU state ($h_t$) and stochastic categorical latent priors/posteriors ($z_t$). Consequence predictor (health, food, position, inventory deltas), continuation predictor ($c_t \in [0, 1]$), and latent imagination rollouts. |
| **Phase 5: Exploration & Curiosity** | Intrinsic Motivation | Random Network Distillation (RND) fixed target vs predictor, transition novelty, prediction-error reward, and information-gain seeking. |
| **Phase 6: Unsupervised Temporal Skills** | DIAYN Skill Discovery | Latent skill conditioning ($z_s$), DIAYN mutual information discriminator maximizing $I(Z; S)$, skill termination classifier $\beta(s, z_s)$, and persistent skill library. |
| **Phase 7: Hierarchical Latent Planning** | Latent MPC | Model Predictive Control simulating candidate action trajectories entirely inside the learned latent RSSM prior dynamics, evaluating predicted value and survival before stepping in the real world. |
| **Phase 8: Lifelong Autonomous Learning** | Async Actor / Learner | Decoupled non-blocking PyTorch learner thread, atomic checkpointing (`.tmp` write + atomic replace), versioned parameter synchronization, and automatic crash recovery across restarts. |
| **Phase 9: Scientific Evaluation & Proof** | Benchmarks & Ablations | Benchmark suite comparing Random, Untrained, Reactive, and Full Model-based planning agents across MSE, KL, survival steps, and skill diversity. Systematic ablation suite testing component contributions. |\n| **Phase 10: Four Equal Agents** | Multi-Agent Learning | LB-01 Explorer, LB-02 Survivor, LB-03 Warrior, LB-04 Opportunist as equal peers; independent recurrent/episode state; shared world model, replay, skills, and learner; concurrent Mineflayer TCP streams. |

---

## Repository Structure

```text
mine-bot/
├── agent/
│   ├── bot/
│   │   ├── agent.py               # Unified LearningAgent integrating all M1–M9 modules
│   │   ├── config.py              # Hyperparameters & network shapes
│   │   ├── models.py              # MultiModalEncoder, WorldModel (RSSM), ActorCritic
│   │   ├── evaluation/            # Phase 9: Scientific benchmark & ablation suite
│   │   │   ├── benchmark.py
│   │   │   ├── ablations.py
│   │   │   └── metrics.py
│   │   ├── memory/                # Phase 3: Persistent replay & spatial memory
│   │   │   ├── replay.py          # Prioritized sequence replay buffer
│   │   │   ├── spatial.py         # Spatial chunk memory
│   │   │   └── experience_graph.py# State-action-consequence topological graph
│   │   ├── planning/              # Phase 7: Latent MPC planner
│   │   │   └── latent_planner.py
│   │   ├── protocol/              # Phase 1: 16-byte binary framing with CRC32
│   │   │   ├── framing.py
│   │   │   ├── messages.py
│   │   │   └── stream_server.py
│   │   ├── skills/                # Phase 6: Unsupervised DIAYN temporal skills
│   │   │   ├── discovery.py       # DIAYN mutual information discriminator
│   │   │   ├── termination.py     # Skill termination model
│   │   │   └── library.py         # Persistent skill library
│   │   └── training/              # Phase 8: Lifelong learning & atomic checkpoints
│   │       ├── checkpoint.py      # AtomicCheckpointManager
│   │       └── learner.py         # AsyncLearnerThread
│   └── tests/
│       └── test_stream.py         # End-to-end framed streaming verification test
├── bridge/
│   ├── bridge.js                  # Main Mineflayer adapter with TCP stream & LAN discovery
│   ├── minecraft.js               # Minecraft client initialization & UDP LAN listener
│   ├── protocol.js                # 16-byte binary framed protocol encoder/decoder
│   ├── actions.js                 # Hierarchical action executor
│   ├── inventory.js               # Tier 4: Inventory management
│   ├── crafting.js                # Tier 5: Recipe transformation
│   ├── containers.js              # Tiers 6 & 7: Chests & furnace smelting
│   └── mechanics.js               # Tiers 2, 3, 8 & 9: Blocks, combat, sleep, switches
├── environment/
│   ├── manifest.py                # Version manifest & protocol IDs
│   ├── registry/                  # Canonical registry JSONs & loader
│   └── schema/                    # Action, Observation & Transition schemas
└── docs/
    ├── ARCHITECTURE.md
    ├── OBJECTIVE.md
    └── STARTUP.md
```

---

## Quickstart (TLauncher & WSL2)

### 1. Launch Minecraft World
1. Open **TLauncher** (`C:\Users\thega\AppData\Roaming\.minecraft`), select **Release 1.20.4**, and start a Singleplayer Survival world.
2. In-game: Press **ESC** → **"Open to LAN"** → **"Start LAN World"**.
   *(Mine-Bot automatically listens on UDP multicast `224.0.2.0:4445` to discover the port).*

### 2. Start the WSL Learning Agent
```bash
wsl bash -c "cd /mnt/d/minecraft_learning_bot/agent && PYTHONPATH=.:.. /mnt/d/minecraft_learning_bot/.venv/bin/python -m bot.runtime --mode stream --port 9099"
```

### 3. Start the Bridge
```bash
wsl bash -c "PATH=/home/tg/.local/bin:\$PATH; cd /mnt/d/minecraft_learning_bot/bridge && node bridge.js"
```
*(Or run `node bridge.js` from Windows PowerShell).*

### 4. Run Offline Synthetic Verification
```bash
wsl bash -c "cd /mnt/d/minecraft_learning_bot/agent && PYTHONPATH=.:.. /mnt/d/minecraft_learning_bot/.venv/bin/python -m bot.runtime --mode synthetic"
```

### 5. Run Scientific Benchmark & Ablations
```bash
wsl bash -c "cd /mnt/d/minecraft_learning_bot/agent && PYTHONPATH=.:.. /mnt/d/minecraft_learning_bot/.venv/bin/python -m bot.evaluation.benchmark"
wsl bash -c "cd /mnt/d/minecraft_learning_bot/agent && PYTHONPATH=.:.. /mnt/d/minecraft_learning_bot/.venv/bin/python -m bot.evaluation.ablations"
```


---

## Phase 10 — Four Equal Autonomous Peers

```text
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│    LB-01     │    │    LB-02     │    │    LB-03     │    │    LB-04     │
│   EXPLORER   │    │   SURVIVOR   │    │    WARRIOR   │    │  OPPORTUNIST │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
       │                   │                   │                   │
       └───────────────────┴───────────────────┴───────────────────┘
                                   │
                         SHARED LEARNING SYSTEM
```

All four agents are equal peers. No agent is a leader or subordinate.

See docs/MULTI_AGENT.md for the M10 runtime and startup procedure.
