# M10 — Four Equal Autonomous Minecraft Agents

## Objective

Run four **equal peer agents** in parallel:

```text
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│    LB-01     │    │    LB-02     │    │    LB-03     │    │    LB-04     │
│              │    │              │    │              │    │              │
│   EXPLORER   │    │   SURVIVOR   │    │    WARRIOR   │    │  OPPORTUNIST │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
       │                   │                   │                   │
       └───────────────────┴───────────────────┴───────────────────┘
                                   │
                         SHARED LEARNING SYSTEM
```

The four bots have equal authority and equal architectural status. There is no
leader/follower relationship.

## Shared vs. individual state

### Shared

- RSSM/world-model parameters
- encoder parameters
- RND predictor/target
- actor/critic parameters
- shared replay
- spatial memory
- experience graph
- skill library
- optimizers
- checkpoints

### Per-agent

- agent ID
- personality
- recurrent latent state
- current skill
- episode state
- action history
- local cognitive state

Death resets the per-agent episode/recurrent state while the shared learned
parameters and experience remain available.

## Personalities

### LB-01 — EXPLORER

Behavioral constitution:

- very high curiosity
- high experimentation
- high novelty tolerance
- moderate risk tolerance
- high adaptability

The Explorer is biased toward discovering unfamiliar states. It is not given a
hard-coded list of Minecraft objectives.

### LB-02 — SURVIVOR

Behavioral constitution:

- low risk tolerance
- high persistence
- high preservation preference
- controlled experimentation
- high adaptability

The Survivor is biased toward maintaining a continuing learning episode and
reducing uncertainty about dangerous states.

### LB-03 — WARRIOR

Behavioral constitution:

- maximum confrontation preference
- zero avoidance preference
- maximum persistence
- maximum risk tolerance
- high experimentation

The Warrior's explicit personality rule is confrontation: when Minecraft exposes
an immediately attackable entity, LB-03 confronts it rather than selecting
avoidance. The actual combat technique remains learned.

### LB-04 — OPPORTUNIST

Behavioral constitution:

- maximum exploitation preference
- maximum adaptability
- moderate experimentation
- moderate/high risk tolerance
- balanced curiosity

The Opportunist is biased toward exploiting situations that the learned system
estimates as advantageous.

## Runtime topology

Each bridge process represents one Minecraft peer:

```text
Minecraft/TLauncher
      │
      ├── LB01 Mineflayer ──┐
      ├── LB02 Mineflayer ──┤
      ├── LB03 Mineflayer ──┼── TCP :9099 ──► Shared Learner
      └── LB04 Mineflayer ──┘
```

The Python runtime accepts multiple persistent TCP clients on the same port.
Each HELLO frame contains `agent_id`.

## Windows PowerShell launch

Start the shared learner once:

```powershell
wsl bash -c "cd /mnt/d/minecraft_learning_bot/agent && PYTHONPATH=.:.. /mnt/d/minecraft_learning_bot/.venv/bin/python -m bot.runtime --mode stream --port 9099"
```

Then start four bridge processes. Each should run in its own PowerShell window:

```powershell
$env:MC_AGENT_ID="LB-01"; $env:MC_USERNAME="LB01"; cd D:\minecraft_learning_bot\bridge; node bridge.js
```

```powershell
$env:MC_AGENT_ID="LB-02"; $env:MC_USERNAME="LB02"; cd D:\minecraft_learning_bot\bridge; node bridge.js
```

```powershell
$env:MC_AGENT_ID="LB-03"; $env:MC_USERNAME="LB03"; cd D:\minecraft_learning_bot\bridge; node bridge.js
```

```powershell
$env:MC_AGENT_ID="LB-04"; $env:MC_USERNAME="LB04"; cd D:\minecraft_learning_bot\bridge; node bridge.js
```

The usernames are deliberately `LB01`...`LB04` because Minecraft player
usernames cannot use the hyphen in the agent IDs.

## Verification

The shared server should report four equal peers:

```text
Equal peers: LB-01, LB-02, LB-03, LB-04
```

Each bridge should report:

```text
Agent ID: LB-0X
...
[Agent] LB-0X registered as <PERSONALITY> peer.
```

The first milestone is not "four bots survive." It is:

1. all four connect concurrently;
2. each receives observations and returns actions;
3. each maintains independent recurrent/episode state;
4. all four contribute to one replay/learner;
5. personality-conditioned behavior is visible in telemetry;
6. LB-03 confronts attackable threats rather than avoiding them;
7. one agent dying does not terminate the other three;
8. restarting the runtime restores shared model state.

## Scaling rule

The architecture is intentionally written as a peer registry. The current
configured population is four. Increasing the population later should not
require introducing a leader or changing the hierarchy.

## Performance note

This first M10 implementation uses one shared learner to avoid creating four
copies of the GPU model on the RTX 3050. The four environment streams are
concurrent, while the shared learner processes their experiences. Batched actor
inference and asynchronous learner execution are subsequent optimization steps.
