# Architecture

## 1. System Topology

```text
             WINDOWS (or Host)                             WSL2 (Linux)
┌────────────────────────────────────────┐        ┌────────────────────────────────────────┐
│ TLauncher                              │        │ Persistent TCP Streaming Server        │
│    ↓                                   │        │    ↓                                   │
│ Minecraft Java Edition                 │        │ MultiModalObservationEncoder           │
│    ↕                                   │  TCP   │    ↓                                   │
│ Mineflayer Bridge (Node.js)            │ Stream │ Recurrent State Space Model (RSSM)     │
│  - Mechanical observation extractor    │◄──────►│    ↓                                   │
│  - Physical actuator executor          │ 9099   │ Latent Imagination + RND Curiosity     │
│  - ZERO artificial intelligence        │        │    ↓                                   │
│                                        │        │ Actor-Critic Policy Head               │
└────────────────────────────────────────┘        └────────────────────────────────────────┘
```

---

## 2. Hard Boundary: Zero Intelligence in Bridge

The Windows Mineflayer bridge acts strictly as an **I/O translator**:
- Reads game packets and extracts the local 11x11x11 voxel grid, physical kinematic state, inventory slots, nearby entities, and mechanical affordances (e.g. targeted block raycast, light level).
- Sends length-prefixed frames over a persistent TCP socket to port `9099`.
- Receives the 12-dimensional action vector and maps it directly to Mineflayer motor controls:
  - `move_x`, `move_z` -> `setControlState('forward'|'back'|'left'|'right')`
  - `jump`, `sneak`, `sprint` -> `setControlState('jump'|'sneak'|'sprint')`
  - `yaw_delta`, `pitch_delta` -> `look()`
  - `slot` -> `setQuickBarSlot()`
  - `attack` -> `attack()` / `dig()`
  - `use` -> `activateItem()` / `placeBlock()`
  - `drop` -> `tossStack()`

---

## 3. Direct Persistent Streaming Transport

- Replaces REST APIs to eliminate HTTP header overhead and connection churn.
- Binary length-prefixed protocol:
  - 4 bytes Big-Endian unsigned integer (payload length)
  - UTF-8 JSON payload
- Bidirectional, persistent socket with auto-reconnection and sub-millisecond roundtrips.

---

## 4. Multi-Modal Observation Perception

Observations are encoded using specialized neural heads and fused into a unified latent vector $e_t \in \mathbb{R}^{256}$:
- **Terrain Voxels**: 3D Convolutional Network (`Conv3D`) over block ID embeddings.
- **Kinematic State**: Linear MLP over velocity, position, pitch, yaw, and status flags.
- **Inventory Matrix**: Slot & item embeddings with permutation-invariant pooling.
- **Nearby Entities**: Kinematic features + type embeddings with set pooling.
- **Mechanical Affordances**: Linear projection of targeted block, raycast distance, mineability, and ambient light.

---

## 5. Recurrent World Model (RSSM / DreamerV3 Inspired)

- **Deterministic State**: $h_t = \text{GRU}(h_{t-1}, [z_{t-1}, a_{t-1}])$
- **Posterior Latent**: $z_t \sim q(z_t \mid h_t, e_t)$ (inferred from observation)
- **Prior Dynamics**: $\hat{z}_t \sim p(z_t \mid h_t)$ (predicted without observation)
- **Observation Feature Reconstructor**: $\hat{e}_t = f(h_t, z_t)$
- **Continuation Predictor (Survival)**: $c_t = \sigma(W_c [h_t, z_t]) \in [0, 1]$. Predicts whether the agent survives this transition without dying, enabling the agent to learn the value of living (**LIVE**) through consequences.

---

## 6. Intrinsic Curiosity & Exploration (RND)

- **Random Network Distillation (RND)**:
  - Fixed random projection network $\phi_{\text{target}}(e_t)$
  - Trainable predictor network $\phi_{\text{pred}}(e_t)$
  - Intrinsic reward $r_t^{\text{int}} = \|\phi_{\text{pred}}(e_t) - \phi_{\text{target}}(e_t)\|^2$
- **Epistemic Prediction Error**: Tracks transition uncertainty to reward visiting under-explored environments.

---

## 7. Latent Imagination & Actor-Critic

- Instead of slow game-step rollouts, the policy and value functions are trained using **imagined rollouts entirely within the world model's latent space**:
  - Trajectories are projected forward $H$ steps into the future.
  - Generalized Advantage Estimation (GAE) computes imagined returns using the continuation discount $c_t$ and intrinsic rewards.
  - Policy gradient and value function updates occur continuously in the background.

---

## 8. Persistence

- **State Across Deaths**: A death event signals termination ($c_t = 0$), ending the episode and resetting recurrent hidden states, but preserves the replay buffer, world-model dynamics, curiosity statistics, and network weights.
- **Session Checkpoints**: Stored atomically in `agent/checkpoints/` (`agent_checkpoint.pt` and `replay_buffer.pt`), ensuring full recovery across restarts.
