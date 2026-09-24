# Mine-Bot Startup & Operation Guide (TLauncher & WSL2)

## Overview & Architecture

Mine-Bot is an autonomous Minecraft learning system adhering to **PLAY → LEARN → LIVE → GROW**.
- **Agent Runtime**: PyTorch on CUDA inside WSL2 (Ubuntu 26.04) running recurrent world models, DIAYN unsupervised skills, latent MPC planning, and prioritized sequence replay.
- **Environment Bridge**: Node.js Mineflayer bridge that connects to your local Minecraft Java game (TLauncher) via offline mode and streams structured numerical observations and receives hierarchical actions over a 16-byte binary TCP framed socket (`port 9099`).

---

## 1. Prerequisites & Environment Setup

### A. TLauncher Minecraft Directory
- **Path**: `C:\Users\thega\AppData\Roaming\.minecraft`
- **Recommended Version**: Minecraft **1.20.4** Java Edition (Release or OptiFine 1.20.4, Protocol 765).
- **Accounts**: Any offline username in TLauncher (e.g. `Player`). Mineflayer connects with `auth: 'offline'`.

### B. WSL2 Python Environment
The PyTorch agent virtual environment is located at:
```bash
/mnt/d/minecraft_learning_bot/.venv
```
GPU Acceleration: NVIDIA GeForce RTX 3050 Laptop GPU with CUDA is configured and verified.

---

## 2. Launch Steps

### Step 1: Start your Minecraft World in TLauncher
1. Launch **TLauncher** and select **Release 1.20.4** (or OptiFine 1.20.4).
2. Create or load a Singleplayer Survival world.
3. Once in the world, press **ESC** → Click **"Open to LAN"**.
   - Game Mode: Survival
   - Allow Cheats: ON (optional, useful for testing)
   - Click **"Start LAN World"**.
   - Minecraft will broadcast its LAN port on UDP multicast `224.0.2.0:4445`.

> **Note**: Mine-Bot's bridge features automatic LAN discovery! It will detect your open LAN world automatically. If you wish to pin a specific port, set `MC_PORT=<port>`.

---

### Step 2: Start the PyTorch Learning Agent in WSL
Open a PowerShell terminal or WSL terminal:

```bash
wsl bash -c "cd /mnt/d/minecraft_learning_bot/agent && PYTHONPATH=.:.. /mnt/d/minecraft_learning_bot/.venv/bin/python -m bot.runtime --mode stream --port 9099"
```

Expected log output:
```text
[StreamServer] INFO: Environment Contract v1 Persistent Streaming Server listening on 0.0.0.0:9099
Learning Agent initialized on device: cuda
Manifest: Minecraft 1.20.4 (Protocol 765)
[Model] Latent World Model (RSSM), DIAYN Skills, Latent MPC, RND Curiosity initialized.
```

---

### Step 3: Start the Mineflayer Bridge
Open a second terminal (can be run either in Windows or WSL):

**In WSL:**
```bash
wsl bash -c "PATH=/home/tg/.local/bin:\$PATH; cd /mnt/d/minecraft_learning_bot/bridge && node bridge.js"
```

**Or in Windows PowerShell:**
```powershell
cd d:\minecraft_learning_bot\bridge
node bridge.js
```

What happens next:
1. The bridge automatically listens for LAN multicast broadcasts from TLauncher (`224.0.2.0:4445`) and captures the game port.
2. The bot `MineBot_Agent` spawns into your world.
3. The bridge connects to the WSL agent on `127.0.0.1:9099`, performs the 16-byte binary handshake (`HELLO` → `WELCOME`), and begins streaming 20 Hz observation/action transitions with CRC32 verification.

---

## 3. Running Synthetic Tests & Benchmarks (Offline / Headless)

You can run and test the complete neural learning loop, Latent MPC planner, DIAYN skill discriminator, and atomic checkpoint system without running Minecraft:

### A. Synthetic Smoke Test
Runs 40 environment steps with synthetic observations, world-model RSSM prediction, policy optimization, and atomic checkpoint saving:
```bash
wsl bash -c "cd /mnt/d/minecraft_learning_bot/agent && PYTHONPATH=.:.. /mnt/d/minecraft_learning_bot/.venv/bin/python -m bot.runtime --mode synthetic"
```

### B. Protocol Framed Stream Test
Tests the 16-byte binary framing protocol, sequence IDs, CRC32, PING/PONG heartbeats, and observation/action packet serialization:
```bash
wsl bash -c "cd /mnt/d/minecraft_learning_bot/agent && PYTHONPATH=.:.. /mnt/d/minecraft_learning_bot/.venv/bin/python tests/test_stream.py"
```

### C. Scientific Benchmark Evaluation (Phase 9)
Evaluates Random, Untrained, Reactive, and Full Model-based planning agents across MSE, KL, survival steps, and skill diversity:
```bash
wsl bash -c "cd /mnt/d/minecraft_learning_bot/agent && PYTHONPATH=.:.. /mnt/d/minecraft_learning_bot/.venv/bin/python -m bot.evaluation.benchmark"
```

### D. Architectural Ablation Studies (Phase 9)
Tests relative performance drops when ablating components (No World Model, No RND, No Memory, No Skills, No Planner):
```bash
wsl bash -c "cd /mnt/d/minecraft_learning_bot/agent && PYTHONPATH=.:.. /mnt/d/minecraft_learning_bot/.venv/bin/python -m bot.evaluation.ablations"
```

---

## 4. Checkpoint & Memory Resilience

All knowledge is retained persistently in `agent/checkpoints/`:
- `agent_checkpoint.pt`: Neural weights (World Model RSSM, Actor-Critic, DIAYN Discriminator, Skill Termination, RND Predictor & Target).
- `replay_buffer.pt`: Prioritized sequence replay buffer with transition history and TD/WM priority weights.
- `atomic checkpointing`: Uses `.tmp` writes followed by atomic renames to prevent corruption if interrupted.
- **Death Resilience**: When the bot dies in Minecraft, the episode terminates ($h_0 \leftarrow 0$), but all learned neural weights, replay memories, and spatial embeddings are retained.
