from collections import deque
import random
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from pathlib import Path

class TrajectoryBuffer:
    """
    Episodic sequence buffer for world-model and latent policy training.
    Persists experience sequences across restarts and deaths.
    """
    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.episodes: List[List[Dict[str, np.ndarray]]] = []
        self.current_episode: List[Dict[str, np.ndarray]] = []
        self.total_steps = 0

    def add(
        self,
        voxels: np.ndarray,
        player_state: np.ndarray,
        inventory: np.ndarray,
        entities: np.ndarray,
        affordances: np.ndarray,
        action: np.ndarray,
        reward: float,
        continuation: float,
        done: bool,
    ):
        step_dict = {
            "voxels": np.asarray(voxels, dtype=np.int32),
            "player_state": np.asarray(player_state, dtype=np.float32),
            "inventory": np.asarray(inventory, dtype=np.float32),
            "entities": np.asarray(entities, dtype=np.float32),
            "affordances": np.asarray(affordances, dtype=np.float32),
            "action": np.asarray(action, dtype=np.float32),
            "reward": np.float32(reward),
            "continuation": np.float32(continuation),
            "done": bool(done),
        }
        self.current_episode.append(step_dict)
        self.total_steps += 1

        if done or len(self.current_episode) >= 1000:
            self.episodes.append(self.current_episode)
            self.current_episode = []
            # Trim old episodes if over capacity
            while self.total_steps > self.capacity and len(self.episodes) > 1:
                removed = self.episodes.pop(0)
                self.total_steps -= len(removed)

    def sample_sequences(
        self, batch_size: int, seq_len: int, device: torch.device
    ) -> Optional[Dict[str, torch.Tensor]]:
        # Filter episodes that have at least seq_len steps
        valid_episodes = [ep for ep in self.episodes if len(ep) >= seq_len]
        if not valid_episodes:
            if len(self.current_episode) >= seq_len:
                valid_episodes = [self.current_episode]
            else:
                return None

        batch_voxels = []
        batch_player = []
        batch_inventory = []
        batch_entities = []
        batch_affordances = []
        batch_actions = []
        batch_rewards = []
        batch_continuations = []
        batch_dones = []

        for _ in range(batch_size):
            ep = random.choice(valid_episodes)
            max_start = len(ep) - seq_len
            start_idx = random.randint(0, max(0, max_start))
            slice_steps = ep[start_idx : start_idx + seq_len]

            batch_voxels.append([s["voxels"] for s in slice_steps])
            batch_player.append([s["player_state"] for s in slice_steps])
            batch_inventory.append([s["inventory"] for s in slice_steps])
            batch_entities.append([s["entities"] for s in slice_steps])
            batch_affordances.append([s["affordances"] for s in slice_steps])
            batch_actions.append([s["action"] for s in slice_steps])
            batch_rewards.append([s["reward"] for s in slice_steps])
            batch_continuations.append([s["continuation"] for s in slice_steps])
            batch_dones.append([s["done"] for s in slice_steps])

        return {
            "voxels": torch.tensor(np.array(batch_voxels), dtype=torch.long, device=device),
            "player_state": torch.tensor(np.array(batch_player), dtype=torch.float32, device=device),
            "inventory": torch.tensor(np.array(batch_inventory), dtype=torch.float32, device=device),
            "entities": torch.tensor(np.array(batch_entities), dtype=torch.float32, device=device),
            "affordances": torch.tensor(np.array(batch_affordances), dtype=torch.float32, device=device),
            "actions": torch.tensor(np.array(batch_actions), dtype=torch.float32, device=device),
            "rewards": torch.tensor(np.array(batch_rewards), dtype=torch.float32, device=device).unsqueeze(-1),
            "continuations": torch.tensor(np.array(batch_continuations), dtype=torch.float32, device=device).unsqueeze(-1),
            "dones": torch.tensor(np.array(batch_dones), dtype=torch.bool, device=device).unsqueeze(-1),
        }

    def __len__(self) -> int:
        return self.total_steps + len(self.current_episode)

    def save_state(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "episodes": self.episodes[-100:],  # keep last 100 full episodes on disk
            "total_steps": self.total_steps
        }, p)

    def load_state(self, path: str):
        p = Path(path)
        if p.exists():
            data = torch.load(p, map_location="cpu", weights_only=False)
            self.episodes = data.get("episodes", [])
            self.total_steps = data.get("total_steps", 0)
