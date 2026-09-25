import random
from typing import Dict, List, Optional, Tuple, Any
import torch
import numpy as np
from pathlib import Path

class PrioritizedSequenceBuffer:
    """
    Prioritized episodic sequence buffer for World-Model and Actor-Critic learning.
    Samples sequences with probability proportional to prediction error, TD error, and novelty.
    """
    def __init__(self, capacity: int = 100_000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.episodes: List[List[Dict[str, np.ndarray]]] = []
        self.episode_priorities: List[float] = []
        self.current_episode: List[Dict[str, np.ndarray]] = []
        self.current_max_prio: float = 1.0
        self.total_steps = 0

    def add(
        self,
        voxels: np.ndarray,
        player_state: np.ndarray,
        inventory: np.ndarray,
        entities: np.ndarray,
        affordances: np.ndarray,
        validity_mask: np.ndarray,
        action: np.ndarray,
        reward: float,
        continuation: float,
        done: bool,
        priority: Optional[float] = None,
        success: bool = True,
        failure_reason: str = "NONE",
        consequence_delta: float = 0.0,
    ):
        prio = priority if priority is not None else self.current_max_prio
        self.current_max_prio = max(self.current_max_prio, prio)

        step_dict = {
            "voxels": np.asarray(voxels, dtype=np.int32),
            "player_state": np.asarray(player_state, dtype=np.float32),
            "inventory": np.asarray(inventory, dtype=np.float32),
            "entities": np.asarray(entities, dtype=np.float32),
            "affordances": np.asarray(affordances, dtype=np.float32),
            "validity_mask": np.asarray(validity_mask, dtype=np.float32),
            "action": np.asarray(action, dtype=np.float32),
            "reward": np.float32(reward),
            "continuation": np.float32(continuation),
            "done": bool(done),
            "priority": float(prio),
            "success": bool(success),
            "failure_reason": str(failure_reason),
            "consequence_delta": np.float32(consequence_delta),
        }
        self.current_episode.append(step_dict)
        self.total_steps += 1

        if done or len(self.current_episode) >= 1000:
            mean_prio = float(np.mean([s["priority"] for s in self.current_episode]))
            self.episodes.append(self.current_episode)
            self.episode_priorities.append(mean_prio)
            self.current_episode = []

            while self.total_steps > self.capacity and len(self.episodes) > 1:
                removed = self.episodes.pop(0)
                self.episode_priorities.pop(0)
                self.total_steps -= len(removed)

    def sample_sequences(
        self, batch_size: int, seq_len: int, device: torch.device
    ) -> Optional[Dict[str, torch.Tensor]]:
        valid_indices = [i for i, ep in enumerate(self.episodes) if len(ep) >= seq_len]
        if not valid_indices:
            if len(self.current_episode) >= seq_len:
                valid_indices = [-1]
            else:
                return None

        # Prioritized distribution
        if valid_indices == [-1]:
            chosen_indices = [-1] * batch_size
        else:
            prios = np.array([self.episode_priorities[i] for i in valid_indices], dtype=np.float32)
            probs = prios ** self.alpha
            probs_sum = probs.sum()
            probs = probs / probs_sum if probs_sum > 0 else np.ones_like(probs) / len(probs)
            chosen_indices = np.random.choice(valid_indices, size=batch_size, p=probs)

        batch_voxels = []
        batch_player = []
        batch_inventory = []
        batch_entities = []
        batch_affordances = []
        batch_validity = []
        batch_actions = []
        batch_rewards = []
        batch_continuations = []
        batch_dones = []
        batch_successes = []
        batch_consequences = []

        for idx in chosen_indices:
            ep = self.current_episode if idx == -1 else self.episodes[idx]
            max_start = len(ep) - seq_len
            start_idx = random.randint(0, max(0, max_start))
            slice_steps = ep[start_idx : start_idx + seq_len]

            batch_voxels.append([s["voxels"] for s in slice_steps])
            batch_player.append([s["player_state"] for s in slice_steps])
            batch_inventory.append([s["inventory"] for s in slice_steps])
            batch_entities.append([s["entities"] for s in slice_steps])
            batch_affordances.append([s["affordances"] for s in slice_steps])
            batch_validity.append([s["validity_mask"] for s in slice_steps])
            batch_actions.append([s["action"] for s in slice_steps])
            batch_rewards.append([s["reward"] for s in slice_steps])
            batch_continuations.append([s["continuation"] for s in slice_steps])
            batch_dones.append([s["done"] for s in slice_steps])
            batch_successes.append([s.get("success", True) for s in slice_steps])
            batch_consequences.append([s.get("consequence_delta", 0.0) for s in slice_steps])

        return {
            "voxels": torch.tensor(np.array(batch_voxels), dtype=torch.long, device=device),
            "player_state": torch.tensor(np.array(batch_player), dtype=torch.float32, device=device),
            "inventory": torch.tensor(np.array(batch_inventory), dtype=torch.float32, device=device),
            "entities": torch.tensor(np.array(batch_entities), dtype=torch.float32, device=device),
            "affordances": torch.tensor(np.array(batch_affordances), dtype=torch.float32, device=device),
            "validity_mask": torch.tensor(np.array(batch_validity), dtype=torch.float32, device=device),
            "actions": torch.tensor(np.array(batch_actions), dtype=torch.float32, device=device),
            "rewards": torch.tensor(np.array(batch_rewards), dtype=torch.float32, device=device).unsqueeze(-1),
            "continuations": torch.tensor(np.array(batch_continuations), dtype=torch.float32, device=device).unsqueeze(-1),
            "dones": torch.tensor(np.array(batch_dones), dtype=torch.bool, device=device).unsqueeze(-1),
            "successes": torch.tensor(np.array(batch_successes), dtype=torch.float32, device=device).unsqueeze(-1),
            "consequences": torch.tensor(np.array(batch_consequences), dtype=torch.float32, device=device).unsqueeze(-1),
        }

    def __len__(self) -> int:
        return self.total_steps

    def to_dict(self) -> Dict[str, Any]:
        """Serializes replay state for atomic checkpointing."""
        return {
            "episodes": self.episodes[-200:],  # retain recent 200 episodes for bounded disk payload
            "priorities": self.episode_priorities[-200:],
            "current_episode": self.current_episode,
            "current_max_prio": self.current_max_prio,
            "total_steps": self.total_steps,
        }

    def load_from_dict(self, data: Dict[str, Any]):
        """Restores replay state from atomic checkpoint payload."""
        if not data:
            return
        self.episodes = data.get("episodes", [])
        self.episode_priorities = data.get("priorities", [1.0] * len(self.episodes))
        self.current_episode = data.get("current_episode", [])
        self.current_max_prio = data.get("current_max_prio", 1.0)
        self.total_steps = data.get("total_steps", 0)

    def save_state(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_dict(), p)

    def load_state(self, path: str):
        p = Path(path)
        if p.exists():
            data = torch.load(p, map_location="cpu", weights_only=False)
            self.load_from_dict(data)
