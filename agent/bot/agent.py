from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .config import Config
from .schemas import Observation, Action
from .models import (
    MultiModalObservationEncoder,
    RecurrentWorldModel,
    RNDCuriosity,
    SkillDiscovery,
    ActorCritic,
)
from .memory import TrajectoryBuffer
from .learning import train_world_model_step, train_actor_critic_imagination

class LearningAgent:
    """
    Autonomous Minecraft Learning Agent (Play -> Learn -> Live -> Grow).
    Learns purely through experience, world modeling, curiosity, and latent imagination.
    """
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Neural Modules
        self.encoder = MultiModalObservationEncoder(
            voxel_vocab=self.cfg.voxel_vocab,
            voxel_emb_dim=self.cfg.voxel_emb_dim,
            item_vocab=self.cfg.item_vocab,
            item_emb_dim=self.cfg.item_emb_dim,
            entity_vocab=self.cfg.entity_vocab,
            entity_emb_dim=self.cfg.entity_emb_dim,
            player_state_dim=self.cfg.player_state_dim,
            affordance_dim=self.cfg.affordance_dim,
            hidden_dim=self.cfg.hidden_dim,
        ).to(self.device)

        self.world_model = RecurrentWorldModel(
            hidden_dim=self.cfg.recurrent_dim,
            latent_dim=self.cfg.latent_dim,
            action_dim=self.cfg.action_dim,
        ).to(self.device)

        self.rnd = RNDCuriosity(in_dim=self.cfg.hidden_dim, out_dim=64).to(self.device)
        self.skill_net = SkillDiscovery(
            state_dim=self.cfg.recurrent_dim + self.cfg.latent_dim, num_skills=8
        ).to(self.device)
        self.actor_critic = ActorCritic(
            hidden_dim=self.cfg.recurrent_dim,
            latent_dim=self.cfg.latent_dim,
            num_skills=8,
            action_dim=self.cfg.action_dim,
        ).to(self.device)

        # 2. Optimizers
        self.wm_params = (
            list(self.encoder.parameters())
            + list(self.world_model.parameters())
            + list(self.rnd.predictor.parameters())
            + list(self.skill_net.parameters())
        )
        self.wm_opt = torch.optim.Adam(self.wm_params, lr=self.cfg.learning_rate)
        self.ac_opt = torch.optim.Adam(self.actor_critic.parameters(), lr=self.cfg.learning_rate)

        # 3. Memory & Episode State
        self.memory = TrajectoryBuffer(capacity=self.cfg.replay_capacity)
        self.h = None
        self.prev_z = None
        self.prev_a = None
        self.prev_transition_data = None
        self.total_steps = 0
        self.episode_count = 0

        # Load existing checkpoint if available
        self.load_checkpoint()

    def reset_episode(self):
        """Called upon death or new session reset."""
        self.h = None
        self.prev_z = None
        self.prev_a = None
        self.prev_transition_data = None
        self.episode_count += 1

    def _convert_obs_to_tensors(self, obs: Observation) -> Dict[str, torch.Tensor]:
        # 1. Voxels: [1, 11, 11, 11]
        shape = tuple(obs.voxel_shape) if len(obs.voxel_shape) == 3 else (11, 11, 11)
        expected_size = shape[0] * shape[1] * shape[2]
        vox = obs.voxels if len(obs.voxels) == expected_size else [0] * expected_size
        t_vox = torch.tensor(vox, dtype=torch.long, device=self.device).reshape(1, *shape)

        # 2. Player state: [1, 17]
        p_state = list(obs.player_state)
        p_state += [0.0] * max(0, self.cfg.player_state_dim - len(p_state))
        t_play = torch.tensor(p_state[: self.cfg.player_state_dim], dtype=torch.float32, device=self.device).unsqueeze(0)

        # 3. Inventory: [1, 36, 3]
        inv_data = []
        for i in range(self.cfg.max_inventory_slots):
            if i < len(obs.inventory):
                s = obs.inventory[i]
                inv_data.append([s.item_id, s.count, s.durability])
            else:
                inv_data.append([0, 0, 0.0])
        t_inv = torch.tensor(inv_data, dtype=torch.float32, device=self.device).unsqueeze(0)

        # 4. Entities: [1, 16, 9]
        ent_data = []
        for i in range(self.cfg.max_entities):
            if i < len(obs.entities):
                e = obs.entities[i]
                ent_data.append([e.type_id, e.dx, e.dy, e.dz, e.vx, e.vy, e.vz, e.health, 1.0 if e.is_alive else 0.0])
            else:
                ent_data.append([0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        t_ent = torch.tensor(ent_data, dtype=torch.float32, device=self.device).unsqueeze(0)

        # 5. Affordances: [1, 9]
        aff = obs.affordances
        aff_data = [
            float(aff.target_block_id),
            float(aff.target_block_distance),
            1.0 if aff.can_mine else 0.0,
            float(aff.light_level),
            float(aff.time_of_day),
            1.0 if aff.is_raining else 0.0,
            1.0 if aff.is_sleeping else 0.0,
            1.0 if aff.is_swimming else 0.0,
            float(aff.equipped_item_id),
        ]
        t_aff = torch.tensor(aff_data, dtype=torch.float32, device=self.device).unsqueeze(0)

        return {
            "voxels": t_vox,
            "player_state": t_play,
            "inventory": t_inv,
            "entities": t_ent,
            "affordances": t_aff,
        }

    def step(self, obs: Observation) -> Tuple[Action, Dict[str, float]]:
        self.total_steps += 1
        tensors = self._convert_obs_to_tensors(obs)

        # 1. Encode observation into unified latent vector e_t
        with torch.no_grad():
            e_t = self.encoder(
                tensors["voxels"],
                tensors["player_state"],
                tensors["inventory"],
                tensors["entities"],
                tensors["affordances"],
            )

            # 2. Intrinsic Exploration & Curiosity (RND)
            rnd_reward, _ = self.rnd(e_t)
            intrinsic_curiosity = float(rnd_reward.item())

            # Survival / Continuation signal
            continuation = 0.0 if obs.done else 1.0
            
            # Intrinsic progress signal: living bonus + novelty
            step_reward = 0.05 + self.cfg.rnd_weight * intrinsic_curiosity

        # 3. Store prior transition into replay memory
        if self.prev_transition_data is not None:
            self.memory.add(
                voxels=self.prev_transition_data["voxels"].squeeze(0).cpu().numpy(),
                player_state=self.prev_transition_data["player_state"].squeeze(0).cpu().numpy(),
                inventory=self.prev_transition_data["inventory"].squeeze(0).cpu().numpy(),
                entities=self.prev_transition_data["entities"].squeeze(0).cpu().numpy(),
                affordances=self.prev_transition_data["affordances"].squeeze(0).cpu().numpy(),
                action=self.prev_a.squeeze(0).cpu().numpy(),
                reward=step_reward,
                continuation=continuation,
                done=obs.done,
            )

        if obs.done:
            self.reset_episode()
            return Action(), {"curiosity": intrinsic_curiosity, "reward": step_reward, "done": 1.0}

        # 4. Update Recurrent State & Infer Posterior Latent
        if self.h is None:
            self.h = torch.zeros(1, self.cfg.recurrent_dim, device=self.device)
            self.prev_z = torch.zeros(1, self.cfg.latent_dim, device=self.device)
            self.prev_a = torch.zeros(1, self.cfg.action_dim, device=self.device)

        with torch.no_grad():
            self.h = self.world_model.recurrent_step(self.prev_z, self.prev_a, self.h)
            z, _, _ = self.world_model.infer_posterior(self.h, e_t)

            # 5. Skill Discovery & Policy Action Sampling
            skill_logits = self.skill_net(torch.cat([self.h, z], dim=-1))
            skill_id = int(torch.argmax(skill_logits, dim=-1).item())
            skill_one_hot = F.one_hot(torch.tensor([skill_id], device=self.device), num_classes=8).float()

            full_state = torch.cat([self.h, z, skill_one_hot], dim=-1)
            _, action_dist = self.actor_critic.forward_policy(full_state)
            action_tensor = action_dist.sample()
            val = self.actor_critic.forward_value(full_state).item()

        action_vec = action_tensor[0].cpu().tolist()
        action = Action.from_vector(action_vec)

        # 6. Save current state for next transition
        self.prev_z = z
        self.prev_a = action_tensor
        self.prev_transition_data = tensors

        # 7. Online Continuous Learning Step (if replay has enough experience)
        train_metrics = self.training_step()

        # 8. Checkpoint periodically
        if self.total_steps % self.cfg.checkpoint_interval == 0:
            self.save_checkpoint()

        metrics = {
            "step": self.total_steps,
            "curiosity": intrinsic_curiosity,
            "value": float(val),
            "skill_id": skill_id,
            **train_metrics,
        }
        return action, metrics

    def training_step(self) -> Dict[str, float]:
        """Runs a continuous learning update over replayed sequences."""
        batch = self.memory.sample_sequences(
            batch_size=self.cfg.batch_size,
            seq_len=self.cfg.sequence_length,
            device=self.device,
        )
        if batch is None:
            return {}

        # 1. Train World Model & Encoder
        wm_loss, wm_metrics, last_h, last_z = train_world_model_step(
            self.encoder,
            self.world_model,
            batch,
            kl_weight=self.cfg.kl_weight,
            continuation_weight=self.cfg.continuation_weight,
        )
        self.wm_opt.zero_grad()
        wm_loss.backward()
        nn.utils.clip_grad_norm_(self.wm_params, max_norm=10.0)
        self.wm_opt.step()

        # 2. Train Actor-Critic via Latent Imagination (Dreaming)
        ac_loss, ac_metrics = train_actor_critic_imagination(
            self.actor_critic,
            self.world_model,
            self.skill_net,
            start_h=last_h,
            start_z=last_z,
            horizon=self.cfg.imagination_horizon,
            gamma=self.cfg.gamma,
            lambda_gae=self.cfg.lambda_gae,
        )
        self.ac_opt.zero_grad()
        ac_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=10.0)
        self.ac_opt.step()

        return {**wm_metrics, **ac_metrics}

    def save_checkpoint(self, path: Optional[str] = None):
        target_dir = Path(path or self.cfg.checkpoint_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = target_dir / "agent_checkpoint.pt"
        torch.save({
            "encoder": self.encoder.state_dict(),
            "world_model": self.world_model.state_dict(),
            "rnd": self.rnd.state_dict(),
            "skill_net": self.skill_net.state_dict(),
            "actor_critic": self.actor_critic.state_dict(),
            "wm_opt": self.wm_opt.state_dict(),
            "ac_opt": self.ac_opt.state_dict(),
            "total_steps": self.total_steps,
            "episode_count": self.episode_count,
        }, ckpt_path)
        self.memory.save_state(str(target_dir / "replay_buffer.pt"))

    def load_checkpoint(self, path: Optional[str] = None):
        target_dir = Path(path or self.cfg.checkpoint_dir)
        ckpt_path = target_dir / "agent_checkpoint.pt"
        if not ckpt_path.exists():
            return
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.world_model.load_state_dict(ckpt["world_model"])
        self.rnd.load_state_dict(ckpt["rnd"])
        self.skill_net.load_state_dict(ckpt["skill_net"])
        self.actor_critic.load_state_dict(ckpt["actor_critic"])
        if "wm_opt" in ckpt:
            self.wm_opt.load_state_dict(ckpt["wm_opt"])
        if "ac_opt" in ckpt:
            self.ac_opt.load_state_dict(ckpt["ac_opt"])
        self.total_steps = ckpt.get("total_steps", 0)
        self.episode_count = ckpt.get("episode_count", 0)
        self.memory.load_state(str(target_dir / "replay_buffer.pt"))
