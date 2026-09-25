from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .config import Config
from .schemas import (
    FullObservation,
    HierarchicalAction,
    ContinuousMotorControl,
    DiscreteActionCommand,
    ActionCategory,
    ActionPrimitive,
    IDX_TO_PRIMITIVE,
    EnvironmentManifest,
)
from .models import (
    MultiModalObservationEncoder,
    RecurrentWorldModel,
    RNDCuriosity,
    SkillDiscovery,
    HierarchicalActorCritic,
)
from .memory import PrioritizedSequenceBuffer, SpatialMemory, ExperienceGraph
from .skills import SkillLibrary
from .planning import LatentMPCPlanner
from .training import AtomicCheckpointManager, AsyncLearnerThread
from .evaluation import ExperimentMetricsLogger, ScientificBenchmarkSuite
from .learning import train_world_model_step, train_actor_critic_imagination

class LearningAgent:
    """
    Autonomous Minecraft Learning Agent (Play -> Learn -> Live -> Grow).
    Integrates multi-modal perception, RSSM dynamics, RND curiosity, DIAYN skills,
    latent MPC planning, spatial memory, and lifelong learning.
    """
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.manifest: Optional[EnvironmentManifest] = None

        # 1. Perception
        self.encoder = MultiModalObservationEncoder(
            voxel_vocab=self.cfg.voxel_vocab,
            voxel_emb_dim=self.cfg.voxel_emb_dim,
            item_vocab=self.cfg.item_vocab,
            item_emb_dim=self.cfg.item_emb_dim,
            entity_vocab=self.cfg.entity_vocab,
            entity_emb_dim=self.cfg.entity_emb_dim,
            player_state_dim=self.cfg.player_state_dim,
            affordance_dim=self.cfg.affordance_dim,
            validity_mask_dim=self.cfg.validity_mask_dim,
            hidden_dim=self.cfg.hidden_dim,
        ).to(self.device)

        # 2. Recurrent World Model (RSSM)
        self.world_model = RecurrentWorldModel(
            hidden_dim=self.cfg.recurrent_dim,
            latent_dim=self.cfg.latent_dim,
            action_dim=self.cfg.motor_dim + self.cfg.num_primitives,
        ).to(self.device)

        # 3. Exploration, Skills & Policy
        self.rnd = RNDCuriosity(in_dim=self.cfg.hidden_dim, out_dim=64).to(self.device)
        self.skill_net = SkillDiscovery(
            state_dim=self.cfg.recurrent_dim + self.cfg.latent_dim, num_skills=8
        ).to(self.device)
        self.actor_critic = HierarchicalActorCritic(
            hidden_dim=self.cfg.recurrent_dim,
            latent_dim=self.cfg.latent_dim,
            num_skills=8,
            motor_dim=self.cfg.motor_dim,
            num_primitives=self.cfg.num_primitives,
        ).to(self.device)

        # 4. Latent MPC Planner
        self.planner = LatentMPCPlanner(
            self.world_model,
            self.actor_critic,
            horizon=self.cfg.imagination_horizon,
            num_candidates=8,
            gamma=self.cfg.gamma,
        )

        # 5. Optimizers
        self.wm_params = (
            list(self.encoder.parameters())
            + list(self.world_model.parameters())
            + list(self.rnd.predictor.parameters())
            + list(self.skill_net.parameters())
        )
        self.wm_opt = torch.optim.Adam(self.wm_params, lr=self.cfg.learning_rate)
        self.ac_opt = torch.optim.Adam(self.actor_critic.parameters(), lr=self.cfg.learning_rate)

        # 6. Memory Subsystems
        self.memory = PrioritizedSequenceBuffer(capacity=self.cfg.replay_capacity)
        self.spatial_memory = SpatialMemory(chunk_size=16)
        self.experience_graph = ExperienceGraph(max_nodes=5000)
        self.skill_library = SkillLibrary(num_skills=8)

        # 7. Persistence & Evaluation
        self.checkpoint_manager = AtomicCheckpointManager(checkpoint_dir=self.cfg.checkpoint_dir)
        self.metrics_logger = ExperimentMetricsLogger()
        self.benchmark_suite = ScientificBenchmarkSuite()

        # Active Episode State
        self.h = None
        self.prev_z = None
        self.prev_a = None
        self.prev_transition_data = None
        self.prev_latent = None
        self.current_skill_id = 0
        self.skill_duration_ticks = 0
        self.episode_steps = 0
        self.total_steps = 0
        self.episode_count = 0

        self.load_checkpoint()

        # 8. Decoupled Asynchronous Background Learner Lane
        self.learner_thread = AsyncLearnerThread(
            self,
            batch_size=16,
            seq_len=16,
            horizon=self.cfg.imagination_horizon,
            sleep_interval=0.01,
        )
        self.learner_thread.start()

    def set_environment_manifest(self, manifest: EnvironmentManifest):
        self.manifest = manifest

    def reset_episode(self):
        self.benchmark_suite.record_episode_end(self.episode_steps)
        self.h = None
        self.prev_z = None
        self.prev_a = None
        self.prev_transition_data = None
        self.prev_latent = None
        self.skill_duration_ticks = 0
        self.episode_steps = 0
        self.episode_count += 1

    def _convert_obs_to_tensors(self, obs: FullObservation) -> Dict[str, torch.Tensor]:
        shape = tuple(obs.voxel_shape) if len(obs.voxel_shape) == 3 else (11, 11, 11)
        expected_size = shape[0] * shape[1] * shape[2]
        vox = obs.voxels if len(obs.voxels) == expected_size else [0] * expected_size
        t_vox = torch.tensor(vox, dtype=torch.long, device=self.device).reshape(1, *shape)

        p_state = list(obs.player_state)
        p_state += [0.0] * max(0, self.cfg.player_state_dim - len(p_state))
        t_play = torch.tensor(p_state[: self.cfg.player_state_dim], dtype=torch.float32, device=self.device).unsqueeze(0)

        inv_data = []
        raw_slots = obs.inventory.slots if obs.inventory else []
        for i in range(36):
            if i < len(raw_slots):
                s = raw_slots[i]
                inv_data.append([s.item_canonical_id, s.count, s.durability])
            else:
                inv_data.append([0, 0, 0.0])
        for a in [obs.inventory.armor_head, obs.inventory.armor_chest, obs.inventory.armor_legs, obs.inventory.armor_feet]:
            inv_data.append([a.item_canonical_id, a.count, a.durability])
        oh = obs.inventory.offhand
        inv_data.append([oh.item_canonical_id, oh.count, oh.durability])

        t_inv = torch.tensor(inv_data, dtype=torch.float32, device=self.device).unsqueeze(0)

        ent_data = []
        for i in range(self.cfg.max_entities):
            if i < len(obs.entities):
                e = obs.entities[i]
                ent_data.append([e.canonical_type_id, e.dx, e.dy, e.dz, e.vx, e.vy, e.vz, e.health, 1.0 if e.is_alive else 0.0])
            else:
                ent_data.append([0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        t_ent = torch.tensor(ent_data, dtype=torch.float32, device=self.device).unsqueeze(0)

        aff = obs.affordances
        aff_data = [
            float(aff.targeted_block_canonical_id),
            float(aff.targeted_block_distance),
            float(aff.targeted_block_face),
            1.0 if aff.can_mine_target else 0.0,
            float(aff.targeted_entity_idx),
            float(aff.light_level),
            float(aff.sky_light),
            1.0 if aff.open_container_type != "none" else 0.0,
        ]
        t_aff = torch.tensor(aff_data, dtype=torch.float32, device=self.device).unsqueeze(0)

        val = obs.validity_mask
        val_data = [
            1.0 if val.can_jump else 0.0,
            1.0 if val.can_sprint else 0.0,
            1.0 if val.can_sneak else 0.0,
            1.0 if val.can_attack_entity else 0.0,
            1.0 if val.can_dig_block else 0.0,
            1.0 if val.can_place_block else 0.0,
            1.0 if val.can_use_item else 0.0,
            1.0 if val.can_open_container else 0.0,
            1.0 if val.can_sleep else 0.0,
        ]
        t_val = torch.tensor(val_data, dtype=torch.float32, device=self.device).unsqueeze(0)

        return {
            "voxels": t_vox,
            "player_state": t_play,
            "inventory": t_inv,
            "entities": t_ent,
            "affordances": t_aff,
            "validity_mask": t_val,
        }

    def step(self, obs: FullObservation) -> Tuple[HierarchicalAction, Dict[str, float]]:
        self.total_steps += 1
        self.episode_steps += 1
        tensors = self._convert_obs_to_tensors(obs)

        # 1. Multi-modal Perception
        with torch.no_grad():
            e_t = self.encoder(
                tensors["voxels"],
                tensors["player_state"],
                tensors["inventory"],
                tensors["entities"],
                tensors["affordances"],
                tensors["validity_mask"],
            )

            # 2. Decomposed Curiosity & Spatial Signals
            rnd_reward, _ = self.rnd(e_t)
            curiosity = float(rnd_reward.item())

            # Spatial memory tracking with directional heading awareness
            px = obs.player_state[4] if len(obs.player_state) > 4 else 0.0
            py = obs.player_state[5] if len(obs.player_state) > 5 else 64.0
            pz = obs.player_state[6] if len(obs.player_state) > 6 else 0.0
            yaw = obs.player_state[11] if len(obs.player_state) > 11 else 0.0
            spatial_novelty = self.spatial_memory.get_spatial_novelty(px, pz, yaw=yaw)

            continuation = 0.0 if obs.done else 1.0

            # Consequence signal from the previous action's real-world outcome.
            # A failed action (success=False) or a no-op consequence (zero state delta)
            # carries an opportunity cost that must enter the training target.
            # This is a general cognitive prior: repeated ineffective actions should
            # acquire negative expected value without hardcoding "don't craft".
            prev_res = obs.last_action_result
            action_failed = prev_res is not None and not prev_res.success
            state_delta_magnitude = abs(prev_res.state_delta.get("health_delta", 0.0)) if prev_res else 0.0
            action_ineffective = action_failed or state_delta_magnitude < 0.001
            consequence_penalty = -0.5 if action_ineffective else 0.0
            consequence_delta = float(prev_res.state_delta.get("health_delta", 0.0)) if prev_res else 0.0

            step_reward = self.cfg.rnd_weight * curiosity + 0.2 * spatial_novelty + consequence_penalty

        # 3. Store prior transition into Prioritized Replay
        if self.prev_transition_data is not None:
            # Boost replay priority for negative outcomes so the model trains
            # more heavily on failure transitions.
            priority = max(curiosity + abs(consequence_penalty) * 0.5, 0.1)
            prev_success = obs.last_action_result.success if obs.last_action_result else True
            prev_reason = obs.last_action_result.failure_reason if obs.last_action_result else "NONE"
            self.memory.add(
                voxels=self.prev_transition_data["voxels"].squeeze(0).cpu().numpy(),
                player_state=self.prev_transition_data["player_state"].squeeze(0).cpu().numpy(),
                inventory=self.prev_transition_data["inventory"].squeeze(0).cpu().numpy(),
                entities=self.prev_transition_data["entities"].squeeze(0).cpu().numpy(),
                affordances=self.prev_transition_data["affordances"].squeeze(0).cpu().numpy(),
                validity_mask=self.prev_transition_data["validity_mask"].squeeze(0).cpu().numpy(),
                action=self.prev_a.squeeze(0).cpu().numpy(),
                reward=step_reward,
                continuation=continuation,
                done=obs.done,
                priority=priority,
                success=prev_success,
                failure_reason=prev_reason,
                consequence_delta=consequence_delta,
            )

            # Record in Experience Graph
            if self.prev_latent is not None:
                self.experience_graph.record_transition(
                    from_latent=self.prev_latent,
                    action_name=obs.last_action_result.action_primitive,
                    to_latent=e_t[0].cpu().numpy()[:64],
                    consequence_delta=consequence_delta,
                    success=obs.last_action_result.success,
                )

        # Update spatial memory with actual consequence magnitude
        self.spatial_memory.record_visit(px, py, pz, yaw, e_t[0].cpu().numpy()[:64], consequence_delta=consequence_delta)

        if obs.done:
            self.reset_episode()
            return HierarchicalAction(), {"curiosity": curiosity, "reward": step_reward, "done": 1.0}

        # 4. Recurrent World Model Step
        if self.h is None:
            self.h = torch.zeros(1, self.cfg.recurrent_dim, device=self.device)
            self.prev_z = torch.zeros(1, self.cfg.latent_dim, device=self.device)
            self.prev_a = torch.zeros(1, self.cfg.motor_dim + self.cfg.num_primitives, device=self.device)

        with torch.no_grad():
            self.h = self.world_model.recurrent_step(self.prev_z, self.prev_a, self.h)
            z, _, _ = self.world_model.infer_posterior(self.h, e_t)

            # 5. Temporal Skill Selection & Model Predictive Planning
            if self.skill_duration_ticks <= 0:
                skill_logits = self.skill_net(torch.cat([self.h, z], dim=-1))
                self.current_skill_id = int(torch.argmax(skill_logits, dim=-1).item())
                self.skill_duration_ticks = 6 # Execute skill over 6 ticks

            self.skill_duration_ticks -= 1
            skill_one_hot = F.one_hot(torch.tensor([self.current_skill_id], device=self.device), num_classes=8).float()

            # Latent MPC Planning — with experience-graph prior feedback
            val_mask_bool = torch.tensor(obs.validity_mask.valid_primitives_mask[:self.cfg.num_primitives], device=self.device).bool()
            best_motor, best_prim_idx, planned_value = self.planner.plan_best_action(
                self.h, z, skill_one_hot,
                validity_mask=val_mask_bool,
                experience_graph=self.experience_graph,
                current_latent=self.prev_latent,
            )

        # Build HierarchicalAction
        m_vec = best_motor[0].cpu().tolist()
        motor_control = ContinuousMotorControl(
            move_x=float(np.clip(m_vec[0], -1.0, 1.0)),
            move_z=float(np.clip(m_vec[1], -1.0, 1.0)),
            yaw_delta=float(np.clip(m_vec[2], -1.0, 1.0)),
            pitch_delta=float(np.clip(m_vec[3], -1.0, 1.0)),
            jump=bool(m_vec[4] > 0.0),
            sprint=bool(m_vec[5] > 0.0),
            sneak=bool(m_vec[6] > 0.0),
        )

        prim_name = IDX_TO_PRIMITIVE.get(best_prim_idx, ActionPrimitive.NOOP.value)
        command = DiscreteActionCommand(
            primitive=ActionPrimitive(prim_name),
            target_entity_idx=obs.affordances.targeted_entity_idx if obs.affordances.targeted_entity_idx >= 0 else 0,
            target_slot=obs.inventory.selected_hotbar_slot if obs.inventory else 0,
            duration_ticks=1,
        )
        action = HierarchicalAction(motor=motor_control, command=command)

        # Construct flat action vector: motor(7) + primitive_one_hot(27) = 34
        prim_one_hot = F.one_hot(torch.tensor([best_prim_idx], device=self.device), num_classes=self.cfg.num_primitives).float()
        action_tensor = torch.cat([best_motor, prim_one_hot], dim=-1)

        self.prev_z = z
        self.prev_a = action_tensor
        self.prev_transition_data = tensors
        self.prev_latent = e_t[0].cpu().numpy()[:64]

        # 6. Non-blocking Asynchronous Learner Metrics (Zero Stall on Minecraft Ticks)
        train_metrics = self.learner_thread.get_metrics()

        # 7. Scientific Benchmarking & Metrics
        current_health = obs.player_state[0] if len(obs.player_state) > 0 else 20.0
        current_food = obs.player_state[1] if len(obs.player_state) > 1 else 20.0
        self.benchmark_suite.record_step(health=current_health, prediction_error=train_metrics.get("wm_loss", 0.0))

        checkpoint_saved = False
        if self.total_steps % self.cfg.checkpoint_interval == 0:
            self.save_checkpoint()
            checkpoint_saved = True

        res = obs.last_action_result
        metrics = {
            "step": self.total_steps,
            "pos": [round(px, 2), round(py, 2), round(pz, 2)],
            "health": current_health,
            "food": current_food,
            "voxels_count": len(obs.voxels) if obs.voxels else 0,
            "entities_count": len(obs.entities) if obs.entities else 0,
            "curiosity": curiosity,
            "spatial_novelty": spatial_novelty,
            "value": float(planned_value),
            "skill_id": self.current_skill_id,
            "skill_duration": self.skill_duration_ticks,
            "primitive": best_prim_idx,
            "primitive_name": prim_name,
            "motor": [round(m_vec[0], 2), round(m_vec[1], 2), round(m_vec[2], 2), round(m_vec[3], 2)],
            "jump": bool(m_vec[4] > 0.0),
            "sprint": bool(m_vec[5] > 0.0),
            "sneak": bool(m_vec[6] > 0.0),
            "replay_size": self.memory.total_steps,
            "replay_episodes": len(self.memory.episodes),
            "regions_explored": self.spatial_memory.total_regions_discovered(),
            "last_action_primitive": res.action_primitive if res else "none",
            "last_action_success": res.success if res else True,
            "last_action_delta": res.state_delta if res else {},
            "consequence_penalty": consequence_penalty,
            "step_reward": step_reward,
            "checkpoint_saved": checkpoint_saved,
            **train_metrics,
        }
        self.metrics_logger.log(self.total_steps, metrics)

        return action, metrics

    def training_step(self) -> Dict[str, float]:
        batch = self.memory.sample_sequences(
            batch_size=self.cfg.batch_size,
            seq_len=self.cfg.sequence_length,
            device=self.device,
        )
        if batch is None:
            return {}

        wm_loss, wm_metrics, last_h, last_z = train_world_model_step(
            self.encoder,
            self.world_model,
            batch,
            kl_weight=self.cfg.kl_weight,
            continuation_weight=self.cfg.continuation_weight,
            rnd=self.rnd,
        )
        self.wm_opt.zero_grad()
        wm_loss.backward()
        nn.utils.clip_grad_norm_(self.wm_params, max_norm=10.0)
        self.wm_opt.step()

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
        model_payload = {
            "manifest": self.manifest.to_dict() if self.manifest else None,
            "encoder": self.encoder.state_dict(),
            "world_model": self.world_model.state_dict(),
            "rnd": self.rnd.state_dict(),
            "skill_net": self.skill_net.state_dict(),
            "actor_critic": self.actor_critic.state_dict(),
            "wm_opt": self.wm_opt.state_dict(),
            "ac_opt": self.ac_opt.state_dict(),
            "total_steps": self.total_steps,
            "episode_count": self.episode_count,
        }
        self.checkpoint_manager.save_checkpoint(
            step=self.total_steps,
            model_payload=model_payload,
            memory_payload=self.memory.to_dict(),
            spatial_payload=self.spatial_memory.to_dict(),
            skills_payload=self.skill_library.to_dict(),
        )

    def load_checkpoint(self, path: Optional[str] = None):
        ckpt_dict = self.checkpoint_manager.load_latest_checkpoint()
        if not ckpt_dict:
            return

        model_data = ckpt_dict["model"]
        try:
            self.encoder.load_state_dict(model_data["encoder"], strict=False)
            self.world_model.load_state_dict(model_data["world_model"], strict=False)
            self.rnd.load_state_dict(model_data["rnd"], strict=False)
            self.skill_net.load_state_dict(model_data["skill_net"], strict=False)
            self.actor_critic.load_state_dict(model_data["actor_critic"], strict=False)
            if "wm_opt" in model_data:
                self.wm_opt.load_state_dict(model_data["wm_opt"])
            if "ac_opt" in model_data:
                self.ac_opt.load_state_dict(model_data["ac_opt"])
            if model_data.get("manifest"):
                self.manifest = EnvironmentManifest.from_dict(model_data["manifest"])
            self.total_steps = model_data.get("total_steps", 0)
            self.episode_count = model_data.get("episode_count", 0)

            if ckpt_dict.get("replay"):
                self.memory.load_from_dict(ckpt_dict["replay"])
            if ckpt_dict.get("spatial"):
                self.spatial_memory.load_from_dict(ckpt_dict["spatial"])
            if ckpt_dict.get("skills"):
                self.skill_library.load_from_dict(ckpt_dict["skills"])

            print(f"[Checkpoint] Resumed from step {self.total_steps} (Episode {self.episode_count}).")
        except RuntimeError as e:
            print(f"[Checkpoint] Architecture mismatch with existing checkpoint: {e}. Starting fresh weights.")

    def close(self):
        """Gracefully stops background worker threads and flushes checkpoints."""
        if hasattr(self, "learner_thread") and self.learner_thread:
            self.learner_thread.stop()
            try:
                self.learner_thread.join(timeout=1.0)
            except Exception:
                pass
        try:
            self.save_checkpoint()
        except Exception:
            pass
