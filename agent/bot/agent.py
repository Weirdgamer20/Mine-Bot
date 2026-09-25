from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List
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
        self.learner_thread = None

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

    def _convert_obs_batch_to_tensors(self, obs_list: List[FullObservation]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(obs_list)
        vox_batch = np.zeros((batch_size, 11, 11, 11), dtype=np.int64)
        play_batch = np.zeros((batch_size, self.cfg.player_state_dim), dtype=np.float32)
        inv_batch = np.zeros((batch_size, 41, 3), dtype=np.float32)
        ent_batch = np.zeros((batch_size, self.cfg.max_entities, 9), dtype=np.float32)
        aff_batch = np.zeros((batch_size, 8), dtype=np.float32)
        val_batch = np.zeros((batch_size, 9), dtype=np.float32)

        for b_idx, o in enumerate(obs_list):
            if len(o.voxels) == 1331:
                vox_batch[b_idx] = np.asarray(o.voxels, dtype=np.int64).reshape(11, 11, 11)
            p = o.player_state
            n_p = min(len(p), self.cfg.player_state_dim)
            play_batch[b_idx, :n_p] = p[:n_p]

            if o.inventory:
                slots = o.inventory.slots
                for s_idx in range(min(36, len(slots))):
                    s = slots[s_idx]
                    inv_batch[b_idx, s_idx] = [s.item_canonical_id, s.count, s.durability]
                armors = [o.inventory.armor_head, o.inventory.armor_chest, o.inventory.armor_legs, o.inventory.armor_feet]
                for a_idx, a in enumerate(armors):
                    inv_batch[b_idx, 36 + a_idx] = [a.item_canonical_id, a.count, a.durability]
                oh = o.inventory.offhand
                inv_batch[b_idx, 40] = [oh.item_canonical_id, oh.count, oh.durability]

            for e_idx in range(min(self.cfg.max_entities, len(o.entities))):
                e = o.entities[e_idx]
                ent_batch[b_idx, e_idx] = [e.canonical_type_id, e.dx, e.dy, e.dz, e.vx, e.vy, e.vz, e.health, 1.0 if e.is_alive else 0.0]

            aff = o.affordances
            aff_batch[b_idx] = [
                float(aff.targeted_block_canonical_id),
                float(aff.targeted_block_distance),
                float(aff.targeted_block_face),
                1.0 if aff.can_mine_target else 0.0,
                float(aff.targeted_entity_idx),
                float(aff.light_level),
                float(aff.sky_light),
                1.0 if aff.open_container_type != "none" else 0.0,
            ]

            val = o.validity_mask
            val_batch[b_idx] = [
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

        t_vox = torch.from_numpy(vox_batch).to(self.device, non_blocking=True)
        t_play = torch.from_numpy(play_batch).to(self.device, non_blocking=True)
        t_inv = torch.from_numpy(inv_batch).to(self.device, non_blocking=True)
        t_ent = torch.from_numpy(ent_batch).to(self.device, non_blocking=True)
        t_aff = torch.from_numpy(aff_batch).to(self.device, non_blocking=True)
        t_val = torch.from_numpy(val_batch).to(self.device, non_blocking=True)

        return t_vox, t_play, t_inv, t_ent, t_aff, t_val

    @torch.inference_mode()
    def rt_step(
        self,
        obs: FullObservation,
        h: Optional[torch.Tensor] = None,
        prev_z: Optional[torch.Tensor] = None,
        prev_a: Optional[torch.Tensor] = None,
        current_skill_id: int = 0,
        skill_duration_ticks: int = 0,
        planner_snapshot: Optional[Any] = None,
    ) -> Tuple[HierarchicalAction, torch.Tensor, torch.Tensor, torch.Tensor, int, int, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Pure, zero-allocation fast inference path for the 100 Hz RT loop.
        Contains NO optimizer steps, NO replay sampling, NO disk I/O, NO sync MPC.
        """
        tensors = self._convert_obs_to_tensors(obs)
        e_t = self.encoder(
            tensors["voxels"],
            tensors["player_state"],
            tensors["inventory"],
            tensors["entities"],
            tensors["affordances"],
            tensors["validity_mask"],
        )

        if h is None:
            h = torch.zeros(1, self.cfg.recurrent_dim, device=self.device)
            prev_z = torch.zeros(1, self.cfg.latent_dim, device=self.device)
            prev_a = torch.zeros(1, self.cfg.motor_dim + self.cfg.num_primitives, device=self.device)

        next_h = self.world_model.recurrent_step(prev_z, prev_a, h)
        next_z, _, _ = self.world_model.infer_posterior(next_h, e_t)

        if skill_duration_ticks <= 0:
            skill_logits = self.skill_net(torch.cat([next_h, next_z], dim=-1))
            current_skill_id = int(torch.argmax(skill_logits, dim=-1))
            skill_duration_ticks = 6

        skill_duration_ticks -= 1
        skill_one_hot = F.one_hot(torch.tensor([current_skill_id], device=self.device), num_classes=8).float()

        # Check for async plan from PlannerWorker
        if planner_snapshot is not None and getattr(planner_snapshot, "is_valid", lambda: True)():
            best_motor = planner_snapshot.best_motor
            best_prim_idx = int(planner_snapshot.best_prim_idx)
        else:
            state = torch.cat([next_h, next_z, skill_one_hot], dim=-1)
            val_mask = torch.tensor(obs.validity_mask.valid_primitives_mask[:self.cfg.num_primitives], device=self.device).bool().unsqueeze(0)
            motor_dist, prim_dist, _ = self.actor_critic.forward_policy(state, val_mask)
            best_motor = motor_dist.mean
            best_prim_idx = int(torch.argmax(prim_dist.logits, dim=-1))

        # Format fast action
        m_vec = best_motor[0].tolist()
        motor_control = ContinuousMotorControl(
            move_x=float(max(-1.0, min(1.0, m_vec[0]))),
            move_z=float(max(-1.0, min(1.0, m_vec[1]))),
            yaw_delta=float(max(-1.0, min(1.0, m_vec[2]))),
            pitch_delta=float(max(-1.0, min(1.0, m_vec[3]))),
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

        prim_one_hot = F.one_hot(torch.tensor([best_prim_idx], device=self.device), num_classes=self.cfg.num_primitives).float()
        next_a = torch.cat([best_motor, prim_one_hot], dim=-1)

        return action, next_h, next_z, next_a, current_skill_id, skill_duration_ticks, e_t, tensors

    @torch.inference_mode()
    def rt_step_batch(
        self,
        obs_list: List[FullObservation],
        h_batch: torch.Tensor,
        prev_z_batch: torch.Tensor,
        prev_a_batch: torch.Tensor,
        skill_ids: List[int],
        skill_durations: List[int],
        planner_snapshots: Optional[List[Optional[Any]]] = None,
    ) -> Tuple[List[HierarchicalAction], torch.Tensor, torch.Tensor, torch.Tensor, List[int], List[int]]:
        """
        Batched GPU inference for all peer bots in a single kernel launch.
        """
        batch_size = len(obs_list)
        if batch_size == 0:
            return [], h_batch, prev_z_batch, prev_a_batch, skill_ids, skill_durations

        t_vox, t_play, t_inv, t_ent, t_aff, t_val = self._convert_obs_batch_to_tensors(obs_list)
        e_t = self.encoder(t_vox, t_play, t_inv, t_ent, t_aff, t_val)

        next_h = self.world_model.recurrent_step(prev_z_batch, prev_a_batch, h_batch)
        next_z, _, _ = self.world_model.infer_posterior(next_h, e_t)

        next_skill_ids = list(skill_ids)
        next_skill_durations = list(skill_durations)

        # Update skills
        need_update = [i for i, d in enumerate(next_skill_durations) if d <= 0]
        if need_update:
            sub_state = torch.cat([next_h[need_update], next_z[need_update]], dim=-1)
            new_logits = self.skill_net(sub_state)
            new_ids = torch.argmax(new_logits, dim=-1).tolist()
            for idx, new_id in zip(need_update, new_ids):
                next_skill_ids[idx] = new_id
                next_skill_durations[idx] = 6

        for i in range(batch_size):
            next_skill_durations[i] -= 1

        skill_tensor = F.one_hot(torch.tensor(next_skill_ids, device=self.device), num_classes=8).float()
        combined_state = torch.cat([next_h, next_z, skill_tensor], dim=-1)

        val_masks = torch.stack([
            torch.tensor(o.validity_mask.valid_primitives_mask[:self.cfg.num_primitives], device=self.device).bool()
            for o in obs_list
        ], dim=0)

        motor_dist, prim_dist, _ = self.actor_critic.forward_policy(combined_state, val_masks)
        best_motors = motor_dist.mean
        best_prims = torch.argmax(prim_dist.logits, dim=-1)

        # Override with planner snapshots where valid
        if planner_snapshots is not None:
            for i, snap in enumerate(planner_snapshots):
                if snap is not None and getattr(snap, "is_valid", lambda: True)():
                    best_motors[i] = snap.best_motor[0]
                    best_prims[i] = int(snap.best_prim_idx)

        actions = []
        for i in range(batch_size):
            m_vec = best_motors[i].tolist()
            motor = ContinuousMotorControl(
                move_x=float(max(-1.0, min(1.0, m_vec[0]))),
                move_z=float(max(-1.0, min(1.0, m_vec[1]))),
                yaw_delta=float(max(-1.0, min(1.0, m_vec[2]))),
                pitch_delta=float(max(-1.0, min(1.0, m_vec[3]))),
                jump=bool(m_vec[4] > 0.0),
                sprint=bool(m_vec[5] > 0.0),
                sneak=bool(m_vec[6] > 0.0),
            )
            p_idx = int(best_prims[i])
            prim_name = IDX_TO_PRIMITIVE.get(p_idx, ActionPrimitive.NOOP.value)
            cmd = DiscreteActionCommand(
                primitive=ActionPrimitive(prim_name),
                target_entity_idx=obs_list[i].affordances.targeted_entity_idx if obs_list[i].affordances.targeted_entity_idx >= 0 else 0,
                target_slot=obs_list[i].inventory.selected_hotbar_slot if obs_list[i].inventory else 0,
                duration_ticks=1,
            )
            actions.append(HierarchicalAction(motor=motor, command=cmd))

        prim_one_hots = F.one_hot(best_prims, num_classes=self.cfg.num_primitives).float()
        next_a_batch = torch.cat([best_motors, prim_one_hots], dim=-1)

        return actions, next_h, next_z, next_a_batch, next_skill_ids, next_skill_durations

    def step(self, obs: FullObservation, agent_id: Optional[str] = None) -> Tuple[HierarchicalAction, Dict[str, float]]:
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
            "agent_id": agent_id,
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
