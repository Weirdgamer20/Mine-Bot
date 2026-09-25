import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from ..schemas import PRIMITIVE_TO_IDX

class LatentMPCPlanner:
    """
    Hierarchical Model Predictive Control (MPC) Planner.
    Vectorized batched GPU implementation:
    - Evaluates all N candidate trajectories in parallel [N, latent_dim]
    - Zero .item() / CPU-GPU synchronization points in the inner rollout loop
    - Uses @torch.inference_mode() for minimal overhead
    - Incorporates world-model predicted action success and empirical failure penalties
    """
    def __init__(
        self,
        world_model: nn.Module,
        actor_critic: nn.Module,
        horizon: int = 8,
        num_candidates: int = 12,
        gamma: float = 0.99,
        epsilon_uniform: float = 0.25,
        death_penalty: float = 100.0,
    ):
        self.world_model = world_model
        self.actor_critic = actor_critic
        self.horizon = horizon
        self.num_candidates = num_candidates
        self.gamma = gamma
        self.epsilon_uniform = epsilon_uniform
        self.death_penalty = death_penalty

    @torch.inference_mode()
    def plan_best_action(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        skill_one_hot: torch.Tensor,
        z_meta: Optional[torch.Tensor] = None,
        validity_mask: Optional[torch.Tensor] = None,
        experience_graph=None,
        current_latent: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, int, float]:
        """
        Samples N candidate action trajectories in parallel on GPU, simulates them
        simultaneously in latent space, and extracts the candidate yielding highest
        discounted return and survival.

        Returns: (best_motor_tensor [1, motor_dim], best_prim_idx, best_expected_value)
        """
        device = h.device
        N = self.num_candidates
        num_primitives = self.actor_critic.num_primitives
        meta_dim = getattr(self.actor_critic, "meta_dim", 32)

        if z_meta is None:
            z_meta = torch.zeros(1, meta_dim, device=device)

        # 1. Expand root state across batch dimension N
        h_batch = h.expand(N, -1)                     # [N, hidden_dim]
        z_batch = z.expand(N, -1)                     # [N, latent_dim]
        skill_batch = skill_one_hot.expand(N, -1)     # [N, num_skills]
        meta_batch = z_meta.expand(N, -1)             # [N, meta_dim]
        full_state = torch.cat([h, z, skill_one_hot, z_meta], dim=-1)

        # 2. Sample N candidate actions from policy
        motor_dist, prim_dist, _ = self.actor_critic.forward_policy(full_state, validity_mask)
        motor_cands = torch.tanh(motor_dist.sample((N,)).squeeze(1))     # [N, motor_dim] bounded [-1, 1]
        prim_cands = prim_dist.sample((N,)).squeeze(-1)      # [N]

        # 3. Epsilon-uniform mix: replace last num_uniform candidates with uniform valid primitives
        num_uniform = max(1, int(N * self.epsilon_uniform))
        if num_uniform > 0:
            if validity_mask is not None:
                valid_ids = torch.where(validity_mask)[0]
                if len(valid_ids) == 0:
                    valid_ids = torch.arange(num_primitives, device=device)
            else:
                valid_ids = torch.arange(num_primitives, device=device)
            rand_choices = torch.randint(0, len(valid_ids), (num_uniform,), device=device)
            prim_cands[-num_uniform:] = valid_ids[rand_choices]

        # 4. Construct candidate action vectors: [N, motor_dim + num_primitives]
        prim_one_hot = F.one_hot(prim_cands, num_classes=num_primitives).float()
        act_vec = torch.cat([motor_cands, prim_one_hot], dim=-1)

        # 5. Batched GPU Imagination Rollout (Zero CPU-GPU synchronization)
        sim_h = h_batch
        sim_z = z_batch
        scores = torch.zeros(N, 1, device=device)
        discount = torch.ones(N, 1, device=device)

        for t in range(self.horizon):
            sim_h = self.world_model.recurrent_step(sim_z, act_vec, sim_h)
            sim_z, _, _ = self.world_model.predict_prior(sim_h)
            pred_cont = self.world_model.predict_continuation(sim_h, sim_z)   # [N, 1]
            pred_r = self.world_model.predict_reward(sim_h, sim_z)            # [N, 1]

            # Consequence feedback: penalize actions predicted to fail
            if hasattr(self.world_model, "predict_action_success"):
                pred_succ = self.world_model.predict_action_success(sim_h, sim_z)
                pred_r = pred_r + 0.5 * (pred_succ - 1.0)

            # Catastrophic death / low continuation penalty:
            # Dying or entering near-death states destroys the candidate's imagined score
            death_risk_penalty = torch.where(pred_cont < 0.5, -self.death_penalty * (1.0 - pred_cont), 0.0)
            scores = scores + discount * (pred_r + death_risk_penalty)
            discount = discount * (self.gamma * pred_cont)

            # For subsequent steps, sample policy forward pass in batch
            if t < self.horizon - 1:
                step_state = torch.cat([sim_h, sim_z, skill_batch, meta_batch], dim=-1)
                sub_motor, sub_prim, _ = self.actor_critic.forward_policy(step_state, validity_mask)
                sub_m_act = torch.tanh(sub_motor.sample())
                sub_p_act = sub_prim.sample().squeeze(-1)
                sub_p_one_hot = F.one_hot(sub_p_act, num_classes=num_primitives).float()
                act_vec = torch.cat([sub_m_act, sub_p_one_hot], dim=-1)

        # 6. Final bootstrap value in batch
        final_state = torch.cat([sim_h, sim_z, skill_batch, meta_batch], dim=-1)
        final_val = self.actor_critic.forward_value(final_state) # [N, 1]
        scores = scores + discount * final_val

        # 7. Apply empirical failure penalty from experience graph
        if experience_graph is not None and current_latent is not None:
            candidates_eg = experience_graph.get_candidate_actions(current_latent)
            for action_name, success_rate, _ in candidates_eg:
                idx = PRIMITIVE_TO_IDX.get(action_name, -1)
                if idx >= 0 and success_rate < 0.2:
                    pen_val = -1.0 * (1.0 - success_rate)
                    mask = (prim_cands == idx).unsqueeze(-1)
                    scores = scores + mask.float() * pen_val

        # 8. Single final extraction — all intermediate work stayed on GPU
        best_idx = int(scores.squeeze(-1).argmax().item())
        best_motor = torch.tanh(motor_cands[best_idx : best_idx + 1])
        best_prim = int(prim_cands[best_idx].item())
        best_score = float(scores[best_idx].item())

        return best_motor, best_prim, best_score

        return best_motor, best_prim, best_score
