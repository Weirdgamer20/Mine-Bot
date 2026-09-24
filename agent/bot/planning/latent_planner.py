import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional

class LatentMPCPlanner:
    """
    Hierarchical Model Predictive Control (MPC) Planner.
    Simulates candidate actions and skills in the learned latent world model
    to rank choices by cumulative expected return and survival probability.
    """
    def __init__(
        self,
        world_model: nn.Module,
        actor_critic: nn.Module,
        horizon: int = 8,
        num_candidates: int = 12,
        gamma: float = 0.99,
    ):
        self.world_model = world_model
        self.actor_critic = actor_critic
        self.horizon = horizon
        self.num_candidates = num_candidates
        self.gamma = gamma

    @torch.no_grad()
    def plan_best_action(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        skill_one_hot: torch.Tensor,
        validity_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int, float]:
        """
        Samples N candidate action trajectories, simulates them in latent imagination,
        and selects the candidate yielding highest discounted return and survival.
        Returns: (best_action_tensor, best_prim_idx, best_expected_value)
        """
        device = h.device
        full_state = torch.cat([h, z, skill_one_hot], dim=-1)

        # 1. Sample N candidate actions from the policy
        motor_dist, prim_dist, param_dists = self.actor_critic.forward_policy(full_state, validity_mask)

        best_score = -float("inf")
        best_motor = None
        best_prim = 0

        for _ in range(self.num_candidates):
            motor_cand = motor_dist.sample()
            prim_cand = prim_dist.sample()
            prim_idx = int(prim_cand[0].item())

            # Construct action vector: motor(7) + prim_one_hot(27) = 34
            prim_one_hot = torch.zeros(1, self.actor_critic.num_primitives, device=device)
            prim_one_hot[0, prim_idx] = 1.0
            act_vec = torch.cat([motor_cand, prim_one_hot], dim=-1)

            # Roll out H steps in imagination
            sim_h = h
            sim_z = z
            score = 0.0
            discount = 1.0

            for t in range(self.horizon):
                sim_h = self.world_model.recurrent_step(sim_z, act_vec, sim_h)
                sim_z, _, _ = self.world_model.predict_prior(sim_h)

                pred_cont = float(self.world_model.predict_continuation(sim_h, sim_z).item())
                pred_r = float(self.world_model.predict_reward(sim_h, sim_z).item())

                score += discount * pred_r
                discount *= (self.gamma * pred_cont)

                if pred_cont < 0.1: # Death predicted
                    score -= 5.0
                    break

                # For subsequent steps, sample from actor
                step_state = torch.cat([sim_h, sim_z, skill_one_hot], dim=-1)
                sub_motor, sub_prim, _ = self.actor_critic.forward_policy(step_state, validity_mask)
                sub_act = sub_motor.sample()
                sub_prim_idx = int(sub_prim.sample()[0].item())
                sub_prim_one_hot = torch.zeros(1, self.actor_critic.num_primitives, device=device)
                sub_prim_one_hot[0, sub_prim_idx] = 1.0
                act_vec = torch.cat([sub_act, sub_prim_one_hot], dim=-1)

            # Final bootstrap value
            final_val = float(self.actor_critic.forward_value(torch.cat([sim_h, sim_z, skill_one_hot], dim=-1)).item())
            score += discount * final_val

            if score > best_score:
                best_score = score
                best_motor = motor_cand
                best_prim = prim_idx

        return best_motor, best_prim, best_score
