import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

def compute_kl_loss(
    post_mean: torch.Tensor,
    post_std: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_std: torch.Tensor,
    free_nats: float = 1.0,
) -> torch.Tensor:
    """Computes KL divergence KL(q(z|h,e) || p(z|h)) with free bits clipping."""
    var_post = post_std.pow(2)
    var_prior = prior_std.pow(2)
    kl = 0.5 * (
        torch.log(var_prior / var_post)
        + (var_post + (post_mean - prior_mean).pow(2)) / var_prior
        - 1.0
    ).sum(dim=-1)
    # Free bits: prevent posterior collapse
    kl_clipped = torch.clamp(kl, min=free_nats)
    return kl_clipped.mean()

def train_world_model_step(
    encoder: nn.Module,
    world_model: nn.Module,
    batch: Dict[str, torch.Tensor],
    kl_weight: float = 0.1,
    continuation_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor, torch.Tensor]:
    """
    Computes RSSM sequence losses:
      - Dynamics KL loss
      - Latent observation reconstruction loss
      - Survival / continuation prediction BCE loss (learning to LIVE)
      - Transition reward MSE loss
    """
    voxels = batch["voxels"]               # [B, T, 11, 11, 11]
    player = batch["player_state"]         # [B, T, 17]
    inv = batch["inventory"]               # [B, T, 36, 3]
    entities = batch["entities"]           # [B, T, 16, 9]
    affordances = batch["affordances"]     # [B, T, 9]
    actions = batch["actions"]             # [B, T, 12]
    rewards = batch["rewards"]             # [B, T, 1]
    continuations = batch["continuations"] # [B, T, 1]
    
    B, T = actions.shape[0], actions.shape[1]
    device = actions.device

    # 1. Encode all observations in the sequence: [B*T, ...]
    e_all = encoder(
        voxels.reshape(-1, 11, 11, 11),
        player.reshape(-1, player.shape[-1]),
        inv.reshape(-1, inv.shape[-2], inv.shape[-1]),
        entities.reshape(-1, entities.shape[-2], entities.shape[-1]),
        affordances.reshape(-1, affordances.shape[-1]),
    ).reshape(B, T, -1)  # [B, T, hidden_dim]

    h = torch.zeros(B, world_model.hidden_dim, device=device)
    prev_z = torch.zeros(B, world_model.latent_dim, device=device)
    prev_a = torch.zeros(B, world_model.action_dim, device=device)

    kl_losses = []
    recon_losses = []
    cont_losses = []
    reward_losses = []

    last_h = h
    last_z = prev_z

    for t in range(T):
        # Deterministic recurrent update
        h = world_model.recurrent_step(prev_z, prev_a, h)
        
        # Posterior inference q(z_t | h_t, e_t)
        z, post_mean, post_std = world_model.infer_posterior(h, e_all[:, t])
        
        # Prior dynamics p(z_t | h_t)
        _, prior_mean, prior_std = world_model.predict_prior(h)
        
        # KL loss
        kl = compute_kl_loss(post_mean, post_std, prior_mean, prior_std)
        kl_losses.append(kl)
        
        # Reconstruction loss
        e_recon = world_model.reconstruct_obs(h, z)
        recon_loss = F.mse_loss(e_recon, e_all[:, t].detach())
        recon_losses.append(recon_loss)
        
        # Continuation loss (survival)
        c_pred = world_model.predict_continuation(h, z)
        cont_loss = F.binary_cross_entropy(c_pred, continuations[:, t])
        cont_losses.append(cont_loss)
        
        # Reward loss
        r_pred = world_model.predict_reward(h, z)
        r_loss = F.mse_loss(r_pred, rewards[:, t])
        reward_losses.append(r_loss)
        
        prev_z = z
        prev_a = actions[:, t]
        last_h = h
        last_z = z

    total_kl = torch.stack(kl_losses).mean()
    total_recon = torch.stack(recon_losses).mean()
    total_cont = torch.stack(cont_losses).mean()
    total_reward = torch.stack(reward_losses).mean()

    wm_loss = total_recon + kl_weight * total_kl + continuation_weight * total_cont + total_reward

    metrics = {
        "wm_loss": float(wm_loss.item()),
        "kl_loss": float(total_kl.item()),
        "recon_loss": float(total_recon.item()),
        "cont_loss": float(total_cont.item()),
        "reward_loss": float(total_reward.item()),
    }
    return wm_loss, metrics, last_h.detach(), last_z.detach()

def train_actor_critic_imagination(
    actor_critic: nn.Module,
    world_model: nn.Module,
    skill_net: nn.Module,
    start_h: torch.Tensor,
    start_z: torch.Tensor,
    horizon: int = 12,
    gamma: float = 0.99,
    lambda_gae: float = 0.95,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Trains Actor and Critic using latent trajectories imagined entirely inside the world model.
    Imagined returns combine predicted continuation (survival) with intrinsic reward.
    """
    B = start_h.shape[0]
    device = start_h.device

    # Infer or sample skill vector
    state_repr = torch.cat([start_h, start_z], dim=-1)
    skill_logits = skill_net(state_repr)
    skill_one_hot = F.gumbel_softmax(skill_logits, tau=1.0, hard=True)

    h = start_h
    z = start_z

    imagined_states = []
    imagined_actions = []
    imagined_log_probs = []
    imagined_rewards = []
    imagined_conts = []
    imagined_values = []

    for k in range(horizon):
        full_state = torch.cat([h, z, skill_one_hot], dim=-1)
        action_mean, dist = actor_critic.forward_policy(full_state)
        # Sample action with exploration noise
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        val = actor_critic.forward_value(full_state).squeeze(-1)

        imagined_states.append(full_state)
        imagined_actions.append(action)
        imagined_log_probs.append(log_prob)
        imagined_values.append(val)

        # Step forward in imagination
        h = world_model.recurrent_step(z, action, h)
        z, _, _ = world_model.predict_prior(h)
        pred_cont = world_model.predict_continuation(h, z).squeeze(-1)
        pred_reward = world_model.predict_reward(h, z).squeeze(-1)

        imagined_conts.append(pred_cont)
        imagined_rewards.append(pred_reward)

    # Final bootstrap value
    final_state = torch.cat([h, z, skill_one_hot], dim=-1)
    final_val = actor_critic.forward_value(final_state).squeeze(-1).detach()

    # Calculate GAE / lambda returns backwards
    returns = []
    r_k = final_val
    for t in reversed(range(horizon)):
        reward = imagined_rewards[t].detach()
        discount = (gamma * imagined_conts[t].detach())
        v_next = imagined_values[t+1].detach() if t + 1 < horizon else final_val
        r_k = reward + discount * ((1.0 - lambda_gae) * v_next + lambda_gae * r_k)
        returns.insert(0, r_k)

    returns = torch.stack(returns)          # [H, B]
    values = torch.stack(imagined_values)    # [H, B]
    log_probs = torch.stack(imagined_log_probs)

    advantage = (returns - values.detach())
    policy_loss = -(log_probs * advantage).mean()
    value_loss = F.mse_loss(values, returns)

    total_ac_loss = policy_loss + 0.5 * value_loss

    metrics = {
        "ac_loss": float(total_ac_loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "mean_imagined_return": float(returns.mean().item()),
    }
    return total_ac_loss, metrics
