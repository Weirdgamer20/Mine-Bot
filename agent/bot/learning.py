import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

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
    kl_clipped = torch.clamp(kl, min=free_nats)
    return kl_clipped.mean()

def train_world_model_step(
    encoder: nn.Module,
    world_model: nn.Module,
    batch: Dict[str, torch.Tensor],
    kl_weight: float = 0.1,
    continuation_weight: float = 1.0,
    rnd: Optional[nn.Module] = None,
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor, torch.Tensor]:
    """Computes RSSM sequence losses across full observation modalities."""
    voxels = batch["voxels"]               # [B, T, 11, 11, 11]
    player = batch["player_state"]         # [B, T, 18]
    inv = batch["inventory"]               # [B, T, 41, 3]
    entities = batch["entities"]           # [B, T, 16, 9]
    affordances = batch["affordances"]     # [B, T, 8]
    validity_mask = batch["validity_mask"] # [B, T, 9]
    actions = batch["actions"]             # [B, T, 34]
    rewards = batch["rewards"]             # [B, T, 1]
    continuations = batch["continuations"] # [B, T, 1]

    B, T = actions.shape[0], actions.shape[1]
    device = actions.device

    # 1. Encode all observations in the sequence
    e_all = encoder(
        voxels.reshape(-1, 11, 11, 11),
        player.reshape(-1, player.shape[-1]),
        inv.reshape(-1, inv.shape[-2], inv.shape[-1]),
        entities.reshape(-1, entities.shape[-2], entities.shape[-1]),
        affordances.reshape(-1, affordances.shape[-1]),
        validity_mask.reshape(-1, validity_mask.shape[-1]),
    ).reshape(B, T, -1)

    h = torch.zeros(B, world_model.hidden_dim, device=device)
    prev_z = torch.zeros(B, world_model.latent_dim, device=device)
    prev_a = torch.zeros(B, world_model.action_dim, device=device)

    kl_losses = []
    recon_losses = []
    cont_losses = []
    reward_losses = []
    succ_losses = []

    last_h = h
    last_z = prev_z

    for t in range(T):
        h = world_model.recurrent_step(prev_z, prev_a, h)
        z, post_mean, post_std = world_model.infer_posterior(h, e_all[:, t])
        _, prior_mean, prior_std = world_model.predict_prior(h)

        kl = compute_kl_loss(post_mean, post_std, prior_mean, prior_std)
        kl_losses.append(kl)

        e_recon = world_model.reconstruct_obs(h, z)
        recon_loss = F.mse_loss(e_recon, e_all[:, t].detach())
        recon_losses.append(recon_loss)

        c_pred = world_model.predict_continuation(h, z)
        cont_loss = F.binary_cross_entropy(c_pred, continuations[:, t])
        cont_losses.append(cont_loss)

        r_pred = world_model.predict_reward(h, z)
        r_loss = F.mse_loss(r_pred, rewards[:, t])
        reward_losses.append(r_loss)

        if "successes" in batch and hasattr(world_model, "predict_action_success"):
            succ_pred = world_model.predict_action_success(h, z)
            succ_loss = F.binary_cross_entropy(succ_pred, batch["successes"][:, t])
            succ_losses.append(succ_loss)

        prev_z = z
        prev_a = actions[:, t]
        last_h = h
        last_z = z

    total_kl = torch.stack(kl_losses).mean()
    total_recon = torch.stack(recon_losses).mean()
    total_cont = torch.stack(cont_losses).mean()
    total_reward = torch.stack(reward_losses).mean()
    total_succ = torch.stack(succ_losses).mean() if succ_losses else torch.tensor(0.0, device=device)

    wm_loss = total_recon + kl_weight * total_kl + continuation_weight * total_cont + total_reward + 0.5 * total_succ

    rnd_distill_loss = 0.0
    if rnd is not None:
        rnd_loss = rnd.compute_distillation_loss(e_all.detach())
        wm_loss = wm_loss + rnd_loss
        rnd_distill_loss = float(rnd_loss.item())

    metrics = {
        "wm_loss": float(wm_loss.item()),
        "kl_loss": float(total_kl.item()),
        "recon_loss": float(total_recon.item()),
        "cont_loss": float(total_cont.item()),
        "reward_loss": float(total_reward.item()),
        "succ_loss": float(total_succ.item()) if succ_losses else 0.0,
        "rnd_loss": rnd_distill_loss,
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
    """Trains hierarchical policy and value function in latent imagination."""
    B = start_h.shape[0]
    device = start_h.device

    state_repr = torch.cat([start_h, start_z], dim=-1)
    skill_logits = skill_net(state_repr)
    skill_one_hot = F.gumbel_softmax(skill_logits, tau=1.0, hard=True)

    h = start_h
    z = start_z

    imagined_log_probs = []
    imagined_rewards = []
    imagined_conts = []
    imagined_values = []

    for k in range(horizon):
        full_state = torch.cat([h, z, skill_one_hot], dim=-1)
        motor_dist, prim_dist, param_dists = actor_critic.forward_policy(full_state)

        motor_act = motor_dist.rsample()
        motor_log_prob = motor_dist.log_prob(motor_act).sum(-1)

        prim_act = prim_dist.sample()
        prim_log_prob = prim_dist.log_prob(prim_act)

        total_log_prob = motor_log_prob + prim_log_prob
        val = actor_critic.forward_value(full_state).squeeze(-1)

        imagined_log_probs.append(total_log_prob)
        imagined_values.append(val)

        # Construct flat action vector for world-model transition: motor(7) + primitive_one_hot(27) = 34
        prim_one_hot = F.one_hot(prim_act, num_classes=actor_critic.num_primitives).float()
        action_vec = torch.cat([motor_act, prim_one_hot], dim=-1)

        h = world_model.recurrent_step(z, action_vec, h)
        z, _, _ = world_model.predict_prior(h)
        pred_cont = world_model.predict_continuation(h, z).squeeze(-1)
        pred_reward = world_model.predict_reward(h, z).squeeze(-1)

        # DIAYN mutual information diversity bonus: log q(s | z) - log p(s)
        if hasattr(skill_net, "compute_skill_diversity_reward"):
            skill_idx = skill_one_hot.argmax(dim=-1)
            diayn_reward = skill_net.compute_skill_diversity_reward(torch.cat([h, z], dim=-1), skill_idx)
            pred_reward = pred_reward + 0.1 * diayn_reward

        # Action feasibility penalty: penalize imagined actions predicted to fail
        if hasattr(world_model, "predict_action_success"):
            pred_succ = world_model.predict_action_success(h, z).squeeze(-1)
            pred_reward = pred_reward + 0.5 * (pred_succ - 1.0)

        imagined_conts.append(pred_cont)
        imagined_rewards.append(pred_reward)

    final_state = torch.cat([h, z, skill_one_hot], dim=-1)
    final_val = actor_critic.forward_value(final_state).squeeze(-1).detach()

    returns = []
    r_k = final_val
    for t in reversed(range(horizon)):
        reward = imagined_rewards[t].detach()
        discount = gamma * imagined_conts[t].detach()
        v_next = imagined_values[t + 1].detach() if t + 1 < horizon else final_val
        r_k = reward + discount * ((1.0 - lambda_gae) * v_next + lambda_gae * r_k)
        returns.insert(0, r_k)

    returns = torch.stack(returns)
    values = torch.stack(imagined_values)
    log_probs = torch.stack(imagined_log_probs)

    advantage = returns - values.detach()
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
