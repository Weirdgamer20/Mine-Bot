import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List

class VoxelEncoder(nn.Module):
    """Encodes 11x11x11 canonical block IDs around the agent using 3D convolutions."""
    def __init__(self, vocab_size: int = 1200, emb_dim: int = 32, out_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.conv = nn.Sequential(
            nn.Conv3d(emb_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1), # [B, 64, 6, 6, 6]
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(2),                                # [B, 64, 2, 2, 2] = 512
        )
        self.proj = nn.Linear(512, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x.long()).permute(0, 4, 1, 2, 3)
        feat = self.conv(emb).flatten(1)
        return self.proj(feat)

class MultiModalObservationEncoder(nn.Module):
    """
    Fuses terrain voxels, kinematic player state, full inventory (main + armor + offhand),
    nearby entities, mechanical affordances, and validity masks into a unified latent vector e_t.
    """
    def __init__(
        self,
        voxel_vocab: int = 1200,
        voxel_emb_dim: int = 32,
        item_vocab: int = 1500,
        item_emb_dim: int = 32,
        entity_vocab: int = 150,
        entity_emb_dim: int = 32,
        player_state_dim: int = 28,
        affordance_dim: int = 8,
        validity_mask_dim: int = 9,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.voxel_encoder = VoxelEncoder(voxel_vocab, voxel_emb_dim, 64)
        
        # Player state vector
        self.player_encoder = nn.Sequential(
            nn.Linear(player_state_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

        # Inventory (41 slots total: 36 main + 4 armor + 1 offhand)
        self.item_embedding = nn.Embedding(item_vocab, item_emb_dim, padding_idx=0)
        self.slot_mlp = nn.Sequential(
            nn.Linear(item_emb_dim + 2, 32), # item_emb + count + durability
            nn.ReLU(),
        )
        self.inv_pool = nn.Linear(64, 64) # mean + max pooled

        # Entities (up to 16 closest)
        self.entity_embedding = nn.Embedding(entity_vocab, entity_emb_dim, padding_idx=0)
        self.entity_mlp = nn.Sequential(
            nn.Linear(entity_emb_dim + 8, 32), # entity_emb + dx,dy,dz,vx,vy,vz,health,is_alive
            nn.ReLU(),
        )
        self.entity_pool = nn.Linear(64, 64)

        # Affordances & Validity Masks
        self.affordance_encoder = nn.Sequential(
            nn.Linear(affordance_dim + validity_mask_dim, 32),
            nn.ReLU(),
        )

        # Total fusion: 64 + 64 + 64 + 64 + 32 = 288 -> hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(288, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        voxels: torch.Tensor,
        player_state: torch.Tensor,
        inventory: torch.Tensor,
        entities: torch.Tensor,
        affordances: torch.Tensor,
        validity_mask: torch.Tensor,
    ) -> torch.Tensor:
        e_vox = self.voxel_encoder(voxels)
        e_play = self.player_encoder(player_state)

        # Inventory [B, 41, 3]
        item_ids = inventory[..., 0].long().clamp(0, self.item_embedding.num_embeddings - 1)
        item_embs = self.item_embedding(item_ids)
        item_feats = torch.cat([item_embs, inventory[..., 1:]], dim=-1)
        slot_repr = self.slot_mlp(item_feats)
        inv_mean = slot_repr.mean(dim=1)
        inv_max, _ = slot_repr.max(dim=1)
        e_inv = self.inv_pool(torch.cat([inv_mean, inv_max], dim=-1))

        # Entities [B, 16, 9]
        ent_ids = entities[..., 0].long().clamp(0, self.entity_embedding.num_embeddings - 1)
        ent_embs = self.entity_embedding(ent_ids)
        ent_feats = torch.cat([ent_embs, entities[..., 1:]], dim=-1)
        ent_repr = self.entity_mlp(ent_feats)
        ent_mean = ent_repr.mean(dim=1)
        ent_max, _ = ent_repr.max(dim=1)
        e_ent = self.entity_pool(torch.cat([ent_mean, ent_max], dim=-1))

        # Affordances & validity
        aff_all = torch.cat([affordances, validity_mask], dim=-1)
        e_aff = self.affordance_encoder(aff_all)

        # Fuse
        return self.fusion(torch.cat([e_vox, e_play, e_inv, e_ent, e_aff], dim=-1))

class RecurrentWorldModel(nn.Module):
    """
    Recurrent State Space Model (RSSM).
    Maintains deterministic recurrent state h_t and stochastic latent state z_t.
    Predicts:
      - Prior dynamics p(z_t | h_t)
      - Posterior latent q(z_t | h_t, e_t)
      - Observation feature reconstruction e_hat_t
      - Survival / continuation probability c_t in [0, 1] (LIVE)
      - Intrinsic reward / environmental consequence r_t
    """
    def __init__(self, hidden_dim: int = 256, latent_dim: int = 64, action_dim: int = 34):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        # Deterministic recurrent GRU: h_t = GRU([z_(t-1), a_(t-1)], h_(t-1))
        self.rnn = nn.GRUCell(latent_dim + action_dim, hidden_dim)

        # Prior dynamics: p(z_t | h_t)
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )

        # Posterior inference: q(z_t | h_t, e_t)
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )

        # Observation reconstructor: e_hat_t
        self.obs_reconstructor = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Continuation (survival): c_t in [0, 1]
        self.continuation_head = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Reward / consequence predictor
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Action consequence predictor: P(action succeeds | h_t, z_t) in [0, 1]
        self.action_success_head = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def recurrent_step(self, prev_z: torch.Tensor, prev_action: torch.Tensor, prev_h: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([prev_z, prev_action], dim=-1)
        return self.rnn(inputs, prev_h)

    def get_distribution(self, stats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = torch.chunk(stats, 2, dim=-1)
        log_std = torch.clamp(log_std, -5.0, 2.0)
        std = torch.exp(log_std)
        noise = torch.randn_like(mean)
        sample = mean + std * noise
        return sample, mean, std

    def infer_posterior(self, h: torch.Tensor, e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stats = self.posterior_net(torch.cat([h, e], dim=-1))
        return self.get_distribution(stats)

    def predict_prior(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stats = self.prior_net(h)
        return self.get_distribution(stats)

    def predict_continuation(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        logits = self.continuation_head(torch.cat([h, z], dim=-1))
        return torch.sigmoid(logits)

    def predict_reward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.reward_head(torch.cat([h, z], dim=-1))

    def predict_action_success(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Predicts probability P(action succeeds | h, z) in [0, 1]."""
        logits = self.action_success_head(torch.cat([h, z], dim=-1))
        return torch.sigmoid(logits)

    def reconstruct_obs(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.obs_reconstructor(torch.cat([h, z], dim=-1))

class RNDCuriosity(nn.Module):
    """
    Random Network Distillation (RND) intrinsic curiosity engine.
    Computes curiosity reward as scaled distillation error between a fixed randomized
    target network and a trainable predictor network.
    """
    def __init__(self, in_dim: int = 256, out_dim: int = 64, scale: float = 50.0):
        super().__init__()
        self.scale = scale
        self.target = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )
        for p in self.target.parameters():
            p.requires_grad = False

        self.predictor = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            target_feat = self.target(e)
        pred_feat = self.predictor(e)
        raw_error = F.mse_loss(pred_feat, target_feat, reduction='none').mean(dim=-1)
        # Scale curiosity so the signal is prominent and numerically meaningful (~0.5 to 2.5)
        scaled_reward = raw_error * self.scale
        return scaled_reward, pred_feat

    def compute_distillation_loss(self, e: torch.Tensor) -> torch.Tensor:
        """Computes MSE loss for training the predictor to distill the target network."""
        with torch.no_grad():
            target_feat = self.target(e)
        pred_feat = self.predictor(e)
        return F.mse_loss(pred_feat, target_feat)

class SkillDiscovery(nn.Module):
    """
    DIAYN skill discovery module.
    Maximizes mutual information I(S; Z) between skill index s and state latent [h, z].
    """
    def __init__(self, state_dim: int = 320, num_skills: int = 8):
        super().__init__()
        self.num_skills = num_skills
        self.classifier = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_skills),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.classifier(state)

    def compute_skill_diversity_reward(self, state: torch.Tensor, skill_idx: torch.Tensor) -> torch.Tensor:
        """
        DIAYN intrinsic reward: log q(s | z) - log p(s)
        Encourages the policy to visit states that make the skill easily distinguishable.
        """
        logits = self.forward(state)
        log_probs = F.log_softmax(logits, dim=-1)
        log_prior = -torch.log(torch.tensor(float(self.num_skills), device=state.device))
        if skill_idx.dim() == 1:
            skill_idx = skill_idx.unsqueeze(-1)
        selected_log_prob = log_probs.gather(-1, skill_idx).squeeze(-1)
        return selected_log_prob - log_prior

class HierarchicalActorCritic(nn.Module):
    """
    Hierarchical Policy & Value Function:
      - Motor Head: Continuous locomotion (move_x, move_z, yaw, pitch, jump, sprint, sneak)
      - Primitive Head: Categorical distribution over discrete action primitives with validity masking
      - Parameter Heads: Discrete target entity (0..15), target slot (0..35), destination slot (0..35)
      - Value Head: Critic V(h, z, skill)
    """
    def __init__(
        self,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        num_skills: int = 8,
        motor_dim: int = 7,
        num_primitives: int = 27,
    ):
        super().__init__()
        in_dim = hidden_dim + latent_dim + num_skills
        self.num_primitives = num_primitives

        # Shared representation trunk
        self.actor_trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # 1. Continuous Motor Head (move_x, move_z, yaw_delta, pitch_delta, jump, sprint, sneak)
        self.motor_mean = nn.Linear(hidden_dim, motor_dim)
        init_log_std = torch.zeros(motor_dim)
        init_log_std[2] = -1.2  # std ≈ 0.30 for yaw: avoid extreme spinning jitter during exploration
        init_log_std[3] = -1.5  # std ≈ 0.22 for pitch: stable horizon gaze
        self.motor_log_std = nn.Parameter(init_log_std)
        with torch.no_grad():
            # Beginner gamer locomotion prior:
            self.motor_mean.bias[1] = 0.6   # move_z: forward exploration bias
            self.motor_mean.bias[3] = 0.0   # pitch: horizontal gaze bias
            self.motor_mean.bias[4] = -0.4  # jump: avoid constant frantic hopping
            self.motor_mean.bias[5] = 0.3   # sprint: forward momentum
            self.motor_mean.bias[6] = -1.5  # sneak: avoid permanent crawl-lock

        # 2. Discrete Primitive Head
        self.primitive_logits = nn.Linear(hidden_dim, num_primitives)

        # Value Head (Critic)
        self.critic_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward_policy(
        self, state: torch.Tensor, validity_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.distributions.Normal, torch.distributions.Categorical, Dict[str, Any]]:
        feat = self.actor_trunk(state)

        # Continuous motor distribution
        mean = self.motor_mean(feat)
        std = torch.exp(torch.clamp(self.motor_log_std, -2.5, 0.2))
        motor_dist = torch.distributions.Normal(mean, std)

        # Discrete primitive logits with validity masking
        prim_logits = self.primitive_logits(feat)
        if validity_mask is not None:
            # Mask out invalid primitives with -1e9
            mask = validity_mask.bool()
            prim_logits = prim_logits.masked_fill(~mask, -1e9)
        prim_dist = torch.distributions.Categorical(logits=prim_logits)

        return motor_dist, prim_dist, {}

    def forward_value(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic_net(state)
