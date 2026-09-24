import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

class VoxelEncoder(nn.Module):
    """Encodes 11x11x11 block IDs around player using 3D convolutions."""
    def __init__(self, vocab_size: int = 4096, emb_dim: int = 32, out_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.conv = nn.Sequential(
            nn.Conv3d(emb_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),  # [B, 64, 6, 6, 6]
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(2),                                 # [B, 64, 2, 2, 2] = 512
        )
        self.proj = nn.Linear(512, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 11, 11, 11]
        emb = self.embedding(x.long()).permute(0, 4, 1, 2, 3)  # [B, C, D, H, W]
        feat = self.conv(emb).flatten(1)
        return self.proj(feat)

class MultiModalObservationEncoder(nn.Module):
    """
    Fuses 3D voxel terrain, player physical state, inventory slots, 
    nearby entities, and mechanical affordances into a unified observation embedding e_t.
    """
    def __init__(
        self,
        voxel_vocab: int = 4096,
        voxel_emb_dim: int = 32,
        item_vocab: int = 2048,
        item_emb_dim: int = 32,
        entity_vocab: int = 256,
        entity_emb_dim: int = 32,
        player_state_dim: int = 17,
        affordance_dim: int = 9,
        hidden_dim: int = 256,
    ):
        super().__init__()
        # Voxel grid (11x11x11)
        self.voxel_encoder = VoxelEncoder(voxel_vocab, voxel_emb_dim, 64)
        
        # Player state vector
        self.player_encoder = nn.Sequential(
            nn.Linear(player_state_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        
        # Inventory slots [B, 36, 3] (item_id, count, durability)
        self.item_embedding = nn.Embedding(item_vocab, item_emb_dim, padding_idx=0)
        self.slot_mlp = nn.Sequential(
            nn.Linear(item_emb_dim + 2, 32),
            nn.ReLU(),
        )
        self.inv_pool = nn.Linear(64, 64) # mean + max pooled (32 + 32)
        
        # Nearby entities [B, 16, 9] (type_id, dx, dy, dz, vx, vy, vz, health, is_alive)
        self.entity_embedding = nn.Embedding(entity_vocab, entity_emb_dim, padding_idx=0)
        self.entity_mlp = nn.Sequential(
            nn.Linear(entity_emb_dim + 8, 32),
            nn.ReLU(),
        )
        self.entity_pool = nn.Linear(64, 64) # mean + max pooled
        
        # Mechanical affordances
        self.affordance_encoder = nn.Sequential(
            nn.Linear(affordance_dim, 32),
            nn.ReLU(),
        )
        
        # Total fusion: 64 (vox) + 64 (player) + 64 (inv) + 64 (ent) + 32 (aff) = 288
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
    ) -> torch.Tensor:
        # 1. Voxels: [B, 11, 11, 11] -> [B, 64]
        e_vox = self.voxel_encoder(voxels)
        
        # 2. Player: [B, 17] -> [B, 64]
        e_play = self.player_encoder(player_state)
        
        # 3. Inventory: [B, 36, 3]
        item_ids = inventory[..., 0].long().clamp(0, self.item_embedding.num_embeddings - 1)
        item_embs = self.item_embedding(item_ids)  # [B, 36, item_emb_dim]
        item_feats = torch.cat([item_embs, inventory[..., 1:]], dim=-1)  # [B, 36, item_emb_dim + 2]
        slot_repr = self.slot_mlp(item_feats)  # [B, 36, 32]
        inv_mean = slot_repr.mean(dim=1)
        inv_max, _ = slot_repr.max(dim=1)
        e_inv = self.inv_pool(torch.cat([inv_mean, inv_max], dim=-1))  # [B, 64]
        
        # 4. Entities: [B, 16, 9]
        ent_ids = entities[..., 0].long().clamp(0, self.entity_embedding.num_embeddings - 1)
        ent_embs = self.entity_embedding(ent_ids)
        ent_feats = torch.cat([ent_embs, entities[..., 1:]], dim=-1)
        ent_repr = self.entity_mlp(ent_feats)  # [B, 16, 32]
        ent_mean = ent_repr.mean(dim=1)
        ent_max, _ = ent_repr.max(dim=1)
        e_ent = self.entity_pool(torch.cat([ent_mean, ent_max], dim=-1))  # [B, 64]
        
        # 5. Affordances: [B, 9] -> [B, 32]
        e_aff = self.affordance_encoder(affordances)
        
        # 6. Fuse all
        concat = torch.cat([e_vox, e_play, e_inv, e_ent, e_aff], dim=-1)
        return self.fusion(concat)

class RecurrentWorldModel(nn.Module):
    """
    Recurrent State Space Model (RSSM) inspired by DreamerV3.
    Maintains deterministic recurrent state h_t and stochastic latent state z_t.
    Predicts:
      - Prior latent dynamics p(z_t | h_t)
      - Posterior latent inference q(z_t | h_t, e_t)
      - Observation feature reconstruction e_hat_t
      - Survival / Continuation probability c_t in [0, 1] (agent learning to LIVE)
      - Transition reward / consequence
    """
    def __init__(self, hidden_dim: int = 256, latent_dim: int = 64, action_dim: int = 12):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
        # Deterministic recurrent cell: h_t = GRU([z_(t-1), a_(t-1)], h_(t-1))
        self.rnn = nn.GRUCell(latent_dim + action_dim, hidden_dim)
        
        # Prior dynamics head: p(z_t | h_t) -> (mean, log_std)
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )
        
        # Posterior inference head: q(z_t | h_t, e_t) -> (mean, log_std)
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )
        
        # Observation feature reconstruction: e_hat_t from [h_t, z_t]
        self.obs_reconstructor = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Continuation predictor (predicts survival vs death): c_t in [0, 1]
        self.continuation_head = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        
        # Consequence / intrinsic reward predictor
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def recurrent_step(self, prev_z: torch.Tensor, prev_action: torch.Tensor, prev_h: torch.Tensor) -> torch.Tensor:
        """Computes deterministic recurrent transition: h_t = GRU([z_(t-1), a_(t-1)], h_(t-1))."""
        inputs = torch.cat([prev_z, prev_action], dim=-1)
        return self.rnn(inputs, prev_h)

    def get_distribution(self, stats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Splits statistics into mean and std, sampling with reparameterization trick."""
        mean, log_std = torch.chunk(stats, 2, dim=-1)
        log_std = torch.clamp(log_std, -5.0, 2.0)
        std = torch.exp(log_std)
        noise = torch.randn_like(mean)
        sample = mean + std * noise
        return sample, mean, std

    def infer_posterior(self, h: torch.Tensor, e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """q(z_t | h_t, e_t)"""
        stats = self.posterior_net(torch.cat([h, e], dim=-1))
        return self.get_distribution(stats)

    def predict_prior(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """p(z_t | h_t) - used during latent imagination without observations"""
        stats = self.prior_net(h)
        return self.get_distribution(stats)

    def predict_continuation(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """c_t = sigmoid(W [h_t, z_t]) in (0, 1)"""
        logits = self.continuation_head(torch.cat([h, z], dim=-1))
        return torch.sigmoid(logits)

    def predict_reward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.reward_head(torch.cat([h, z], dim=-1))

    def reconstruct_obs(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.obs_reconstructor(torch.cat([h, z], dim=-1))

class RNDCuriosity(nn.Module):
    """
    Random Network Distillation (RND) for intrinsic novelty exploration.
    - Target network: randomly initialized, weights frozen.
    - Predictor network: trained to distill target network outputs.
    - Error ||phi_pred(e_t) - phi_target(e_t)||^2 rewards visiting novel Minecraft states.
    """
    def __init__(self, in_dim: int = 256, out_dim: int = 64):
        super().__init__()
        # Target network (fixed random projection)
        self.target = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )
        for p in self.target.parameters():
            p.requires_grad = False
            
        # Predictor network (trainable)
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
        intrinsic_reward = F.mse_loss(pred_feat, target_feat, reduction='none').mean(dim=-1)
        return intrinsic_reward, pred_feat

class SkillDiscovery(nn.Module):
    """
    Discovers discrete behavioral modes / temporal abstraction skills s in {0, ..., num_skills-1}.
    Provides skill conditioning to the policy.
    """
    def __init__(self, state_dim: int = 320, num_skills: int = 8):
        super().__init__()
        self.num_skills = num_skills
        self.classifier = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_skills),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.classifier(state)

class ActorCritic(nn.Module):
    """
    Actor-Critic policy operating on latent state [h_t, z_t, skill_one_hot].
    - Actor: outputs continuous means & log_stds for 12 physical action dimensions.
    - Critic: predicts expected cumulative returns V(h_t, z_t, s_t).
    """
    def __init__(self, hidden_dim: int = 256, latent_dim: int = 64, num_skills: int = 8, action_dim: int = 12):
        super().__init__()
        in_dim = hidden_dim + latent_dim + num_skills
        
        # Policy Network
        self.actor_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Value Network (Critic)
        self.critic_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward_policy(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.distributions.Normal]:
        feat = self.actor_net(state)
        mean = self.actor_mean(feat)
        std = torch.exp(torch.clamp(self.actor_log_std, -2.0, 0.5))
        dist = torch.distributions.Normal(mean, std)
        return mean, dist

    def forward_value(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic_net(state)
