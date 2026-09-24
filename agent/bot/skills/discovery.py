import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SkillDiscriminator(nn.Module):
    """
    DIAYN-inspired skill discovery discriminator.
    Maximizes mutual information I(S; Z) between discrete skill latent S and state latent Z:
      reward_skill = log q(s | z) - log p(s)
    Drives the emergence of distinguishable behavioral skills without external rewards.
    """
    def __init__(self, state_dim: int = 320, num_skills: int = 8):
        super().__init__()
        self.num_skills = num_skills
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_skills),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

    def compute_skill_diversity_reward(self, state: torch.Tensor, skill_idx: int) -> torch.Tensor:
        logits = self.forward(state)
        log_probs = F.log_softmax(logits, dim=-1)
        # Prior is uniform p(s) = 1 / num_skills -> log p(s) = -log(num_skills)
        log_prior = -torch.log(torch.tensor(float(self.num_skills), device=state.device))
        diversity_reward = log_probs[:, skill_idx] - log_prior
        return diversity_reward.squeeze(-1)
