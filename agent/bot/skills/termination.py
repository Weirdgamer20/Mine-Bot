import torch
import torch.nn as nn

class SkillTerminationModel(nn.Module):
    """
    Learns state-dependent skill termination probability beta(s, z_s) in [0, 1].
    Allows temporally extended skills to execute over variable durations (4-20 ticks).
    """
    def __init__(self, state_dim: int = 320, num_skills: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + num_skills, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, state: torch.Tensor, skill_one_hot: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([state, skill_one_hot], dim=-1)
        return self.net(inputs)
