from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .agent import LearningAgent
from .config import Config
from .personality import PERSONALITIES, PersonalityProfile, apply_personality


@dataclass
class AgentContext:
    """Per-bot state; all neural parameters and replay are shared."""

    agent_id: str
    personality: PersonalityProfile
    h: object = None
    prev_z: object = None
    prev_a: object = None
    prev_transition_data: object = None
    prev_latent: object = None
    current_skill_id: int = 0
    skill_duration_ticks: int = 0
    episode_steps: int = 0
    episode_count: int = 0


class MultiAgentLearningSystem:
    """Four equal autonomous peers using one shared learner."""

    AGENT_IDS = tuple(PERSONALITIES.keys())

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self.shared = LearningAgent(self.cfg)
        self.contexts: Dict[str, AgentContext] = {
            agent_id: AgentContext(agent_id, profile)
            for agent_id, profile in PERSONALITIES.items()
        }

    @property
    def device(self):
        return self.shared.device

    @property
    def total_steps(self) -> int:
        return self.shared.total_steps

    def register(self, agent_id: str) -> AgentContext:
        if agent_id not in self.contexts:
            raise ValueError(
                f"Unknown agent_id={agent_id!r}. Expected {list(self.AGENT_IDS)}"
            )
        return self.contexts[agent_id]

    def _activate(self, ctx: AgentContext) -> None:
        s = self.shared
        s.h, s.prev_z, s.prev_a = ctx.h, ctx.prev_z, ctx.prev_a
        s.prev_transition_data, s.prev_latent = ctx.prev_transition_data, ctx.prev_latent
        s.current_skill_id, s.skill_duration_ticks = ctx.current_skill_id, ctx.skill_duration_ticks
        s.episode_steps, s.episode_count = ctx.episode_steps, ctx.episode_count

    def _capture(self, ctx: AgentContext) -> None:
        s = self.shared
        ctx.h, ctx.prev_z, ctx.prev_a = s.h, s.prev_z, s.prev_a
        ctx.prev_transition_data, ctx.prev_latent = s.prev_transition_data, s.prev_latent
        ctx.current_skill_id, ctx.skill_duration_ticks = s.current_skill_id, s.skill_duration_ticks
        ctx.episode_steps, ctx.episode_count = s.episode_steps, s.episode_count

    def reset_agent(self, agent_id: str) -> None:
        ctx = self.register(agent_id)
        self._activate(ctx)
        self.shared.reset_episode()
        self._capture(ctx)

    def step(self, agent_id: str, obs) -> Tuple[object, dict]:
        ctx = self.register(agent_id)
        self._activate(ctx)
        action, metrics = self.shared.step(obs)

        action, source = apply_personality(action, obs, ctx.personality)

        # Keep the shared agent's next-transition action aligned with the
        # personality-adjusted primitive so the replay transition is truthful.
        if source != "policy" and self.shared.prev_a is not None:
            import torch
            import torch.nn.functional as F
            from .schemas import PRIMITIVE_TO_IDX

            idx = PRIMITIVE_TO_IDX[action.command.primitive]
            one_hot = F.one_hot(
                torch.tensor([idx], device=self.shared.device),
                num_classes=self.cfg.num_primitives,
            ).float()
            self.shared.prev_a = torch.cat(
                [self.shared.prev_a[:, : self.cfg.motor_dim], one_hot], dim=-1
            )

        metrics["agent_id"] = agent_id
        metrics["personality"] = ctx.personality.name
        metrics["personality_action_source"] = source
        metrics.update({
            "personality_curiosity": ctx.personality.curiosity,
            "personality_experimentation": ctx.personality.experimentation,
            "personality_risk_tolerance": ctx.personality.risk_tolerance,
            "personality_confrontation": ctx.personality.confrontation,
            "personality_avoidance": ctx.personality.avoidance,
            "personality_exploitation": ctx.personality.exploitation,
        })
        self._capture(ctx)
        return action, metrics

    def snapshot(self) -> dict:
        return {
            agent_id: {
                "personality": ctx.personality.name,
                "episode_steps": ctx.episode_steps,
                "episode_count": ctx.episode_count,
                "skill_id": ctx.current_skill_id,
            }
            for agent_id, ctx in self.contexts.items()
        }
