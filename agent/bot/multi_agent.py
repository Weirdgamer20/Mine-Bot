from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

from .agent import LearningAgent
from .config import Config
from .personality import PERSONALITIES, PersonalityProfile


class MultiAgentLearningSystem:
    """
    Four equal autonomous peers sharing one neural representation and replay buffer,
    while maintaining 100% isolated recurrent state, spatial memory, skill libraries,
    and personality profiles.
    
    Eliminates all mutable singleton multiplexing (_activate / _capture).
    """

    AGENT_IDS = tuple(PERSONALITIES.keys())

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()

        # 1. Master/Shared Agent hosting shared neural models, optimizers, and replay buffer
        self.shared = LearningAgent(self.cfg, agent_id="MASTER")
        shared_models = {
            "encoder": self.shared.encoder,
            "world_model": self.shared.world_model,
            "rnd": self.shared.rnd,
            "skill_net": self.shared.skill_net,
            "actor_critic": self.shared.actor_critic,
            "planner": self.shared.planner,
        }
        shared_optimizers = (self.shared.wm_opt, self.shared.ac_opt)

        # 2. Four first-class independent peer agent instances
        self.peers: Dict[str, LearningAgent] = {
            aid: LearningAgent(
                cfg=self.cfg,
                agent_id=aid,
                personality=profile,
                shared_models=shared_models,
                shared_memory=self.shared.memory,
                shared_optimizers=shared_optimizers,
                shared_checkpoint_manager=self.shared.checkpoint_manager,
            )
            for aid, profile in PERSONALITIES.items()
        }

        # Contexts alias for backward-compatible inspections
        self.contexts = self.peers

    @property
    def device(self):
        return self.shared.device

    @property
    def total_steps(self) -> int:
        return sum(peer.total_steps for peer in self.peers.values())

    def get_agent(self, agent_id: str) -> LearningAgent:
        if agent_id not in self.peers:
            raise ValueError(
                f"Unknown agent_id={agent_id!r}. Expected {list(self.AGENT_IDS)}"
            )
        return self.peers[agent_id]

    def register(self, agent_id: str) -> LearningAgent:
        """Backward-compatible register method returning the peer agent directly."""
        return self.get_agent(agent_id)

    def reset_agent(self, agent_id: str) -> None:
        """Resets the isolated recurrent and episode state of a single peer bot."""
        self.get_agent(agent_id).reset_episode()

    def step(self, agent_id: str, obs) -> Tuple[object, dict]:
        """
        Step an individual peer agent directly on its own isolated state.
        Zero state multiplexing, zero thread lock contention on shared mutable attributes.
        """
        peer = self.get_agent(agent_id)
        return peer.step(obs)

    def snapshot(self) -> dict:
        return {
            agent_id: {
                "personality": peer.personality.name if peer.personality else "NONE",
                "episode_steps": peer.episode_steps,
                "episode_count": peer.episode_count,
                "skill_id": peer.current_skill_id,
                "regions_explored": peer.spatial_memory.total_regions_discovered(),
            }
            for agent_id, peer in self.peers.items()
        }

