from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .schemas import ActionPrimitive, FullObservation, HierarchicalAction


@dataclass(frozen=True)
class PersonalityProfile:
    """Persistent behavioral constitution for one equal peer agent.

    Personality is not a Minecraft strategy table. It supplies general behavioral
    dispositions; the learned policy/world model remains responsible for tactics.
    """

    agent_id: str
    name: str
    curiosity: float
    experimentation: float
    risk_tolerance: float
    persistence: float
    confrontation: float
    avoidance: float
    exploitation: float
    adaptability: float
    skill_temperature: float
    warrior_confrontation: bool = False


PERSONALITIES = {
    "LB-01": PersonalityProfile("LB-01", "EXPLORER", 1.00, 0.90, 0.70, 0.65, 0.35, 0.35, 0.40, 0.85, 1.35),
    "LB-02": PersonalityProfile("LB-02", "SURVIVOR", 0.55, 0.35, 0.20, 0.90, 0.45, 0.90, 0.70, 0.75, 0.70),
    "LB-03": PersonalityProfile("LB-03", "WARRIOR", 0.70, 0.85, 1.00, 1.00, 1.00, 0.00, 0.45, 0.75, 0.75, True),
    "LB-04": PersonalityProfile("LB-04", "OPPORTUNIST", 0.60, 0.65, 0.65, 0.60, 0.65, 0.35, 1.00, 1.00, 1.00),
}


def get_personality(agent_id: str) -> PersonalityProfile:
    if agent_id not in PERSONALITIES:
        raise ValueError(f"Unknown agent_id={agent_id!r}; expected {sorted(PERSONALITIES)}")
    return PERSONALITIES[agent_id]


def choose_skill(skill_logits, profile: PersonalityProfile, previous_skill: int, duration: int):
    """Personality changes skill exploration/commitment, not skill semantics."""
    import torch
    import torch.nn.functional as F

    if duration > 0:
        return previous_skill
    temperature = max(profile.skill_temperature, 0.05)
    probabilities = F.softmax(skill_logits / temperature, dim=-1)
    return int(torch.multinomial(probabilities, 1).item())


def apply_personality(
    action: HierarchicalAction,
    obs: FullObservation,
    profile: PersonalityProfile,
) -> Tuple[HierarchicalAction, str]:
    """Apply personality-level constraints after learned planning.

    LB-03's explicit constitutional rule: an immediately attackable entity is
    confronted rather than avoided. The combat technique itself remains learned.
    """
    if not profile.warrior_confrontation:
        return action, "policy"

    if not obs.validity_mask.can_attack_entity or obs.affordances.targeted_entity_idx < 0:
        return action, "policy"

    action.command.primitive = ActionPrimitive.ATTACK_ENTITY
    action.command.target_entity_idx = int(obs.affordances.targeted_entity_idx)
    action.command.duration_ticks = 1
    return action, "warrior_confrontation"


def personality_metrics(profile: PersonalityProfile) -> dict:
    return {
        "personality_curiosity": profile.curiosity,
        "personality_experimentation": profile.experimentation,
        "personality_risk_tolerance": profile.risk_tolerance,
        "personality_persistence": profile.persistence,
        "personality_confrontation": profile.confrontation,
        "personality_avoidance": profile.avoidance,
        "personality_exploitation": profile.exploitation,
        "personality_adaptability": profile.adaptability,
    }
