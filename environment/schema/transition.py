from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
from .observation import FullObservation
from .action import HierarchicalAction, ActionResult

@dataclass
class DecomposedReward:
    """Explicitly decomposed reward and consequence signals."""
    environment: float = 0.0      # Physical state changes (e.g. inventory deltas, damage sustained)
    intrinsic_novelty: float = 0.0 # RND state novelty
    prediction_error: float = 0.0  # Epistemic world model prediction error
    skill_diversity: float = 0.0   # Empowerment / mutual information signal
    total: float = 0.0

@dataclass
class StepTransition:
    """
    Fundamental transition tuple: (s_t, a_t, r_t, s_{t+1}, d_t, c_t, result_t)
    The foundational building block of episodic memory, world models, and policy optimization.
    """
    state: FullObservation
    action: HierarchicalAction
    reward: DecomposedReward
    next_state: FullObservation
    done: bool
    continuation: float # c_t in [0.0, 1.0] (0.0 upon death)
    result: ActionResult
    episode_id: int = 0
    step_id: int = 0
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.to_dict(),
            "action": self.action.to_dict(),
            "reward": asdict(self.reward),
            "next_state": self.next_state.to_dict(),
            "done": bool(self.done),
            "continuation": float(self.continuation),
            "result": self.result.to_dict(),
            "episode_id": int(self.episode_id),
            "step_id": int(self.step_id),
            "timestamp": float(self.timestamp),
        }
