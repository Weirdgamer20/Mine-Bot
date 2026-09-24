from enum import Enum
from typing import Dict, Any

class AblationMode(str, Enum):
    FULL = "full"
    NO_WORLD_MODEL = "no_world_model"
    NO_RND = "no_rnd"
    NO_SKILLS = "no_skills"
    NO_SPATIAL_MEMORY = "no_spatial_memory"

def apply_ablation_config(cfg, mode: AblationMode):
    """Configures agent hyperparameters for scientific ablation testing."""
    if mode == AblationMode.NO_RND:
        cfg.rnd_weight = 0.0
    elif mode == AblationMode.NO_WORLD_MODEL:
        cfg.imagination_horizon = 1
        cfg.prediction_error_weight = 0.0
    elif mode == AblationMode.NO_SKILLS:
        pass
    return cfg
