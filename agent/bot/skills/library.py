from typing import Dict, List, Any
import numpy as np

class SkillLibrary:
    """
    Persistent library cataloging discovered behavioral skills.
    Tracks initiation distributions, execution durations, usage counts, and outcome statistics.
    """
    def __init__(self, num_skills: int = 8):
        self.num_skills = num_skills
        self.skills: Dict[int, Dict[str, Any]] = {
            i: {
                "skill_id": i,
                "activation_count": 0,
                "total_duration_ticks": 0,
                "mean_consequence": 0.0,
                "success_rate": 0.5,
            }
            for i in range(num_skills)
        }

    def record_skill_execution(self, skill_id: int, duration_ticks: int, consequence_delta: float, success: bool):
        if skill_id in self.skills:
            sk = self.skills[skill_id]
            sk["activation_count"] += 1
            sk["total_duration_ticks"] += duration_ticks
            sk["mean_consequence"] = 0.9 * sk["mean_consequence"] + 0.1 * float(consequence_delta)
            succ_val = 1.0 if success else 0.0
            sk["success_rate"] = 0.9 * sk["success_rate"] + 0.1 * succ_val

    def get_skill_stats(self, skill_id: int) -> Dict[str, Any]:
        return self.skills.get(skill_id, {})

    def to_dict(self) -> Dict:
        return self.skills

    def load_from_dict(self, data: Dict):
        for k, v in data.items():
            self.skills[int(k)] = v
