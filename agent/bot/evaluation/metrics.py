from pathlib import Path
from typing import Dict, Any
import json
import time

class ExperimentMetricsLogger:
    """
    Appends scientific benchmark metrics to metrics.jsonl.
    Enables quantitative tracking of learning curves, loss convergence, and behavioral metrics.
    """
    def __init__(self, log_dir: str = "metrics", filename: str = "metrics.jsonl"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / filename

    def log(self, step: int, metrics: Dict[str, Any]):
        entry = {
            "timestamp": time.time(),
            "step": step,
            **metrics,
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
