from pathlib import Path
from typing import Dict, Any, Optional
import torch
import shutil

class AtomicCheckpointManager:
    """
    Manages atomic versioned checkpoints with crash recovery.
    Prevents corrupt files by writing to a temporary file before atomic renaming.
    """
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        step: int,
        model_payload: Dict[str, Any],
        memory_payload: Optional[Dict[str, Any]] = None,
        spatial_payload: Optional[Dict[str, Any]] = None,
        skills_payload: Optional[Dict[str, Any]] = None,
    ) -> Path:
        step_dir = self.checkpoint_dir / f"step_{step:08d}"
        temp_dir = self.checkpoint_dir / f"temp_{step:08d}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save model weights & optimizers
        torch.save(model_payload, temp_dir / "model.pt")

        # 2. Save memory states
        if memory_payload:
            torch.save(memory_payload, temp_dir / "replay.pt")

        if spatial_payload:
            torch.save(spatial_payload, temp_dir / "spatial.pt")

        if skills_payload:
            torch.save(skills_payload, temp_dir / "skills.pt")

        # 3. Atomic rename
        if step_dir.exists():
            shutil.rmtree(step_dir)
        temp_dir.rename(step_dir)

        # 4. Update 'latest' pointer
        latest_file = self.checkpoint_dir / "latest_step.txt"
        latest_file.write_text(str(step))

        return step_dir

    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        latest_file = self.checkpoint_dir / "latest_step.txt"
        if not latest_file.exists():
            return None

        try:
            step = int(latest_file.read_text().strip())
            step_dir = self.checkpoint_dir / f"step_{step:08d}"
            if not step_dir.exists():
                return None

            model_data = torch.load(step_dir / "model.pt", map_location="cpu", weights_only=False)
            replay_path = step_dir / "replay.pt"
            replay_data = torch.load(replay_path, map_location="cpu", weights_only=False) if replay_path.exists() else None
            spatial_path = step_dir / "spatial.pt"
            spatial_data = torch.load(spatial_path, map_location="cpu", weights_only=False) if spatial_path.exists() else None
            skills_path = step_dir / "skills.pt"
            skills_data = torch.load(skills_path, map_location="cpu", weights_only=False) if skills_path.exists() else None

            return {
                "step": step,
                "model": model_data,
                "replay": replay_data,
                "spatial": spatial_data,
                "skills": skills_data,
            }
        except Exception as e:
            print(f"[Checkpoint] Warning: Failed to load subdirectory checkpoint: {e}")

        # Fallback: check flat checkpoint_latest.pt
        flat_latest = self.checkpoint_dir / "checkpoint_latest.pt"
        if flat_latest.exists():
            try:
                flat_data = torch.load(flat_latest, map_location="cpu", weights_only=False)
                step = flat_data.get("step", 0)
                return {
                    "step": step,
                    "model": flat_data,
                    "replay": None,
                    "spatial": None,
                    "skills": None,
                }
            except Exception as e:
                print(f"[Checkpoint] Warning: Failed to load flat latest checkpoint: {e}")

        return None
