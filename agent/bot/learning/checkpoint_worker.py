from __future__ import annotations

import os
import queue
import threading
import time
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional
import torch

logger = logging.getLogger("CheckpointWorker")


@dataclass
class CheckpointTask:
    state_dicts: Dict[str, Any]
    step: int
    episode: int
    checkpoint_dir: str
    is_best: bool = False


class CheckpointWorker(threading.Thread):
    """
    Dedicated background worker for writing model checkpoints to disk asynchronously.
    Guarantees that `torch.save` and filesystem I/O NEVER block the 100 Hz RT loop
    or the online training loop.
    """

    def __init__(self, max_queue_size: int = 4):
        super().__init__(daemon=True, name="CheckpointWorker")
        self.task_queue: queue.Queue[CheckpointTask] = queue.Queue(maxsize=max_queue_size)
        self.running = False
        self.last_saved_step = 0

    def submit_checkpoint(
        self,
        state_dicts: Dict[str, Any],
        step: int,
        episode: int,
        checkpoint_dir: str,
        is_best: bool = False,
    ) -> bool:
        """Enqueues a checkpoint task. Drops older tasks if queue is full."""
        task = CheckpointTask(
            state_dicts=state_dicts,
            step=step,
            episode=episode,
            checkpoint_dir=checkpoint_dir,
            is_best=is_best,
        )
        try:
            self.task_queue.put_nowait(task)
            return True
        except queue.Full:
            logger.warning("Checkpoint queue full! Dropping checkpoint at step %d.", step)
            return False

    def run(self):
        self.running = True
        logger.info("[CheckpointWorker] Started async checkpoint thread.")

        while self.running:
            try:
                task = self.task_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                self._save_atomic(task)
                self.last_saved_step = task.step
            except Exception as e:
                logger.error("[CheckpointWorker] Failed to write checkpoint: %s", e)
            finally:
                self.task_queue.task_done()

    def _save_atomic(self, task: CheckpointTask):
        os.makedirs(task.checkpoint_dir, exist_ok=True)
        filename = f"checkpoint_step_{task.step:07d}.pt"
        target_path = os.path.join(task.checkpoint_dir, filename)
        tmp_path = target_path + ".tmp"

        payload = {
            "step": task.step,
            "episode": task.episode,
            "timestamp": time.time(),
            **task.state_dicts,
        }

        # 1. Also persist in AtomicCheckpointManager subdirectory format for full loader compatibility
        try:
            from ..training.checkpoint import AtomicCheckpointManager
            mgr = AtomicCheckpointManager(checkpoint_dir=task.checkpoint_dir)
            mgr.save_checkpoint(
                step=task.step,
                model_payload=payload,
            )
        except Exception as e:
            logger.warning("[CheckpointWorker] Failed to write subdirectory checkpoint: %s", e)

        # 2. Save flat file to tmp then atomic rename
        torch.save(payload, tmp_path)
        if os.path.exists(target_path):
            os.remove(target_path)
        os.rename(tmp_path, target_path)

        latest_link = os.path.join(task.checkpoint_dir, "checkpoint_latest.pt")
        latest_tmp = latest_link + ".tmp"
        torch.save(payload, latest_tmp)
        if os.path.exists(latest_link):
            os.remove(latest_link)
        os.rename(latest_tmp, latest_link)

        logger.info("[CheckpointWorker] Asynchronously persisted checkpoint step %d (episode %d).", task.step, task.episode)

    def stop(self):
        self.running = False
