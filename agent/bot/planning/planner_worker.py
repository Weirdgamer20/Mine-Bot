import time
import threading
import torch
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

from .latent_planner import LatentMPCPlanner


@dataclass
class PlannerSnapshot:
    version: int
    generated_at_ns: int
    valid_until_ns: int
    best_motor: torch.Tensor
    best_prim_idx: int
    planned_value: float
    confidence: float = 1.0

    def is_valid(self) -> bool:
        return time.perf_counter_ns() <= self.valid_until_ns


class PlannerWorker:
    """
    Asynchronous Model Predictive Control (MPC) worker.
    Continuously runs latent imagination at 10–20 Hz in the background
    and publishes non-blocking PlannerSnapshots for the 100 Hz RT controller.
    """

    def __init__(
        self,
        planner: LatentMPCPlanner,
        target_hz: float = 15.0,
        plan_ttl_ms: float = 150.0,
    ):
        self.planner = planner
        self.target_period_s = 1.0 / target_hz
        self.plan_ttl_ns = int(plan_ttl_ms * 1_000_000)

        self._snapshots: Dict[str, PlannerSnapshot] = {}
        self._requests: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]] = {}
        self._version_counter = 1
        self._lock = threading.Lock()

        self._running = False
        self._thread: Optional[threading.Thread] = None

    def submit_state(
        self,
        agent_id: str,
        h: torch.Tensor,
        z: torch.Tensor,
        skill_one_hot: torch.Tensor,
        validity_mask: Optional[torch.Tensor] = None,
    ):
        """Called by RT controller to non-blockingly submit latest state for planning."""
        with self._lock:
            self._requests[agent_id] = (h.detach(), z.detach(), skill_one_hot.detach(), validity_mask)

    def get_latest(self, agent_id: str) -> Optional[PlannerSnapshot]:
        """Lock-free read of latest plan by the 100 Hz RT loop."""
        snap = self._snapshots.get(agent_id)
        if snap is not None and snap.is_valid():
            return snap
        return None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, name="AsyncPlannerWorker", daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _run_loop(self):
        while self._running:
            start_time = time.perf_counter()

            # Copy current requests
            with self._lock:
                current_batch = dict(self._requests)

            if current_batch:
                for agent_id, (h, z, skill, val_mask) in current_batch.items():
                    if not self._running:
                        break
                    try:
                        with torch.inference_mode():
                            best_motor, best_prim_idx, planned_value = self.planner.plan_best_action(
                                h, z, skill, validity_mask=val_mask
                            )

                            now_ns = time.perf_counter_ns()
                            snap = PlannerSnapshot(
                                version=self._version_counter,
                                generated_at_ns=now_ns,
                                valid_until_ns=now_ns + self.plan_ttl_ns,
                                best_motor=best_motor,
                                best_prim_idx=best_prim_idx,
                                planned_value=planned_value,
                            )
                            self._snapshots[agent_id] = snap
                            self._version_counter += 1
                    except Exception:
                        pass

            elapsed = time.perf_counter() - start_time
            sleep_time = max(0.001, self.target_period_s - elapsed)
            time.sleep(sleep_time)
