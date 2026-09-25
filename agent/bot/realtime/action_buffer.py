import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, Deque, List

from ..schemas import ContinuousMotorControl, DiscreteActionCommand, HierarchicalAction
from ..experience import ActionEnvelope


@dataclass
class PendingDiscreteAction:
    action_envelope: ActionEnvelope
    enqueued_ns: int = field(default_factory=time.perf_counter_ns)
    status: str = "QUEUED"  # QUEUED, SENT, STARTED, COMPLETED, CANCELLED, TIMEOUT


class ActionBuffer:
    """
    Decoupled action publication buffer for each bot peer:
    1. Continuous controls (forward, strafe, pitch, yaw, jump, sprint, sneak):
       Latest-value buffer updated every 10ms.
    2. Discrete commands (dig, place, craft, attack, sleep, container):
       Serialized FIFO queue with action IDs, lifecycle states, and cancellation.
    """

    def __init__(self, agent_id: str, max_queue_depth: int = 16):
        self.agent_id = agent_id
        self.max_queue_depth = max_queue_depth
        self.latest_continuous: Optional[ContinuousMotorControl] = None
        self.discrete_queue: Deque[PendingDiscreteAction] = deque(maxlen=max_queue_depth)
        self.active_discrete_action: Optional[PendingDiscreteAction] = None
        self._action_id_counter: int = 1

    def generate_action_id(self) -> int:
        aid = self._action_id_counter
        self._action_id_counter += 1
        return aid

    def publish_continuous(self, motor: ContinuousMotorControl):
        self.latest_continuous = motor

    def enqueue_discrete(self, envelope: ActionEnvelope) -> int:
        pending = PendingDiscreteAction(action_envelope=envelope)
        self.discrete_queue.append(pending)
        return envelope.action_id

    def cancel_active_action(self) -> Optional[int]:
        if self.active_discrete_action and self.active_discrete_action.status in ("QUEUED", "SENT", "STARTED"):
            self.active_discrete_action.status = "CANCELLED"
            return self.active_discrete_action.action_envelope.action_id
        return None

    def pop_next_discrete(self) -> Optional[ActionEnvelope]:
        if self.discrete_queue:
            self.active_discrete_action = self.discrete_queue.popleft()
            self.active_discrete_action.status = "SENT"
            return self.active_discrete_action.action_envelope
        return None
