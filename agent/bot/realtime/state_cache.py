import time
from typing import Dict, Optional, Tuple
from ..experience import ObservationEnvelope


class StateCache:
    """
    Lock-free latest-value observation cache for real-time controllers.
    Ensures the 100 Hz RT loop always operates on the freshest state
    without building up backlogs or queue delays.
    """

    def __init__(self, max_stale_ms: float = 250.0):
        self.max_stale_ms = max_stale_ms
        self._latest_by_agent: Dict[str, ObservationEnvelope] = {}
        self._prev_by_agent: Dict[str, ObservationEnvelope] = {}
        self._update_count: Dict[str, int] = {}

    def put(self, envelope: ObservationEnvelope):
        agent_id = envelope.agent_id
        if agent_id in self._latest_by_agent:
            self._prev_by_agent[agent_id] = self._latest_by_agent[agent_id]
        self._latest_by_agent[agent_id] = envelope
        self._update_count[agent_id] = self._update_count.get(agent_id, 0) + 1

    def get_latest(self, agent_id: str) -> Optional[ObservationEnvelope]:
        return self._latest_by_agent.get(agent_id)

    def get_previous(self, agent_id: str) -> Optional[ObservationEnvelope]:
        return self._prev_by_agent.get(agent_id)

    def is_stale(self, agent_id: str) -> bool:
        env = self._latest_by_agent.get(agent_id)
        if env is None:
            return True
        return env.age_ms > self.max_stale_ms

    def has_world_advanced(self, agent_id: str) -> bool:
        """
        True only if the world_tick has incremented since the previous observation.
        Prevents multiple 100 Hz inferences on the same 20 Hz Minecraft frame
        from being treated as multiple environment steps.
        """
        curr = self._latest_by_agent.get(agent_id)
        prev = self._prev_by_agent.get(agent_id)
        if curr is None or prev is None:
            return True
        return curr.world_tick > prev.world_tick
