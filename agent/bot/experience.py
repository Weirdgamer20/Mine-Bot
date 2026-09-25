from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, Tuple

from .schemas import FullObservation, HierarchicalAction, ActionResult, DecomposedReward


class ValidationStatus(str, Enum):
    VALID = "VALID"
    DUPLICATE = "DUPLICATE"
    STALE = "STALE"
    OUT_OF_ORDER = "OUT_OF_ORDER"
    UNKNOWN_OUTCOME = "UNKNOWN_OUTCOME"
    ACTION_TIMEOUT = "ACTION_TIMEOUT"
    MISSING_STATE = "MISSING_STATE"
    CLOCK_ERROR = "CLOCK_ERROR"


@dataclass
class ObservationEnvelope:
    """Envelope wrapping a raw observation with strict temporal provenance."""
    sequence: int
    world_tick: int
    timestamp_ns: int
    received_ns: int
    agent_id: str
    observation: FullObservation

    @property
    def age_ms(self) -> float:
        """State age in milliseconds since original generation."""
        now_ns = time.perf_counter_ns()
        return max(0.0, (now_ns - self.timestamp_ns) / 1_000_000.0)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ObservationEnvelope:
        obs_raw = data.get("observation", {})
        return cls(
            sequence=int(data.get("sequence", 0)),
            world_tick=int(data.get("world_tick", 0)),
            timestamp_ns=int(data.get("timestamp_ns", 0) or (data.get("timestamp", 0) * 1_000_000_000)),
            received_ns=time.perf_counter_ns(),
            agent_id=str(data.get("agent_id", "")).strip().upper(),
            observation=FullObservation.from_dict(obs_raw) if isinstance(obs_raw, dict) else obs_raw,
        )


@dataclass
class ActionEnvelope:
    """Action dispatched with unique identification and version tracking."""
    action_id: int
    agent_id: str
    observation_seq: int
    world_tick: int
    created_ns: int
    model_version: int
    action: HierarchicalAction

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "agent_id": self.agent_id,
            "observation_seq": self.observation_seq,
            "world_tick": self.world_tick,
            "created_ns": self.created_ns,
            "model_version": self.model_version,
            "action": self.action.to_dict(),
        }


@dataclass
class ActionResultEnvelope:
    """Result report returned from the Minecraft action executor."""
    action_id: int
    agent_id: str
    started_ns: int
    completed_ns: int
    world_tick_start: int
    world_tick_end: int
    status: str  # "SUCCESS", "TIMEOUT", "CANCELLED", "UNKNOWN", "REJECTED"
    elapsed_ms: float
    state_delta: Dict[str, Any] = field(default_factory=dict)
    failure_reason: str = "NONE"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ActionResultEnvelope:
        return cls(
            action_id=int(data.get("action_id", 0)),
            agent_id=str(data.get("agent_id", "")),
            started_ns=int(data.get("started_ns", 0)),
            completed_ns=int(data.get("completed_ns", 0)),
            world_tick_start=int(data.get("world_tick_start", 0)),
            world_tick_end=int(data.get("world_tick_end", 0)),
            status=str(data.get("status", "SUCCESS")).upper(),
            elapsed_ms=float(data.get("elapsed_ms", 0.0)),
            state_delta=data.get("state_delta", {}) or {},
            failure_reason=str(data.get("failure_reason", "NONE")),
        )


@dataclass
class TemporalTransition:
    """
    Guaranteed temporally aligned transition:
    (s_t, a_t, r_{t+1}, s_{t+1}, done, provenance)
    """
    agent_id: str
    episode_id: int
    transition_id: int

    # State provenance
    observation_seq: int
    next_observation_seq: int
    world_tick: int
    next_world_tick: int

    # Action provenance
    action_id: int
    model_version: int
    action_created_ns: int
    action_started_ns: int
    action_completed_ns: int

    # Timing metrics
    observation_timestamp_ns: int
    next_observation_timestamp_ns: int
    state_age_ns: int
    action_latency_ns: int

    # RL signals
    state: FullObservation
    action: HierarchicalAction
    reward: DecomposedReward
    next_state: FullObservation
    done: bool
    continuation: float
    result: ActionResult

    # Validation
    validation_status: ValidationStatus = ValidationStatus.VALID


class ExperienceValidator:
    """
    Hard invariant enforcement for reinforcement learning integrity:
    1. Duplicate world states (same world tick) do NOT create environment transitions.
    2. Stale actions or extreme latencies are flagged.
    3. Action timeouts with unknown outcomes are marked UNKNOWN_OUTCOME.
    4. Out-of-order or clock inversion events are rejected.
    """

    def __init__(
        self,
        max_state_age_ms: float = 250.0,
        max_action_latency_ms: float = 1000.0,
    ):
        self.max_state_age_ns = int(max_state_age_ms * 1_000_000)
        self.max_action_latency_ns = int(max_action_latency_ms * 1_000_000)

    def validate(
        self,
        prev_obs: ObservationEnvelope,
        curr_obs: ObservationEnvelope,
        action: ActionEnvelope,
        result: Optional[ActionResultEnvelope],
    ) -> Tuple[ValidationStatus, str]:
        # 1. Check for clock errors / inversions
        if curr_obs.timestamp_ns < prev_obs.timestamp_ns:
            return ValidationStatus.CLOCK_ERROR, "Observation timestamp inverted"
        if curr_obs.world_tick < prev_obs.world_tick:
            return ValidationStatus.CLOCK_ERROR, "World tick decreased"

        # 2. Check for duplicate environment state (100 Hz control tick on same 20 Hz world tick)
        if curr_obs.world_tick == prev_obs.world_tick:
            return ValidationStatus.DUPLICATE, "Same world tick (control-only update, not env transition)"

        # 3. Check for out of order sequences
        if curr_obs.sequence <= prev_obs.sequence:
            return ValidationStatus.OUT_OF_ORDER, f"Sequence non-increasing: {prev_obs.sequence} -> {curr_obs.sequence}"

        # 4. Check action timeouts and uncertain outcomes
        if result is not None:
            if result.status == "TIMEOUT":
                return ValidationStatus.ACTION_TIMEOUT, "Action timed out before execution completed"
            if result.status in ("UNKNOWN", "UNKNOWN_OUTCOME"):
                return ValidationStatus.UNKNOWN_OUTCOME, "Action outcome could not be verified by bridge"

        # 5. Check state staleness
        state_age_ns = action.created_ns - prev_obs.timestamp_ns
        if state_age_ns > self.max_state_age_ns:
            return ValidationStatus.STALE, f"State age {state_age_ns / 1_000_000:.1f}ms exceeds threshold {self.max_state_age_ns / 1_000_000:.1f}ms"

        # 6. Check action latency
        if result and result.completed_ns > 0:
            action_latency_ns = result.completed_ns - action.created_ns
            if action_latency_ns > self.max_action_latency_ns:
                return ValidationStatus.STALE, f"Action latency {action_latency_ns / 1_000_000:.1f}ms exceeds threshold"

        return ValidationStatus.VALID, "Transition temporally sound"
