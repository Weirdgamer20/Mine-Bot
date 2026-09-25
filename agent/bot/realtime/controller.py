from __future__ import annotations

import time
import queue
import logging
from typing import Optional, Dict, Any
import torch

from ..experience import (
    ObservationEnvelope,
    ActionEnvelope,
    ActionResultEnvelope,
    TemporalTransition,
    ExperienceValidator,
    ValidationStatus,
)
from ..schemas import FullObservation, HierarchicalAction, ContinuousMotorControl, ActionPrimitive
from .state_cache import StateCache
from .action_buffer import ActionBuffer
from .timing import MonotonicDeadlineScheduler
from .telemetry import RealtimeTelemetry

logger = logging.getLogger("RealtimeController")


class RealtimeController:
    """
    Dedicated 100 Hz Real-Time Controller for an individual bot peer.
    - CONTROL_PERIOD_NS = 10_000_000 (10ms).
    - Inference only, zero autograd, zero backprop, zero sync disk I/O.
    """
    CONTROL_PERIOD_NS = 10_000_000  # 10 ms

    def __init__(self, agent_id: str, policy, device: Optional[torch.device] = None):
        self.agent_id = agent_id
        self.policy = policy
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scheduler = MonotonicDeadlineScheduler(period_ns=self.CONTROL_PERIOD_NS)
        self.state_cache = StateCache(max_stale_ms=250.0)
        self.action_buffer = ActionBuffer(agent_id=agent_id)
        self.telemetry = RealtimeTelemetry(agent_id=agent_id)

        # Recurrent state isolated to this bot
        self.h = None
        self.prev_z = None
        self.prev_a = None
        self.current_skill_id = 0
        self.skill_duration_ticks = 0

    def run_tick(self) -> Optional[HierarchicalAction]:
        """Runs a single 10ms control cycle."""
        obs_env = self.state_cache.get_latest(self.agent_id)
        if obs_env is None or self.state_cache.is_stale(self.agent_id):
            self.scheduler.sleep_until_next_deadline()
            return None

        t0 = time.perf_counter_ns()
        with torch.inference_mode():
            action, self.h, self.prev_z, self.prev_a, self.current_skill_id, self.skill_duration_ticks, _, _ = (
                self.policy.rt_step(
                    obs=obs_env.observation,
                    h=self.h,
                    prev_z=self.prev_z,
                    prev_a=self.prev_a,
                    current_skill_id=self.current_skill_id,
                    skill_duration_ticks=self.skill_duration_ticks,
                )
            )

        infer_us = (time.perf_counter_ns() - t0) / 1000.0

        # Publish continuous motor control immediately
        self.action_buffer.publish_continuous(action.motor)

        # Enqueue discrete action if non-noop
        if action.command.primitive != ActionPrimitive.NOOP:
            act_id = self.action_buffer.generate_action_id()
            act_env = ActionEnvelope(
                action_id=act_id,
                agent_id=self.agent_id,
                observation_seq=obs_env.sequence,
                world_tick=obs_env.world_tick,
                created_ns=time.perf_counter_ns(),
                model_version=1,
                action=action,
            )
            self.action_buffer.enqueue_discrete(act_env)

        elapsed_ms = self.scheduler.sleep_until_next_deadline()
        self.telemetry.record(
            total_us=elapsed_ms * 1000.0,
            policy_us=infer_us,
            deadline_missed=elapsed_ms > 10.0,
        )
        return action
