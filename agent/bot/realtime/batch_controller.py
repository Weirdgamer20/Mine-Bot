from __future__ import annotations

import time
import queue
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import torch

from ..config import Config
from ..schemas import (
    FullObservation,
    HierarchicalAction,
    ContinuousMotorControl,
    DiscreteActionCommand,
    ActionPrimitive,
    ActionResult,
    DecomposedReward,
)
from ..experience import (
    ObservationEnvelope,
    ActionEnvelope,
    ActionResultEnvelope,
    TemporalTransition,
    ExperienceValidator,
    ValidationStatus,
)
from ..personality import PERSONALITIES, PersonalityProfile
from .state_cache import StateCache
from .action_buffer import ActionBuffer
from .timing import MonotonicDeadlineScheduler
from .telemetry import RealtimeTelemetry
from .model_snapshot import SnapshotRegistry, ModelSnapshot

logger = logging.getLogger("RealtimeBatchController")


@dataclass
class AgentRuntimeState:
    """Isolated runtime memory and state for a single bot peer."""
    agent_id: str
    personality: Optional[PersonalityProfile] = None
    episode_id: int = 1
    episode_steps: int = 0
    h: Optional[torch.Tensor] = None
    prev_z: Optional[torch.Tensor] = None
    prev_a: Optional[torch.Tensor] = None
    current_skill_id: int = 0
    skill_duration_ticks: int = 0

    # Temporal provenance tracking
    last_observation: Optional[ObservationEnvelope] = None
    pending_action: Optional[ActionEnvelope] = None
    world_tick: int = -1
    last_world_tick: int = -1
    is_alive: bool = True

    # Two-clock temporal separation: Environment (~20 Hz) vs Motor Servo (100 Hz)
    last_inference_world_tick: int = -1
    last_action: Optional[HierarchicalAction] = None
    last_motor: Optional[ContinuousMotorControl] = None

    def reset_episode(self, next_episode_id: Optional[int] = None):
        self.h = None
        self.prev_z = None
        self.prev_a = None
        self.current_skill_id = 0
        self.skill_duration_ticks = 0
        self.episode_steps = 0
        self.last_observation = None
        self.pending_action = None
        self.world_tick = -1
        self.last_world_tick = -1
        self.last_inference_world_tick = -1
        self.last_action = None
        self.last_motor = None
        self.is_alive = True
        if next_episode_id is not None:
            self.episode_id = next_episode_id
        else:
            self.episode_id += 1


class BatchRealtimeController:
    """
    100 Hz Real-Time Batch Controller for 4 Minecraft peer bots.
    - Two-clock architecture:
      * 20 Hz Cognitive Loop (RSSM, Skills, Policy, Planner State submission) on new world ticks only.
      * 100 Hz Motor Servo (latest continuous motor re-dispatch & dt rate integration) on duplicate ticks.
    - Strict 10ms monotonic deadline.
    - Zero learning side effects, zero sync disk I/O, zero backprop.
    - Shares neural network parameters across bots, but keeps recurrent state independent.
    - Batches active bots into single GPU kernel launches for low latency.
    - Validates experience before routing to background replay queue.
    """

    def __init__(
        self,
        shared_agent,
        experience_queue: queue.Queue,
        agent_ids: Tuple[str, ...] = ("LB-01", "LB-02", "LB-03", "LB-04"),
        control_period_ns: int = 10_000_000,  # 10 ms = 100 Hz
        peer_agents: Optional[Dict[str, Any]] = None,
    ):
        self.shared_agent = shared_agent
        self.experience_queue = experience_queue
        self.agent_ids = agent_ids
        self.peer_agents = peer_agents or {}
        self.device = shared_agent.device

        self.scheduler = MonotonicDeadlineScheduler(period_ns=control_period_ns)
        self.state_cache = StateCache(max_stale_ms=250.0)
        self.action_buffers: Dict[str, ActionBuffer] = {
            aid: ActionBuffer(agent_id=aid) for aid in agent_ids
        }
        self.telemetry = {aid: RealtimeTelemetry(agent_id=aid) for aid in agent_ids}
        self.global_telemetry = RealtimeTelemetry(agent_id="GLOBAL_BATCH")

        self.validator = ExperienceValidator(max_state_age_ms=250.0, max_action_latency_ms=1000.0)

        # Isolated runtime state per bot with personality
        self.runtime_states: Dict[str, AgentRuntimeState] = {}
        for aid in agent_ids:
            peer = self.peer_agents.get(aid)
            p = getattr(peer, "personality", None) or PERSONALITIES.get(aid)
            self.runtime_states[aid] = AgentRuntimeState(agent_id=aid, personality=p)

        # Model snapshot registry for atomic updates
        self.snapshot_registry: Optional[SnapshotRegistry] = None
        self._action_counter = 1
        self._transition_counter = 1
        self.planner_worker = None

    def set_planner_worker(self, planner_worker):
        self.planner_worker = planner_worker

    def set_snapshot_registry(self, registry: SnapshotRegistry):
        self.snapshot_registry = registry

    def register_observation(self, envelope: ObservationEnvelope):
        """Called by network/bridge layer when an observation arrives."""
        agent_id = envelope.agent_id
        if agent_id in self.runtime_states:
            self.state_cache.put(envelope)

    def register_action_result(self, result: ActionResultEnvelope):
        """Processes execution result of a previous action."""
        agent_id = result.agent_id
        state = self.runtime_states.get(agent_id)
        if state is None or state.pending_action is None:
            return

        # If matching action_id
        if state.pending_action.action_id == result.action_id:
            curr_obs = self.state_cache.get_latest(agent_id)
            if state.last_observation is not None and curr_obs is not None:
                self._process_transition(
                    state=state,
                    prev_obs=state.last_observation,
                    curr_obs=curr_obs,
                    action_env=state.pending_action,
                    result_env=result,
                )
            state.pending_action = None

    def handle_death(self, agent_id: str, last_obs: Optional[ObservationEnvelope] = None):
        """Handles bot death event, resetting recurrent state and invalidating pending actions."""
        state = self.runtime_states.get(agent_id)
        if state:
            logger.info("Bot %s DIED at episode %d. Resetting episode state.", agent_id, state.episode_id)
            # Cancel active actions in buffer
            self.action_buffers[agent_id].cancel_active_action()
            state.reset_episode()

    def _process_transition(
        self,
        state: AgentRuntimeState,
        prev_obs: ObservationEnvelope,
        curr_obs: ObservationEnvelope,
        action_env: ActionEnvelope,
        result_env: Optional[ActionResultEnvelope],
    ):
        """Validates and creates temporal transition for the background learner."""
        val_status, reason = self.validator.validate(
            prev_obs=prev_obs,
            curr_obs=curr_obs,
            action=action_env,
            result=result_env,
        )

        if val_status == ValidationStatus.DUPLICATE:
            # Control frame only on same world tick. Do not submit duplicate environment transitions.
            return

        if val_status != ValidationStatus.VALID:
            # Drop invalid / corrupted / stale transitions from standard replay
            return

        # Compute decomposed reward
        curiosity = 0.0
        health_delta = result_env.state_delta.get("health_delta", 0.0) if result_env else 0.0
        reward = DecomposedReward(
            environmental_reward=float(health_delta),
            curiosity_reward=curiosity,
            total=float(health_delta),
        )

        t_id = self._transition_counter
        self._transition_counter += 1

        trans = TemporalTransition(
            agent_id=state.agent_id,
            episode_id=state.episode_id,
            transition_id=t_id,
            observation_seq=prev_obs.sequence,
            next_observation_seq=curr_obs.sequence,
            world_tick=prev_obs.world_tick,
            next_world_tick=curr_obs.world_tick,
            action_id=action_env.action_id,
            model_version=action_env.model_version,
            action_created_ns=action_env.created_ns,
            action_started_ns=result_env.started_ns if result_env else action_env.created_ns,
            action_completed_ns=result_env.completed_ns if result_env else time.perf_counter_ns(),
            observation_timestamp_ns=prev_obs.timestamp_ns,
            next_observation_timestamp_ns=curr_obs.timestamp_ns,
            state_age_ns=int(prev_obs.age_ms * 1_000_000),
            action_latency_ns=int(result_env.elapsed_ms * 1_000_000) if result_env else 0,
            state=prev_obs.observation,
            action=action_env.action,
            reward=reward,
            next_state=curr_obs.observation,
            done=curr_obs.observation.done,
            continuation=0.0 if curr_obs.observation.done else 1.0,
            result=ActionResult(
                action_primitive=action_env.action.command.primitive.value,
                success=result_env.status == "SUCCESS" if result_env else True,
                failure_reason=result_env.failure_reason if result_env else "NONE",
                elapsed_ticks=max(1, curr_obs.world_tick - prev_obs.world_tick),
                state_delta=result_env.state_delta if result_env else {},
            ),
            validation_status=val_status,
        )

        try:
            self.experience_queue.put_nowait(trans)
        except queue.Full:
            # Controlled backpressure drop when learning lags
            pass

    def run_tick(self) -> Dict[str, Optional[HierarchicalAction]]:
        """
        Single 10ms (100 Hz) control tick:
        1. Reads latest non-stale state for each peer.
        2. GATES cognitive inference: only runs full RSSM/skill/policy if world_tick is new.
        3. If duplicate tick: re-dispatches latest motor intent to 100 Hz servo buffer.
        4. Submits live states to PlannerWorker on new world ticks.
        5. Validates timing against 10ms monotonic deadline.
        """
        start_ns = time.perf_counter_ns()
        active_agents = []
        obs_list = []
        h_list = []
        prev_z_list = []
        prev_a_list = []
        skill_ids = []
        skill_durations = []
        planner_snaps = []

        model_version = 1
        if self.snapshot_registry:
            snap = self.snapshot_registry.get_latest()
            model_version = snap.version

        dispatched_actions: Dict[str, Optional[HierarchicalAction]] = {aid: None for aid in self.agent_ids}

        for aid in self.agent_ids:
            state = self.runtime_states[aid]
            latest_env = self.state_cache.get_latest(aid)

            if latest_env is None or self.state_cache.is_stale(aid):
                continue

            state.world_tick = latest_env.world_tick

            # Temporal gating check:
            if state.world_tick <= state.last_inference_world_tick:
                # 100 Hz SERVO TICK (Duplicate Minecraft state):
                # DO NOT advance RSSM.
                # DO NOT advance skill duration.
                # DO NOT sample a new policy action.
                # DO NOT modify prev_a, h, or z.
                # Just reuse and publish the latest continuous motor intent.
                if state.last_motor is not None:
                    self.action_buffers[aid].publish_continuous(state.last_motor)
                dispatched_actions[aid] = state.last_action

                if state.last_motor is not None and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "[SERVO] %s world_tick=%d new_state=NO yaw_rate=%.4f pitch_rate=%.4f",
                        aid, state.world_tick, state.last_motor.yaw_rate, state.last_motor.pitch_rate
                    )
                continue

            # NEW MINECRAFT WORLD TICK (~20 Hz Cognitive Loop):
            active_agents.append(aid)
            obs_list.append(latest_env.observation)

            # Ensure recurrent tensors exist
            if state.h is None:
                state.h = torch.zeros(1, self.shared_agent.cfg.recurrent_dim, device=self.device)
                state.prev_z = torch.zeros(1, self.shared_agent.cfg.latent_dim, device=self.device)
                state.prev_a = torch.zeros(1, self.shared_agent.cfg.motor_dim + self.shared_agent.cfg.num_primitives, device=self.device)

            h_list.append(state.h)
            prev_z_list.append(state.prev_z)
            prev_a_list.append(state.prev_a)
            skill_ids.append(state.current_skill_id)
            skill_durations.append(state.skill_duration_ticks)

            # Get async plan if available
            p_snap = None
            if self.planner_worker:
                p_snap = self.planner_worker.get_latest_plan(aid)
            planner_snaps.append(p_snap)

        if active_agents:
            h_batch = torch.cat(h_list, dim=0)
            prev_z_batch = torch.cat(prev_z_list, dim=0)
            prev_a_batch = torch.cat(prev_a_list, dim=0)
            personalities = [self.runtime_states[aid].personality for aid in active_agents]

            t0 = time.perf_counter_ns()
            # Fast-path batched GPU inference
            actions, next_h, next_z, next_a, next_skills, next_durations = self.shared_agent.rt_step_batch(
                obs_list=obs_list,
                h_batch=h_batch,
                prev_z_batch=prev_z_batch,
                prev_a_batch=prev_a_batch,
                skill_ids=skill_ids,
                skill_durations=skill_durations,
                planner_snapshots=planner_snaps,
                personalities=personalities,
            )
            infer_us = (time.perf_counter_ns() - t0) / 1000.0

            now_ns = time.perf_counter_ns()
            for i, aid in enumerate(active_agents):
                state = self.runtime_states[aid]
                action = actions[i]
                latest_env = self.state_cache.get_latest(aid)

                # Record skill execution consequence if completed
                peer = self.peer_agents.get(aid)
                skill_lib = getattr(peer, "skill_library", None) or getattr(self.shared_agent, "skill_library", None)
                if skill_durations[i] <= 0 and skill_lib is not None:
                    res = getattr(latest_env, "last_action_result", None)
                    prev_success = res.success if res else True
                    c_delta = float(res.state_delta.get("health_delta", 0.0)) if res and hasattr(res, "state_delta") else 0.0
                    skill_lib.record_skill_execution(
                        skill_id=skill_ids[i],
                        duration_ticks=6,
                        consequence_delta=c_delta,
                        success=prev_success,
                    )

                # Advance recurrent state exactly once per environment tick
                state.h = next_h[i : i + 1]
                state.prev_z = next_z[i : i + 1]
                state.prev_a = next_a[i : i + 1]
                state.current_skill_id = next_skills[i]
                state.skill_duration_ticks = next_durations[i]
                state.episode_steps += 1

                # Submit live state to PlannerWorker for async MPC imagination
                if self.planner_worker:
                    import torch.nn.functional as F
                    skill_vec = F.one_hot(torch.tensor([next_skills[i]], device=self.device), num_classes=8).float()
                    val_mask_t = torch.tensor(
                        latest_env.observation.validity_mask.valid_primitives_mask[:self.shared_agent.cfg.num_primitives],
                        device=self.device
                    ).bool().unsqueeze(0)
                    self.planner_worker.submit_state(
                        agent_id=aid,
                        h=state.h,
                        z=state.prev_z,
                        skill_one_hot=skill_vec,
                        validity_mask=val_mask_t,
                    )

                # 1. Continuous controls -> latest-value buffer
                self.action_buffers[aid].publish_continuous(action.motor)

                # 2. Discrete action -> if not NOOP, wrap in ActionEnvelope and enqueue
                action_id = self._action_counter
                self._action_counter += 1

                act_env = ActionEnvelope(
                    action_id=action_id,
                    agent_id=aid,
                    observation_seq=latest_env.sequence,
                    world_tick=latest_env.world_tick,
                    created_ns=now_ns,
                    model_version=model_version,
                    action=action,
                )

                if action.command.primitive != ActionPrimitive.NOOP:
                    self.action_buffers[aid].enqueue_discrete(act_env)
                    state.pending_action = act_env

                # Save provenance & update temporal gating for next transition
                state.last_observation = latest_env
                state.last_world_tick = state.world_tick
                state.last_inference_world_tick = state.world_tick
                state.last_action = action
                state.last_motor = action.motor
                dispatched_actions[aid] = action

                # Telemetry logging for [CONTROL]
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "[CONTROL] %s world_tick=%d new_state=YES yaw_rate=%.4f pitch_rate=%.4f prim=%s skill=%d(rem=%d)",
                        aid, latest_env.world_tick, action.motor.yaw_rate, action.motor.pitch_rate,
                        action.command.primitive.value, state.current_skill_id, state.skill_duration_ticks
                    )

                self.telemetry[aid].record(
                    total_us=infer_us,
                    policy_us=infer_us,
                    deadline_missed=False,
                )

        compute_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        deadline_missed = compute_ms > 10.0

        self.global_telemetry.record(
            total_us=compute_ms * 1000.0,
            deadline_missed=deadline_missed,
        )

        # Enforce 100 Hz deadline (10ms)
        self.scheduler.sleep_until_next_deadline()

        return dispatched_actions
