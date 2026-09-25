import time
import queue
import torch
import numpy as np

from bot.config import Config
from bot.agent import LearningAgent
from bot.schemas import (
    FullObservation,
    ContinuousMotorControl,
    DiscreteActionCommand,
    HierarchicalAction,
    ActionPrimitive,
    ActionResult,
    DecomposedReward,
    ActionValidityMask,
    MechanicalAffordanceState,
    CompleteInventoryState,
)
from bot.experience import (
    ObservationEnvelope,
    ActionEnvelope,
    ActionResultEnvelope,
    TemporalTransition,
    ExperienceValidator,
    ValidationStatus,
)
from bot.realtime import (
    MonotonicDeadlineScheduler,
    RealtimeTelemetry,
    StateCache,
    ActionBuffer,
    ModelSnapshot,
    SnapshotRegistry,
    BatchRealtimeController,
    AgentRuntimeState,
)
from bot.learning import LatencyGovernor, CheckpointWorker, ReplayWorker, AsyncLearner


def make_dummy_observation(world_tick: int, seq: int, agent_id: str = "LB-01", done: bool = False) -> ObservationEnvelope:
    obs = FullObservation(
        voxels=[0] * 1331,
        voxel_shape=[11, 11, 11],
        player_state=[20.0, 20.0, 5.0, 20.0, 0.0, 64.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        inventory=CompleteInventoryState(),
        entities=[],
        affordances=MechanicalAffordanceState(),
        validity_mask=ActionValidityMask(),
        done=done,
        step_id=seq,
    )
    return ObservationEnvelope(
        sequence=seq,
        world_tick=world_tick,
        timestamp_ns=time.perf_counter_ns(),
        received_ns=time.perf_counter_ns(),
        agent_id=agent_id,
        observation=obs,
    )


def test_experience_validator_duplicate_rejection():
    """Rule 8: Duplicate world states (same world tick) must NOT create environment transitions."""
    validator = ExperienceValidator()
    obs1 = make_dummy_observation(world_tick=100, seq=1)
    obs2 = make_dummy_observation(world_tick=100, seq=2)  # Same world tick!

    action = ActionEnvelope(
        action_id=1,
        agent_id="LB-01",
        observation_seq=1,
        world_tick=100,
        created_ns=time.perf_counter_ns(),
        model_version=1,
        action=HierarchicalAction(),
    )
    res = ActionResultEnvelope(
        action_id=1,
        agent_id="LB-01",
        started_ns=time.perf_counter_ns(),
        completed_ns=time.perf_counter_ns(),
        world_tick_start=100,
        world_tick_end=100,
        status="SUCCESS",
        elapsed_ms=5.0,
    )

    status, reason = validator.validate(obs1, obs2, action, res)
    assert status == ValidationStatus.DUPLICATE
    assert "Same world tick" in reason


def test_experience_validator_timeout_and_valid():
    """Rule 9 & 11: Timeouts marked UNKNOWN/TIMEOUT; valid transitions accepted."""
    validator = ExperienceValidator()
    obs1 = make_dummy_observation(world_tick=100, seq=1)
    obs2 = make_dummy_observation(world_tick=101, seq=2)

    action = ActionEnvelope(
        action_id=1,
        agent_id="LB-01",
        observation_seq=1,
        world_tick=100,
        created_ns=obs1.timestamp_ns,
        model_version=1,
        action=HierarchicalAction(),
    )

    # 1. Timeout test
    res_timeout = ActionResultEnvelope(
        action_id=1,
        agent_id="LB-01",
        started_ns=obs1.timestamp_ns,
        completed_ns=obs1.timestamp_ns + 10_000_000,
        world_tick_start=100,
        world_tick_end=101,
        status="TIMEOUT",
        elapsed_ms=2000.0,
    )
    status, _ = validator.validate(obs1, obs2, action, res_timeout)
    assert status == ValidationStatus.ACTION_TIMEOUT

    # 2. Valid test
    res_success = ActionResultEnvelope(
        action_id=1,
        agent_id="LB-01",
        started_ns=obs1.timestamp_ns,
        completed_ns=obs1.timestamp_ns + 10_000_000,
        world_tick_start=100,
        world_tick_end=101,
        status="SUCCESS",
        elapsed_ms=10.0,
    )
    status, _ = validator.validate(obs1, obs2, action, res_success)
    assert status == ValidationStatus.VALID


def test_monotonic_deadline_scheduler():
    """Rule 1: 100 Hz scheduling without drift."""
    scheduler = MonotonicDeadlineScheduler(period_ns=2_000_000)  # 2ms for fast testing
    t0 = time.perf_counter()
    for _ in range(5):
        scheduler.sleep_until_next_deadline()
    t1 = time.perf_counter()
    elapsed = t1 - t0
    # Should take at least 10ms (5 * 2ms)
    assert elapsed >= 0.009


_shared_test_agent = None

def get_shared_test_agent():
    global _shared_test_agent
    if _shared_test_agent is None:
        _shared_test_agent = LearningAgent(Config())
    return _shared_test_agent


def test_model_snapshot_atomic_swap():
    """Rule 11 & 15: Atomic model snapshot updates."""
    agent = get_shared_test_agent()
    snap1 = ModelSnapshot(
        version=1,
        timestamp_ns=time.perf_counter_ns(),
        encoder=agent.encoder,
        rssm=agent.world_model,
        actor=agent.actor_critic,
        critic=agent.actor_critic,
        rnd=agent.rnd,
        device=agent.device,
    )
    reg = SnapshotRegistry(snap1)
    assert reg.get_latest().version == 1

    v2 = reg.publish_new_version(
        encoder=agent.encoder,
        rssm=agent.world_model,
        actor=agent.actor_critic,
        critic=agent.actor_critic,
        rnd=agent.rnd,
        device=agent.device,
    )
    assert v2 == 2
    assert reg.get_latest().version == 2


def test_batch_realtime_controller_multi_agent():
    """Rule 4, 16, 17, 18: Four bots with independent recurrent states and batched GPU inference."""
    agent = get_shared_test_agent()
    exp_queue = queue.Queue(maxsize=100)

    controller = BatchRealtimeController(
        shared_agent=agent,
        experience_queue=exp_queue,
        agent_ids=("LB-01", "LB-02", "LB-03", "LB-04"),
        control_period_ns=10_000_000,
    )

    # Feed observations for 4 bots
    for aid in ("LB-01", "LB-02", "LB-03", "LB-04"):
        obs = make_dummy_observation(world_tick=10, seq=1, agent_id=aid)
        controller.register_observation(obs)

    # Warmup tick (Rule 46: never let the first frame trigger compilation/allocation)
    controller.run_tick()
    controller.global_telemetry.idx = 0
    controller.global_telemetry.count = 0

    # Measured ticks
    for tick in range(2, 8):
        for aid in ("LB-01", "LB-02", "LB-03", "LB-04"):
            obs = make_dummy_observation(world_tick=10 + tick, seq=tick, agent_id=aid)
            controller.register_observation(obs)
        actions = controller.run_tick()

    for aid in ("LB-01", "LB-02", "LB-03", "LB-04"):
        assert actions[aid] is not None
        assert isinstance(actions[aid].motor, ContinuousMotorControl)
        assert controller.runtime_states[aid].h is not None

    # Check telemetry (excluding cold start; ~13.8ms on WSL RTX 3050)
    prof = controller.global_telemetry.snapshot()
    assert prof.sample_count >= 5
    assert prof.p50_ms < 40.0, f"Expected P50 < 40.0ms on WSL RTX 3050, got {prof.p50_ms}ms"


def test_death_resets_episode_state():
    """Rule 10, 28, 29: Death flushes recurrent state and pending actions."""
    agent = get_shared_test_agent()
    exp_queue = queue.Queue(maxsize=100)

    controller = BatchRealtimeController(
        shared_agent=agent,
        experience_queue=exp_queue,
        agent_ids=("LB-01", "LB-02"),
        control_period_ns=10_000_000,
    )

    obs = make_dummy_observation(world_tick=50, seq=1, agent_id="LB-01")
    controller.register_observation(obs)
    controller.run_tick()

    state = controller.runtime_states["LB-01"]
    assert state.h is not None
    ep1 = state.episode_id

    # Bot dies
    controller.handle_death("LB-01")
    assert state.h is None
    assert state.pending_action is None
    assert state.episode_id == ep1 + 1


def test_latency_governor():
    """Rule 53: Latency feedback controller throttles learner."""
    governor = LatencyGovernor(target_p95_ms=8.0, critical_p95_ms=10.0)
    telem = RealtimeTelemetry(agent_id="TEST")

    # Fast loop: 3ms
    for _ in range(30):
        telem.record(total_us=3000.0)
    b, s = governor.evaluate_training_budget(telem)
    assert b == 16
    assert s == 0.01

    # Overloaded loop: 12ms
    for _ in range(30):
        telem.record(total_us=12000.0)
    b, s = governor.evaluate_training_budget(telem)
    assert b == 4
    assert s == 0.20


def test_temporal_gating_and_servo_redispatch():
    """Verify that RSSM and skills only advance on NEW world_ticks, while 100Hz servo re-dispatches."""
    agent = get_shared_test_agent()
    exp_queue = queue.Queue(maxsize=100)

    controller = BatchRealtimeController(
        shared_agent=agent,
        experience_queue=exp_queue,
        agent_ids=("LB-01",),
        control_period_ns=10_000_000,
    )

    # 1. First observation on world_tick=100 -> Cognitive Step (20 Hz)
    obs1 = make_dummy_observation(world_tick=100, seq=1, agent_id="LB-01")
    controller.register_observation(obs1)
    act1 = controller.run_tick()["LB-01"]

    state = controller.runtime_states["LB-01"]
    expected_duration = agent.cfg.skill_duration_ticks - 1
    assert state.last_inference_world_tick == 100
    assert state.episode_steps == 1
    assert state.skill_duration_ticks == expected_duration  # decremented once for tick 100
    h_tick100 = state.h.clone()

    # 2. Duplicate observation on same world_tick=100 (10ms later) -> Servo Step (100 Hz)
    obs_dup = make_dummy_observation(world_tick=100, seq=2, agent_id="LB-01")
    controller.register_observation(obs_dup)
    act2 = controller.run_tick()["LB-01"]

    # Assert RSSM, skills, and episode steps were NOT advanced!
    assert state.last_inference_world_tick == 100
    assert state.episode_steps == 1, "Duplicate world_tick must NOT increment cognitive episode_steps!"
    assert state.skill_duration_ticks == expected_duration, "Duplicate world_tick must NOT decrement skill duration!"
    assert torch.allclose(state.h, h_tick100), "Duplicate world_tick must NOT advance recurrent state h!"
    # Assert latest motor action was re-dispatched
    assert act2 is not None
    assert act2.motor.yaw_rate == act1.motor.yaw_rate

    # 3. New observation on world_tick=101 -> Next Cognitive Step (20 Hz)
    obs2 = make_dummy_observation(world_tick=101, seq=3, agent_id="LB-01")
    controller.register_observation(obs2)
    act3 = controller.run_tick()["LB-01"]

    assert state.last_inference_world_tick == 101
    assert state.episode_steps == 2, "New world_tick must advance cognitive episode_steps to 2"
    assert state.skill_duration_ticks == expected_duration - 1, "New world_tick must decrement skill duration"


if __name__ == "__main__":
    print("Running Real-Time Architecture Test Suite...")
    test_experience_validator_duplicate_rejection()
    print("  [PASS] test_experience_validator_duplicate_rejection")
    test_experience_validator_timeout_and_valid()
    print("  [PASS] test_experience_validator_timeout_and_valid")
    test_monotonic_deadline_scheduler()
    print("  [PASS] test_monotonic_deadline_scheduler")
    test_latency_governor()
    print("  [PASS] test_latency_governor")
    test_model_snapshot_atomic_swap()
    print("  [PASS] test_model_snapshot_atomic_swap")
    test_batch_realtime_controller_multi_agent()
    print("  [PASS] test_batch_realtime_controller_multi_agent")
    test_death_resets_episode_state()
    print("  [PASS] test_death_resets_episode_state")
    test_temporal_gating_and_servo_redispatch()
    print("  [PASS] test_temporal_gating_and_servo_redispatch")
    print("\nALL 8 REAL-TIME ARCHITECTURE TESTS PASSED SUCCESSFULLY!")
