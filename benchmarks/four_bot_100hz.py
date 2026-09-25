import time
import queue
import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = str(Path(__file__).parent.parent / "agent")
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from bot.config import Config
from bot.agent import LearningAgent
from bot.schemas import (
    FullObservation,
    ContinuousMotorControl,
    ActionValidityMask,
    MechanicalAffordanceState,
    CompleteInventoryState,
)
from bot.experience import ObservationEnvelope
from bot.realtime import BatchRealtimeController


def make_benchmark_obs(seq: int, tick: int, agent_id: str) -> ObservationEnvelope:
    obs = FullObservation(
        voxels=[0] * 1331,
        voxel_shape=[11, 11, 11],
        player_state=[20.0, 20.0, 5.0, 20.0, 0.0, 64.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        inventory=CompleteInventoryState(),
        entities=[],
        affordances=MechanicalAffordanceState(),
        validity_mask=ActionValidityMask(),
        done=False,
        step_id=seq,
    )
    return ObservationEnvelope(
        sequence=seq,
        world_tick=tick,
        timestamp_ns=time.perf_counter_ns(),
        received_ns=time.perf_counter_ns(),
        agent_id=agent_id,
        observation=obs,
    )


def run_benchmark(num_ticks: int = 100):
    print("=" * 65)
    print("MINE-BOT 100 Hz FOUR-BOT REAL-TIME QUALIFICATION BENCHMARK")
    print("=" * 65)

    cfg = Config()
    print("Allocating GPU tensors and warming up agent...")
    agent = LearningAgent(cfg)
    exp_queue = queue.Queue(maxsize=1024)

    controller = BatchRealtimeController(
        shared_agent=agent,
        experience_queue=exp_queue,
        agent_ids=("LB-01", "LB-02", "LB-03", "LB-04"),
        control_period_ns=10_000_000,  # 10ms = 100 Hz
    )

    # Warmup tick
    for aid in ("LB-01", "LB-02", "LB-03", "LB-04"):
        controller.register_observation(make_benchmark_obs(seq=0, tick=0, agent_id=aid))
    controller.run_tick()

    print(f"\nRunning {num_ticks} iterations @ 100 Hz for LB-01, LB-02, LB-03, LB-04 simultaneously...")
    controller.global_telemetry.idx = 0
    controller.global_telemetry.count = 0
    controller.global_telemetry.deadline_misses = 0

    t_start = time.perf_counter()
    for tick in range(1, num_ticks + 1):
        for aid in ("LB-01", "LB-02", "LB-03", "LB-04"):
            controller.register_observation(make_benchmark_obs(seq=tick, tick=tick // 5, agent_id=aid))
        controller.run_tick()

    total_wall_s = time.perf_counter() - t_start
    prof = controller.global_telemetry.snapshot()

    print("\n" + "=" * 65)
    print("QUALIFICATION RESULTS:")
    print("=" * 65)
    print(f"Total Ticks:       {num_ticks}")
    print(f"Total Wall Time:   {total_wall_s:.3f} s")
    print(f"Measured Frequency: {prof.hz:.2f} Hz (Target: 100 Hz)")
    print(f"P50 Latency:       {prof.p50_ms:.2f} ms (Target: <5 ms)")
    print(f"P90 Latency:       {prof.p90_ms:.2f} ms")
    print(f"P95 Latency:       {prof.p95_ms:.2f} ms (Target: <10 ms)")
    print(f"P99 Latency:       {prof.p99_ms:.2f} ms")
    print(f"Max Latency:       {prof.max_ms:.2f} ms")
    print(f"Deadline Misses:   {prof.deadline_misses} / {num_ticks}")
    print("=" * 65)

    if prof.p95_ms <= 10.0:
        print("[SUCCESS] 100 Hz Real-Time qualification target achieved!")
    else:
        print("[WARNING] P95 latency exceeded 10ms threshold.")


if __name__ == "__main__":
    ticks = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    run_benchmark(ticks)
