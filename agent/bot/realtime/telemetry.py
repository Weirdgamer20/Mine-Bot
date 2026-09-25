import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class LatencyProfile:
    agent_id: str
    sample_count: int
    hz: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    deadline_misses: int


class RealtimeTelemetry:
    """
    Zero-allocation circular buffer latency recorder for 100 Hz RT loops.
    Tracks sub-step component timings and distribution percentiles.
    """

    def __init__(self, agent_id: str = "GLOBAL", buffer_size: int = 1000):
        self.agent_id = agent_id
        self.buffer_size = buffer_size
        self.total_latencies_us = np.zeros(buffer_size, dtype=np.float32)
        self.encoder_latencies_us = np.zeros(buffer_size, dtype=np.float32)
        self.rssm_latencies_us = np.zeros(buffer_size, dtype=np.float32)
        self.policy_latencies_us = np.zeros(buffer_size, dtype=np.float32)
        self.idx = 0
        self.count = 0
        self.deadline_misses = 0

    def record(
        self,
        total_us: float,
        encoder_us: float = 0.0,
        rssm_us: float = 0.0,
        policy_us: float = 0.0,
        deadline_missed: bool = False,
    ):
        self.total_latencies_us[self.idx] = total_us
        self.encoder_latencies_us[self.idx] = encoder_us
        self.rssm_latencies_us[self.idx] = rssm_us
        self.policy_latencies_us[self.idx] = policy_us

        self.idx = (self.idx + 1) % self.buffer_size
        self.count += 1
        if deadline_missed:
            self.deadline_misses += 1

    def snapshot(self) -> LatencyProfile:
        valid_count = min(self.count, self.buffer_size)
        if valid_count == 0:
            return LatencyProfile(
                agent_id=self.agent_id,
                sample_count=0,
                hz=0.0,
                p50_ms=0.0,
                p90_ms=0.0,
                p95_ms=0.0,
                p99_ms=0.0,
                max_ms=0.0,
                deadline_misses=self.deadline_misses,
            )

        data_ms = self.total_latencies_us[:valid_count] / 1000.0
        p50 = float(np.percentile(data_ms, 50))
        p90 = float(np.percentile(data_ms, 90))
        p95 = float(np.percentile(data_ms, 95))
        p99 = float(np.percentile(data_ms, 99))
        max_v = float(np.max(data_ms))

        # Effective loop frequency
        mean_ms = float(np.mean(data_ms))
        hz = 1000.0 / mean_ms if mean_ms > 0 else 0.0

        return LatencyProfile(
            agent_id=self.agent_id,
            sample_count=self.count,
            hz=round(hz, 2),
            p50_ms=round(p50, 3),
            p90_ms=round(p90, 3),
            p95_ms=round(p95, 3),
            p99_ms=round(p99, 3),
            max_ms=round(max_v, 3),
            deadline_misses=self.deadline_misses,
        )

    def summary_string(self) -> str:
        s = self.snapshot()
        return (
            f"[RT-TELEMETRY | {s.agent_id}] samples={s.sample_count} | "
            f"p50={s.p50_ms:.2f}ms | p95={s.p95_ms:.2f}ms | p99={s.p99_ms:.2f}ms | "
            f"max={s.max_ms:.2f}ms | misses={s.deadline_misses}"
        )
