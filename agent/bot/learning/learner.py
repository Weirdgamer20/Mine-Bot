from __future__ import annotations

import queue
import threading
import time
import logging
from typing import Dict, Any, Optional, Tuple
import torch

from .losses import train_world_model_step, train_actor_critic_imagination
from .checkpoint_worker import CheckpointWorker
from .replay_worker import ReplayWorker
from ..realtime.model_snapshot import SnapshotRegistry
from ..realtime.telemetry import RealtimeTelemetry

logger = logging.getLogger("AsyncLearner")


class LatencyGovernor:
    """
    Dynamic resource throttle protecting real-time control latency.
    Adjusts training frequency and batch size based on monitored P95/P99 latency.
    """

    def __init__(self, target_p95_ms: float = 8.0, critical_p95_ms: float = 10.0):
        self.target_p95_ms = target_p95_ms
        self.critical_p95_ms = critical_p95_ms

    def evaluate_training_budget(self, telemetry: Optional[RealtimeTelemetry]) -> Tuple[int, float]:
        """
        Returns:
            (batch_size, sleep_seconds)
        """
        if telemetry is None:
            return 16, 0.02

        profile = telemetry.snapshot()
        if profile.sample_count < 20:
            return 16, 0.02

        p95 = profile.p95_ms

        if p95 < 5.0:
            # Latency is excellent (<5ms), allow full training throughput
            return 16, 0.01
        elif p95 < self.target_p95_ms:
            # Latency is good (5-8ms), standard throughput
            return 16, 0.02
        elif p95 < self.critical_p95_ms:
            # Moderate latency pressure (8-10ms), scale down batch size and throttle
            return 8, 0.05
        else:
            # Critical latency pressure (>10ms), enter protection mode: pause training temporarily
            logger.warning("[LatencyGovernor] RT P95=%.2fms exceeded %.1fms! Throttling learner.", p95, self.critical_p95_ms)
            return 4, 0.20


class AsyncLearner(threading.Thread):
    """
    Decoupled Asynchronous Learner System.
    - Consumes from bounded experience queue via ReplayWorker.
    - Trains RSSM World Model, RND curiosity, DIAYN skills, and Actor-Critic in background.
    - Governed by RT latency telemetry to strictly prevent GPU starvation of the 100 Hz loop.
    - Publishes atomic ModelSnapshots to SnapshotRegistry.
    - Asynchronously offloads disk checkpoints via CheckpointWorker.
    """

    def __init__(
        self,
        agent,
        experience_queue: queue.Queue,
        snapshot_registry: Optional[SnapshotRegistry] = None,
        telemetry: Optional[RealtimeTelemetry] = None,
        checkpoint_dir: str = "checkpoints",
        checkpoint_interval_steps: int = 500,
    ):
        super().__init__(daemon=True, name="AsyncLearner")
        self.agent = agent
        self.experience_queue = experience_queue
        self.snapshot_registry = snapshot_registry
        self.telemetry = telemetry
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval_steps = checkpoint_interval_steps

        self.governor = LatencyGovernor(target_p95_ms=8.0, critical_p95_ms=10.0)

        # Worker subsystems
        self.replay_worker = ReplayWorker(
            experience_queue=self.experience_queue,
            memory=self.agent.memory,
            spatial_memory=getattr(self.agent, "spatial_memory", None),
            experience_graph=getattr(self.agent, "experience_graph", None),
        )
        self.checkpoint_worker = CheckpointWorker()

        self.running = False
        self.training_steps = 0
        self.latest_metrics: Dict[str, float] = {}
        self.lock = threading.Lock()

    def run(self):
        self.running = True
        self.replay_worker.start()
        self.checkpoint_worker.start()
        logger.info("[AsyncLearner] Started asynchronous learner thread with LatencyGovernor.")

        while self.running:
            # Need minimum number of steps in memory
            min_required = 32
            if len(self.agent.memory) < min_required:
                time.sleep(0.5)
                continue

            # Latency feedback control
            batch_size, sleep_sec = self.governor.evaluate_training_budget(self.telemetry)

            try:
                batch = self.agent.memory.sample_sequences(
                    batch_size=batch_size,
                    seq_len=16,
                    device=self.agent.device,
                )
                if batch is None:
                    time.sleep(0.1)
                    continue

                # 1. World Model & RND Training Step
                wm_loss, wm_metrics, last_h, last_z = train_world_model_step(
                    self.agent.encoder,
                    self.agent.world_model,
                    batch,
                    kl_weight=self.agent.cfg.kl_weight,
                    continuation_weight=self.agent.cfg.continuation_weight,
                    rnd=self.agent.rnd,
                )

                self.agent.wm_opt.zero_grad()
                wm_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.wm_params, max_norm=10.0)
                self.agent.wm_opt.step()

                # 2. Actor-Critic Latent Imagination Step
                ac_loss, ac_metrics = train_actor_critic_imagination(
                    self.agent.actor_critic,
                    self.agent.world_model,
                    self.agent.skill_net,
                    start_h=last_h,
                    start_z=last_z,
                    horizon=self.agent.cfg.imagination_horizon,
                    gamma=self.agent.cfg.gamma,
                    lambda_gae=self.agent.cfg.lambda_gae,
                )

                self.agent.ac_opt.zero_grad()
                ac_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.actor_critic.parameters(), max_norm=10.0)
                self.agent.ac_opt.step()

                self.training_steps += 1

                with self.lock:
                    self.latest_metrics = {**wm_metrics, **ac_metrics, "training_steps": self.training_steps}

                # 3. Publish Model Snapshot atomically every 50 optimizer steps
                if self.snapshot_registry and self.training_steps % 50 == 0:
                    self.snapshot_registry.publish_new_version(
                        encoder=self.agent.encoder,
                        rssm=self.agent.world_model,
                        actor=self.agent.actor_critic,
                        critic=self.agent.actor_critic,
                        rnd=self.agent.rnd,
                        device=self.agent.device,
                    )

                # 4. Asynchronous Checkpointing
                if self.training_steps % self.checkpoint_interval_steps == 0:
                    state_dicts = {
                        "encoder": self.agent.encoder.state_dict(),
                        "world_model": self.agent.world_model.state_dict(),
                        "actor_critic": self.agent.actor_critic.state_dict(),
                        "rnd": self.agent.rnd.state_dict(),
                        "skill_net": self.agent.skill_net.state_dict(),
                        "wm_opt": self.agent.wm_opt.state_dict(),
                        "ac_opt": self.agent.ac_opt.state_dict(),
                    }
                    self.checkpoint_worker.submit_checkpoint(
                        state_dicts=state_dicts,
                        step=self.training_steps,
                        episode=getattr(self.agent, "episode_count", 1),
                        checkpoint_dir=self.checkpoint_dir,
                    )

            except (KeyboardInterrupt, SystemExit):
                self.running = False
                break
            except Exception as e:
                if self.running:
                    logger.warning("[AsyncLearner] Training step error: %s", e)

            time.sleep(sleep_sec)

    def get_metrics(self) -> Dict[str, float]:
        with self.lock:
            return dict(self.latest_metrics)

    def stop(self):
        self.running = False
        self.replay_worker.stop()
        self.checkpoint_worker.stop()
