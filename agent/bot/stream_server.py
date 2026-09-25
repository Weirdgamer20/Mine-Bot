from __future__ import annotations

import asyncio
import logging
import queue
import sys
import threading
import time
from typing import Optional, Dict, Any

from .config import Config
from .multi_agent import MultiAgentLearningSystem
from .schemas import FullObservation, EnvironmentManifest, ContinuousMotorControl, DiscreteActionCommand
from .protocol.framing import MessageType, encode_frame, StreamFramingReader
from .experience import ObservationEnvelope, ActionResultEnvelope
from .realtime import (
    BatchRealtimeController,
    ModelSnapshot,
    SnapshotRegistry,
    RealtimeTelemetry,
)
from .planning.planner_worker import PlannerWorker
from .learning import AsyncLearner

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("MultiAgentRealtimeServer")


class AgentStreamServer:
    """
    Decoupled 100 Hz Real-Time Stream Server for autonomous Minecraft peers.
    Architecture:
    - 100 Hz RT Control Thread: inference only, strict 10ms monotonic deadline.
    - 15 Hz Async Latent MPC Planner Thread: non-blocking imagined trajectory updates.
    - Async Learner Thread: background replay ingestion, RSSM/AC optimization, and checkpoints.
    - Network I/O: async event loop dispatching actions and reading state envelopes.
    """

    def __init__(
        self,
        system,
        host: str = "0.0.0.0",
        port: int = 9099,
        freeze_learning: bool = False,
        enable_planner: bool = True,
    ):
        if not hasattr(system, "AGENT_IDS"):
            from .multi_agent import MultiAgentLearningSystem
            if hasattr(system, "cfg"):
                wrapped = MultiAgentLearningSystem(system.cfg)
                wrapped.shared = system
                system = wrapped
            else:
                system = MultiAgentLearningSystem()

        self.system = system
        self.shared = system.shared
        self.host = host
        self.port = port
        self.server = None
        self.seq_counter = 1

        self.peer_writers: Dict[str, asyncio.StreamWriter] = {}
        self.loop = None

        # 1. Bounded Experience Queue
        self.experience_queue: queue.Queue = queue.Queue(maxsize=2048)

        # 2. Atomic Model Snapshot Registry
        initial_snapshot = ModelSnapshot(
            version=1,
            timestamp_ns=time.perf_counter_ns(),
            encoder=self.shared.encoder,
            rssm=self.shared.world_model,
            actor=self.shared.actor_critic,
            critic=self.shared.actor_critic,
            rnd=self.shared.rnd,
            device=self.shared.device,
        )
        self.snapshot_registry = SnapshotRegistry(initial_snapshot)

        # 3. Dedicated 100 Hz Batch Realtime Controller
        self.controller = BatchRealtimeController(
            shared_agent=self.shared,
            experience_queue=self.experience_queue,
            agent_ids=self.system.AGENT_IDS,
            control_period_ns=10_000_000,  # 10ms = 100 Hz
            peer_agents=self.system.peers,
        )
        self.controller.set_snapshot_registry(self.snapshot_registry)

        # 4. Asynchronous 15 Hz Latent MPC Planner Worker
        self.planner_worker = None
        if enable_planner:
            self.planner_worker = PlannerWorker(
                agent=self.shared,
                horizon=self.shared.cfg.imagination_horizon,
                num_candidates=8,
                update_interval_s=0.066,
            )
            self.planner_worker.start()
            self.controller.set_planner_worker(self.planner_worker)

        # 5. Asynchronous Background Learner Lane
        self.learner = AsyncLearner(
            agent=self.shared,
            experience_queue=self.experience_queue,
            snapshot_registry=self.snapshot_registry,
            telemetry=self.controller.global_telemetry,
            checkpoint_dir=self.shared.cfg.checkpoint_dir,
            checkpoint_interval_steps=500,
        )
        if freeze_learning:
            self.learner.pause_learning()
        self.learner.start()

        # 6. RT Control Loop Thread
        self.rt_thread = None
        self.running = False

    def _start_rt_thread(self):
        self.running = True
        self.rt_thread = threading.Thread(target=self._rt_loop_worker, daemon=True, name="100Hz_RT_Thread")
        self.rt_thread.start()

    def _rt_loop_worker(self):
        logger.info("[RealtimeServer] 100 Hz hard real-time control thread running.")
        last_log_time = time.time()

        while self.running:
            try:
                actions = self.controller.run_tick()

                # Dispatch actions to connected peers
                if actions and self.loop:
                    for aid, act in actions.items():
                        if act is not None and aid in self.peer_writers:
                            self.loop.call_soon_threadsafe(
                                self._dispatch_action_to_peer,
                                aid,
                                act,
                            )

                now = time.time()
                if now - last_log_time >= 5.0:
                    last_log_time = now
                    prof = self.controller.global_telemetry.snapshot()
                    l_metrics = self.learner.get_metrics()
                    logger.info(
                        "[REALTIME-100Hz] hz=%.2f | p50=%.2fms | p95=%.2fms | p99=%.2fms | max=%.2fms | misses=%d",
                        prof.hz, prof.p50_ms, prof.p95_ms, prof.p99_ms, prof.max_ms, prof.deadline_misses,
                    )
                    if l_metrics:
                        logger.info(
                            "  [LEARNER] steps=%d | wm_loss=%.4f | ac_loss=%.4f | return=%.3f",
                            l_metrics.get("training_steps", 0),
                            l_metrics.get("wm_loss", 0.0),
                            l_metrics.get("ac_loss", 0.0),
                            l_metrics.get("mean_imagined_return", 0.0),
                        )
            except Exception as e:
                logger.error("[RealtimeServer] RT Loop tick error: %s", e)

    def _dispatch_action_to_peer(self, agent_id: str, action):
        writer = self.peer_writers.get(agent_id)
        if not writer or writer.is_closing():
            return

        latest_obs = self.controller.state_cache.get_latest(agent_id)
        act_id = self.controller.action_buffers[agent_id].generate_action_id()

        payload = {
            "action_id": act_id,
            "agent_id": agent_id,
            "observation_seq": latest_obs.sequence if latest_obs else 0,
            "world_tick": latest_obs.world_tick if latest_obs else 0,
            "created_ns": time.perf_counter_ns(),
            "model_version": self.snapshot_registry.get_latest().version,
            "action": action.to_dict(),
        }

        try:
            frame = encode_frame(MessageType.ACTION, self.seq_counter, payload)
            self.seq_counter += 1
            writer.write(frame)
        except Exception as e:
            logger.warning("[RealtimeServer] Error writing action frame to %s: %s", agent_id, e)

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        peer = writer.get_extra_info("peername")
        framing_reader = StreamFramingReader(reader)
        agent_id = None
        logger.info("Peer connection from %s", peer)

        try:
            while True:
                msg = await framing_reader.read_frame()
                if msg is None:
                    logger.info("Peer %s disconnected (agent=%s).", peer, agent_id)
                    break

                msg_type, seq_id, payload = msg

                if msg_type == MessageType.HELLO:
                    agent_id = str(payload.get("agent_id", "LB-01")).strip().upper() or "LB-01"
                    if agent_id not in self.system.AGENT_IDS:
                        frame = encode_frame(
                            MessageType.ERROR,
                            self.seq_counter,
                            {"error": "INVALID_AGENT_ID", "expected": list(self.system.AGENT_IDS)},
                        )
                        self.seq_counter += 1
                        writer.write(frame)
                        await writer.drain()
                        break

                    manifest = EnvironmentManifest.from_dict(payload.get("manifest", {}))
                    self.shared.set_environment_manifest(manifest)
                    self.controller.runtime_states[agent_id].reset_episode()
                    self.peer_writers[agent_id] = writer

                    frame = encode_frame(
                        MessageType.WELCOME,
                        self.seq_counter,
                        {
                            "status": "ready",
                            "agent_id": agent_id,
                            "personality": self.system.contexts[agent_id].personality.name,
                            "agent_device": str(self.shared.device),
                            "peer_count": len(self.peer_writers),
                        },
                    )
                    self.seq_counter += 1
                    writer.write(frame)
                    await writer.drain()

                    logger.info(
                        "HELLO | agent=%s | personality=%s | MC=%s | protocol=%s",
                        agent_id,
                        self.system.contexts[agent_id].personality.name,
                        manifest.minecraft_version,
                        manifest.protocol_version,
                    )

                elif msg_type == MessageType.OBSERVATION:
                    if agent_id is None:
                        continue
                    if not payload.get("agent_id"):
                        payload["agent_id"] = agent_id
                    # Fast lock-free latest-value observation envelope registration
                    envelope = ObservationEnvelope.from_dict(payload)
                    self.controller.register_observation(envelope)

                elif msg_type == MessageType.ACTION_RESULT:
                    if agent_id is None:
                        continue
                    result_envelope = ActionResultEnvelope.from_dict(payload)
                    self.controller.register_action_result(result_envelope)

                elif msg_type == MessageType.DEATH:
                    if agent_id is None:
                        continue
                    self.controller.handle_death(agent_id)

                elif msg_type == MessageType.PING:
                    frame = encode_frame(
                        MessageType.PONG,
                        self.seq_counter,
                        {
                            "global_hz": self.controller.global_telemetry.snapshot().hz,
                            "learner_step": self.learner.training_steps,
                        },
                    )
                    self.seq_counter += 1
                    writer.write(frame)
                    await writer.drain()

        except Exception:
            logger.exception("Error handling stream client %s (agent=%s).", peer, agent_id)
        finally:
            if agent_id and agent_id in self.peer_writers:
                del self.peer_writers[agent_id]
            writer.close()
            await writer.wait_closed()

    async def start(self):
        self.loop = asyncio.get_running_loop()
        self._start_rt_thread()

        self.server = await asyncio.start_server(self.handle_client, self.host, self.port)
        logger.info(
            "Multi-agent Real-Time 100 Hz Server listening on %s:%s",
            self.host,
            self.port,
        )
        logger.info("Equal peers: %s", ", ".join(self.system.AGENT_IDS))
        async with self.server:
            try:
                await self.server.serve_forever()
            except (asyncio.CancelledError, KeyboardInterrupt):
                self.running = False


def run_server(config_path: Optional[str] = None):
    cfg = Config.from_json(config_path) if config_path else Config()
    system = MultiAgentLearningSystem(cfg)
    server = AgentStreamServer(system, host=cfg.stream_host, port=cfg.stream_port)
    asyncio.run(server.start())


if __name__ == "__main__":
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else None
    run_server(cfg_file)
