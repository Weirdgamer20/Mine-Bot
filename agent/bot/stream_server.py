import asyncio
import logging
import sys
from typing import Optional

from .multi_agent import MultiAgentLearningSystem
from .schemas import FullObservation, EnvironmentManifest
from .protocol.framing import MessageType, encode_frame, StreamFramingReader
from .config import Config

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("MultiAgentStreamServer")


class AgentStreamServer:
    """Persistent TCP server for four equal autonomous Minecraft peers.

    Each connection identifies itself as LB-01..LB-04 in HELLO. The peers share
    one neural learner/world model/replay system, while recurrent and episode
    state is isolated per peer.
    """

    def __init__(self, system, host: str = "0.0.0.0", port: int = 9099):
        if not hasattr(system, "AGENT_IDS"):
            from .multi_agent import MultiAgentLearningSystem
            if hasattr(system, "cfg"):
                wrapped = MultiAgentLearningSystem(system.cfg)
                wrapped.shared = system
                system = wrapped
            else:
                system = MultiAgentLearningSystem()
        self.system = system
        self.host = host
        self.port = port
        self.server = None
        self.seq_counter = 1

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
                    agent_id = str(payload.get("agent_id", "")).strip().upper()
                    if agent_id not in self.system.AGENT_IDS:
                        frame = encode_frame(
                            MessageType.ERROR,
                            self.seq_counter,
                            {"error": "INVALID_AGENT_ID", "expected": list(self.system.AGENT_IDS)},
                        )
                        self.seq_counter += 1
                        writer.write(frame)
                        await writer.drain()
                        logger.error("Rejected peer %s with invalid agent_id=%r.", peer, agent_id)
                        break

                    manifest = EnvironmentManifest.from_dict(payload.get("manifest", {}))
                    self.system.shared.set_environment_manifest(manifest)
                    self.system.reset_agent(agent_id)

                    frame = encode_frame(
                        MessageType.WELCOME,
                        self.seq_counter,
                        {
                            "status": "ready",
                            "agent_id": agent_id,
                            "personality": self.system.contexts[agent_id].personality.name,
                            "agent_device": str(self.system.device),
                            "peer_count": len(self.system.contexts),
                        },
                    )
                    self.seq_counter += 1
                    writer.write(frame)
                    await writer.drain()

                    logger.info(
                        "HELLO | agent=%s | personality=%s | MC=%s | protocol=%s | Mineflayer=%s",
                        agent_id,
                        self.system.contexts[agent_id].personality.name,
                        manifest.minecraft_version,
                        manifest.protocol_version,
                        manifest.mineflayer_version,
                    )

                elif msg_type == MessageType.OBSERVATION:
                    if agent_id is None:
                        continue

                    obs = FullObservation.from_dict(payload.get("observation", {}))
                    action, metrics = self.system.step(agent_id, obs)

                    frame = encode_frame(
                        MessageType.ACTION,
                        self.seq_counter,
                        {
                            "agent_id": agent_id,
                            "action": action.to_dict(),
                            "metrics": metrics,
                        },
                    )
                    self.seq_counter += 1
                    writer.write(frame)
                    await writer.drain()

                    step_num = self.system.total_steps
                    if step_num <= 10 or step_num % 5 == 0 or metrics.get("checkpoint_saved"):
                        pos = metrics.get("pos", (0.0, 0.0, 0.0))
                        pos_str = f"({pos[0]}, {pos[1]}, {pos[2]})"
                        motor = metrics.get("motor", (0.0, 0.0, 0.0))
                        motor_str = f"mv_z={motor[1]:+.2f}, mv_x={motor[0]:+.2f}, yaw={motor[2]:+.2f}"
                        if metrics.get("jump"):
                            motor_str += " [JUMP]"
                        if metrics.get("sprint"):
                            motor_str += " [SPRINT]"

                        logger.info(
                            "========== [SHARED STEP %05d | %s | EPISODE %s] ==========",
                            step_num,
                            agent_id,
                            self.system.contexts[agent_id].episode_count,
                        )
                        logger.info(
                            "  [OBS]           pos=%s | health=%.1f | food=%.1f | voxels=%s | entities=%s",
                            pos_str,
                            metrics.get("health", 0.0),
                            metrics.get("food", 0.0),
                            metrics.get("voxels_count", 0),
                            metrics.get("entities_count", 0),
                        )
                        logger.info(
                            "  [ACTION]        motor=(%s) | prim=%s | personality=%s | source=%s",
                            motor_str,
                            metrics.get("primitive_name", action.command.primitive.value),
                            metrics.get("personality", "?"),
                            metrics.get("personality_action_source", "policy"),
                        )
                        logger.info(
                            "  [ACTION_RESULT] prev=%s | success=%s | penalty=%.2f | step_reward=%.4f",
                            metrics.get("last_action_primitive", "noop"),
                            metrics.get("last_action_success", True),
                            metrics.get("consequence_penalty", 0.0),
                            metrics.get("step_reward", 0.0),
                        )
                        logger.info(
                            "  [REPLAY]        buffer_size=%s transitions | stored_episodes=%s",
                            metrics.get("replay_size", 0),
                            metrics.get("replay_episodes", 0),
                        )
                        if "wm_loss" in metrics:
                            logger.info(
                                "  [WORLD_MODEL]   loss=%.4f | recon=%.4f | kl=%.4f",
                                metrics.get("wm_loss", 0.0),
                                metrics.get("recon_loss", 0.0),
                                metrics.get("kl_loss", 0.0),
                            )
                            logger.info(
                                "  [LEARN]         backprop=SUCCESS | ac_loss=%.4f | policy=%.4f | value=%.4f | mean_return=%.4f",
                                metrics.get("ac_loss", 0.0),
                                metrics.get("policy_loss", metrics.get("actor_loss", 0.0)),
                                metrics.get("value_loss", metrics.get("critic_loss", 0.0)),
                                metrics.get("mean_imagined_return", 0.0),
                            )
                        else:
                            logger.info(
                                "  [WORLD_MODEL]   warming up replay buffer (%s/%s needed)",
                                metrics.get("replay_size", 0),
                                self.system.cfg.sequence_length,
                            )
                        logger.info(
                            "  [RND]           curiosity=%.4f | spatial_novelty=%.4f | regions=%s",
                            metrics.get("curiosity", 0.0),
                            metrics.get("spatial_novelty", 0.0),
                            metrics.get("regions_explored", 0),
                        )
                        logger.info(
                            "  [SKILL]         active_skill_id=%s | ticks_remaining=%s",
                            metrics.get("skill_id", 0),
                            metrics.get("skill_duration", 0),
                        )
                        logger.info(
                            "  [MPC]           latent_imagined_value=%.4f | planned_primitive=%s",
                            metrics.get("value", 0.0),
                            metrics.get("primitive_name", action.command.primitive.value),
                        )

                elif msg_type == MessageType.DEATH:
                    if agent_id is None:
                        continue
                    obs = FullObservation.from_dict(payload.get("observation", {}))
                    obs.done = True
                    self.system.step(agent_id, obs)
                    logger.info(
                        "Death processed | agent=%s | episode=%s | shared_step=%s",
                        agent_id,
                        self.system.contexts[agent_id].episode_count,
                        self.system.total_steps,
                    )

                elif msg_type == MessageType.PING:
                    frame = encode_frame(
                        MessageType.PONG,
                        self.seq_counter,
                        {
                            "shared_steps": self.system.total_steps,
                            "agents": self.system.snapshot(),
                        },
                    )
                    self.seq_counter += 1
                    writer.write(frame)
                    await writer.drain()

        except Exception:
            logger.exception("Error handling stream client %s (agent=%s).", peer, agent_id)
        finally:
            writer.close()
            await writer.wait_closed()

    async def start(self):
        self.server = await asyncio.start_server(self.handle_client, self.host, self.port)
        logger.info(
            "Multi-agent Environment Contract v1 server listening on %s:%s",
            self.host,
            self.port,
        )
        logger.info("Equal peers: %s", ", ".join(self.system.AGENT_IDS))
        async with self.server:
            await self.server.serve_forever()


def run_server(config_path: Optional[str] = None):
    cfg = Config.from_json(config_path) if config_path else Config()
    system = MultiAgentLearningSystem(cfg)
    logger.info(
        "Shared learner initialized on %s for equal peers: %s",
        system.device,
        ", ".join(system.AGENT_IDS),
    )
    server = AgentStreamServer(system, host=cfg.stream_host, port=cfg.stream_port)
    asyncio.run(server.start())


if __name__ == "__main__":
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else None
    run_server(cfg_file)
