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

    Every bridge connection identifies itself as LB-01..LB-04 in HELLO.
    All peers share one learner/world model/replay system, while recurrent and
    episode state is isolated per agent.
    """

    def __init__(
        self,
        system: MultiAgentLearningSystem,
        host: str = "0.0.0.0",
        port: int = 9099,
    ):
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
                        error = encode_frame(
                            MessageType.ERROR,
                            self.seq_counter,
                            {
                                "error": "INVALID_AGENT_ID",
                                "expected": list(self.system.AGENT_IDS),
                            },
                        )
                        self.seq_counter += 1
                        writer.write(error)
                        await writer.drain()
                        logger.error("Rejected peer %s with invalid agent_id=%r.", peer, agent_id)
                        break

                    manifest_data = payload.get("manifest", {})
                    manifest = EnvironmentManifest.from_dict(manifest_data)
                    self.system.shared.set_environment_manifest(manifest)
                    self.system.reset_agent(agent_id)

                    welcome = encode_frame(
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
                    writer.write(welcome)
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
                        {"agent_id": agent_id, "action": action.to_dict(), "metrics": metrics},
                    )
                    self.seq_counter += 1
                    writer.write(frame)
                    await writer.drain()

                    if self.system.total_steps % 50 == 0:
                        logger.info(
                            "SharedStep=%s | agent=%s | personality=%s | "
                            "curiosity=%.4f | value=%.3f | skill=%s | primitive=%s",
                            self.system.total_steps,
                            agent_id,
                            metrics.get("personality", "?"),
                            metrics.get("curiosity", 0.0),
                            metrics.get("value", 0.0),
                            metrics.get("skill_id", 0),
                            action.command.primitive.value,
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
