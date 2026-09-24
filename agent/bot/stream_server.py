import asyncio
import logging
import sys
from typing import Optional

from .agent import LearningAgent
from .schemas import FullObservation, EnvironmentManifest
from .protocol.framing import MessageType, encode_frame, StreamFramingReader
from .config import Config

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("StreamServer")

class AgentStreamServer:
    """
    Direct persistent TCP streaming server implementing the Environment Contract v1.
    Uses 16-byte fixed binary header framing with CRC32 verification.
    """
    def __init__(self, agent: LearningAgent, host: str = "0.0.0.0", port: int = 9099):
        self.agent = agent
        self.host = host
        self.port = port
        self.server = None
        self.seq_counter = 1

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        peer = writer.get_extra_info("peername")
        logger.info(f"Mineflayer bridge connected from {peer}")
        framing_reader = StreamFramingReader(reader)
        self.agent.reset_episode()

        try:
            while True:
                msg = await framing_reader.read_frame()
                if msg is None:
                    logger.info("Mineflayer bridge disconnected.")
                    break

                msg_type, seq_id, payload = msg

                if msg_type == MessageType.HELLO:
                    manifest_data = payload.get("manifest", {})
                    manifest = EnvironmentManifest.from_dict(manifest_data)
                    self.agent.set_environment_manifest(manifest)
                    logger.info(
                        f"Handshake HELLO received | MC: {manifest.minecraft_version} | "
                        f"Protocol: {manifest.protocol_version} | Mineflayer: {manifest.mineflayer_version}"
                    )
                    welcome_frame = encode_frame(
                        MessageType.WELCOME,
                        self.seq_counter,
                        {"status": "ready", "agent_device": str(self.agent.device)},
                    )
                    self.seq_counter += 1
                    writer.write(welcome_frame)
                    await writer.drain()

                elif msg_type == MessageType.OBSERVATION:
                    obs_dict = payload.get("observation", {})
                    obs = FullObservation.from_dict(obs_dict)
                    action, metrics = self.agent.step(obs)

                    action_frame = encode_frame(
                        MessageType.ACTION,
                        self.seq_counter,
                        {"action": action.to_dict(), "metrics": metrics},
                    )
                    self.seq_counter += 1
                    writer.write(action_frame)
                    await writer.drain()

                    if self.agent.total_steps % 50 == 0:
                        logger.info(
                            f"Step {self.agent.total_steps} | "
                            f"Curiosity: {metrics.get('curiosity', 0.0):.4f} | "
                            f"Value: {metrics.get('value', 0.0):.3f} | "
                            f"Skill: {metrics.get('skill_id', 0)} | "
                            f"Primitive: {action.command.primitive.value}"
                        )

                elif msg_type == MessageType.DEATH:
                    obs_dict = payload.get("observation", {})
                    obs = FullObservation.from_dict(obs_dict)
                    obs.done = True
                    self.agent.step(obs)
                    logger.info(f"Agent death event processed. Episode {self.agent.episode_count} ended.")

                elif msg_type == MessageType.PING:
                    pong_frame = encode_frame(
                        MessageType.PONG,
                        self.seq_counter,
                        {"steps": self.agent.total_steps},
                    )
                    self.seq_counter += 1
                    writer.write(pong_frame)
                    await writer.drain()

        except Exception as e:
            logger.error(f"Error handling stream client: {e}", exc_info=True)
        finally:
            writer.close()
            await writer.wait_closed()

    async def start(self):
        self.server = await asyncio.start_server(self.handle_client, self.host, self.port)
        logger.info(f"Environment Contract v1 Persistent Streaming Server listening on {self.host}:{self.port}")
        async with self.server:
            await self.server.serve_forever()

def run_server(config_path: Optional[str] = None):
    cfg = Config.from_json(config_path) if config_path else Config()
    agent = LearningAgent(cfg)
    logger.info(f"Learning Agent initialized on device: {agent.device}")
    server = AgentStreamServer(agent, host=cfg.stream_host, port=cfg.stream_port)
    asyncio.run(server.start())

if __name__ == "__main__":
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else None
    run_server(cfg_file)
