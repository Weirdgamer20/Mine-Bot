import asyncio
import struct
import json
import logging
import sys
from typing import Optional

from .agent import LearningAgent
from .schemas import Observation
from .config import Config

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("StreamServer")

async def read_frame(reader: asyncio.StreamReader) -> Optional[dict]:
    """Reads a length-prefixed frame (4-byte big-endian uint32 + JSON)."""
    try:
        header = await reader.readexactly(4)
    except (asyncio.IncompleteReadError, ConnectionResetError):
        return None
    length = struct.unpack("!I", header)[0]
    payload = await reader.readexactly(length)
    return json.loads(payload.decode("utf-8"))

async def write_frame(writer: asyncio.StreamWriter, data: dict):
    """Writes a length-prefixed frame (4-byte big-endian uint32 + JSON)."""
    payload = json.dumps(data).encode("utf-8")
    header = struct.pack("!I", len(payload))
    writer.write(header + payload)
    await writer.drain()

class AgentStreamServer:
    """
    Direct persistent TCP streaming server for the Minecraft Mineflayer bridge.
    Replaces REST APIs with persistent, high-frequency, bidirectional socket streaming.
    """
    def __init__(self, agent: LearningAgent, host: str = "0.0.0.0", port: int = 9099):
        self.agent = agent
        self.host = host
        self.port = port
        self.server = None

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        peer = writer.get_extra_info("peername")
        logger.info(f"Mineflayer bridge connected from {peer}")
        self.agent.reset_episode()

        try:
            while True:
                msg = await read_frame(reader)
                if msg is None:
                    logger.info("Mineflayer bridge disconnected.")
                    break

                msg_type = msg.get("type", "OBSERVE")
                
                if msg_type == "OBSERVE":
                    obs_dict = msg.get("observation", {})
                    obs = Observation.from_dict(obs_dict)
                    action, metrics = self.agent.step(obs)
                    
                    response = {
                        "type": "ACTION",
                        "action": action.to_dict(),
                        "metrics": metrics,
                    }
                    await write_frame(writer, response)

                    if self.agent.total_steps % 50 == 0:
                        logger.info(
                            f"Step {self.agent.total_steps} | "
                            f"Curiosity: {metrics.get('curiosity', 0.0):.4f} | "
                            f"Value: {metrics.get('value', 0.0):.3f} | "
                            f"Skill: {metrics.get('skill_id', 0)}"
                        )

                elif msg_type == "RESET":
                    self.agent.reset_episode()
                    await write_frame(writer, {"type": "ACK", "status": "reset"})

                elif msg_type == "PING":
                    await write_frame(writer, {"type": "PONG", "steps": self.agent.total_steps})

        except Exception as e:
            logger.error(f"Error handling stream client: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    async def start(self):
        self.server = await asyncio.start_server(self.handle_client, self.host, self.port)
        logger.info(f"Direct Persistent Streaming Server listening on {self.host}:{self.port}")
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
