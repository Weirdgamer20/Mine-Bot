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

                    step_num = self.agent.total_steps
                    if step_num <= 10 or step_num % 5 == 0 or metrics.get("checkpoint_saved"):
                        pos_str = f"({metrics['pos'][0]}, {metrics['pos'][1]}, {metrics['pos'][2]})"
                        motor_str = f"mv_z={metrics['motor'][1]:+.2f}, mv_x={metrics['motor'][0]:+.2f}, yaw={metrics['motor'][2]:+.2f}"
                        if metrics.get("jump"):
                            motor_str += " [JUMP]"
                        if metrics.get("sprint"):
                            motor_str += " [SPRINT]"

                        logger.info(f"========== [STEP {step_num:05d} | EPISODE {self.agent.episode_count}] ==========")
                        logger.info(f"  [OBS]           pos={pos_str} | health={metrics['health']:.1f} | food={metrics['food']:.1f} | voxels={metrics['voxels_count']} | entities={metrics['entities_count']}")
                        logger.info(f"  [ACTION]        motor=({motor_str}) | prim={metrics['primitive_name']}")
                        logger.info(f"  [ACTION_RESULT] prev_action={metrics['last_action_primitive']} | success={metrics['last_action_success']} | delta={metrics['last_action_delta']}")
                        logger.info(f"  [REPLAY]        buffer_size={metrics['replay_size']} transitions | stored_episodes={metrics['replay_episodes']}")

                        if "wm_loss" in metrics:
                            logger.info(f"  [WORLD_MODEL]   loss={metrics['wm_loss']:.4f} | recon={metrics.get('recon_loss', 0.0):.4f} | kl={metrics.get('kl_loss', 0.0):.4f}")
                            logger.info(f"  [LEARN]         backprop=SUCCESS | ac_loss={metrics.get('ac_loss', 0.0):.4f} | actor={metrics.get('actor_loss', 0.0):.4f} | critic={metrics.get('critic_loss', 0.0):.4f}")
                        else:
                            logger.info(f"  [WORLD_MODEL]   warming up replay buffer ({metrics['replay_size']}/{self.agent.cfg.sequence_length} needed for sequence batch)")
                            logger.info(f"  [LEARN]         accumulating sequence transitions before backprop")

                        logger.info(f"  [RND]           curiosity={metrics['curiosity']:.4f} | spatial_novelty={metrics['spatial_novelty']:.4f} | regions={metrics['regions_explored']}")
                        logger.info(f"  [SKILL]         active_skill_id={metrics['skill_id']} (DIAYN mode) | ticks_remaining={metrics['skill_duration']}")
                        logger.info(f"  [MPC]           latent_imagined_value={metrics['value']:.4f} | planned_primitive={metrics['primitive_name']}")

                        if metrics.get("checkpoint_saved"):
                            logger.info(f"  [CHECKPOINT]    *** ATOMIC CHECKPOINT SAVED TO DISK (step {step_num}) ***")

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
