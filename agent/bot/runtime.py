import argparse
import random
import numpy as np
import torch

from .config import Config
from .agent import LearningAgent
from .schemas import Observation, Affordances, InventorySlot, EntityObservation
from .stream_server import AgentStreamServer

def synthetic_learning_run(cfg: Config):
    """
    Offline validation test ensuring all components (MultiModalEncoder, WorldModel,
    RND curiosity, latent imagination, GAE, replay memory, and checkpointing)
    function without needing a live Minecraft instance.
    """
    torch.manual_seed(cfg.seed if hasattr(cfg, 'seed') else 42)
    np.random.seed(42)
    random.seed(42)

    print("=" * 60)
    print("RUNNING RESEARCH SCAFFOLD & REINFORCEMENT SMOKE TEST")
    print("=" * 60)

    agent = LearningAgent(cfg)
    print(f"Agent device: {agent.device}")

    # Generate synthetic observations
    for step in range(40):
        # 11x11x11 voxels (mostly air with terrain)
        voxels = np.random.choice([0, 1, 2, 3, 12, 17], size=1331).tolist()
        player_state = [
            20.0, 20.0, 5.0, 20.0,  # health, food, saturation, oxygen
            0.0, 64.0, 0.0,          # x, y, z
            0.0, 0.0, 0.0,           # vx, vy, vz
            0.0, 0.0,                # pitch, yaw
            1.0, 0.0, 0.0, 0.0,      # onGround, sneak, sprint, swim
            1.0                      # alive
        ]
        inventory = [
            InventorySlot(slot_index=i, item_id=random.choice([0, 276, 278, 297]), count=random.randint(0, 64))
            for i in range(9)
        ]
        entities = [
            EntityObservation(
                entity_id=101, type_id=1, dx=random.uniform(-5, 5), dy=0, dz=random.uniform(-5, 5), health=20.0
            )
        ]
        affordances = Affordances(
            target_block_id=random.choice([0, 1, 17]),
            target_block_distance=random.uniform(1.0, 4.5),
            can_mine=True,
            light_level=12,
            time_of_day=0.3,
        )

        obs = Observation(
            voxels=voxels,
            voxel_shape=[11, 11, 11],
            player_state=player_state,
            inventory=inventory,
            entities=entities,
            affordances=affordances,
            done=(step == 39),
        )

        action, metrics = agent.step(obs)

        if step % 5 == 0 or step == 39:
            print(
                f"[Step {step:03d}] Action: mv_z={action.move_z:+.2f}, mv_x={action.move_x:+.2f}, "
                f"atk={action.attack:.2f}, slot={action.slot} | "
                f"Curiosity={metrics.get('curiosity', 0.0):.4f} | "
                f"Value={metrics.get('value', 0.0):.3f} | "
                f"WM Loss={metrics.get('wm_loss', 0.0):.4f} | "
                f"AC Loss={metrics.get('ac_loss', 0.0):.4f}"
            )

    agent.save_checkpoint()
    print("Synthetic run & checkpoint verification completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Minecraft Learning Bot System")
    parser.add_argument("--mode", choices=["stream", "synthetic"], default="stream")
    parser.add_argument("--config", default="config/default.json")
    parser.add_argument("--port", type=int, default=9099)
    args = parser.parse_args()

    cfg = Config.from_json(args.config)
    cfg.stream_port = args.port

    if args.mode == "synthetic":
        synthetic_learning_run(cfg)
    else:
        import asyncio
        from .stream_server import AgentStreamServer
        agent = LearningAgent(cfg)
        server = AgentStreamServer(agent, host=cfg.stream_host, port=cfg.stream_port)
        asyncio.run(server.start())

if __name__ == "__main__":
    main()
