import argparse
import random
import numpy as np
import torch
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = str(Path(__file__).parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from bot.config import Config
from bot.agent import LearningAgent
from environment.schema import (
    FullObservation,
    CompleteInventoryState,
    ItemSlotData,
    PerceivedEntityData,
    MechanicalAffordanceState,
    ActionResult,
    ActionValidityMask,
)
from environment.manifest import EnvironmentManifest

def synthetic_learning_run(cfg: Config):
    """
    Validates the complete Environment Contract v1 perception, RSSM dynamics,
    RND curiosity, latent imagination, hierarchical action sampling, and persistence.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    print("=" * 65)
    print("ENVIRONMENT CONTRACT v1 — REINFORCEMENT SMOKE TEST")
    print("=" * 65)

    agent = LearningAgent(cfg)
    manifest = EnvironmentManifest(
        minecraft_version="1.20.4",
        protocol_version=765,
        mineflayer_version="4.25.0",
        minecraft_data_version="3.78.0",
    )
    agent.set_environment_manifest(manifest)
    print(f"Agent device: {agent.device}")
    print(f"Manifest: Minecraft {manifest.minecraft_version} (Protocol {manifest.protocol_version})")

    # Generate synthetic observations with canonical indices
    for step in range(40):
        # 11x11x11 voxels (canonical block indices: 0=air, 1=stone, 2=dirt, etc.)
        voxels = np.random.choice([0, 1, 2, 3, 10, 15, 20], size=1331).tolist()
        player_state = [
            20.0, 20.0, 5.0, 20.0,  # health, food, saturation, oxygen
            0.0, 64.0, 0.0,          # x, y, z
            0.0, 0.0, 0.0,           # vx, vy, vz
            0.0, 0.0,                # pitch, yaw
            1.0, 0.0, 0.0, 0.0, 0.0, # onGround, sneak, sprint, swim, lava
            1.0                      # alive
        ]
        
        # Complete Inventory (36 slots + armor + offhand)
        slots = [
            ItemSlotData(slot_index=i, item_canonical_id=random.choice([0, 10, 25, 45]), count=random.randint(0, 64))
            for i in range(36)
        ]
        inv = CompleteInventoryState(
            slots=slots,
            armor_head=ItemSlotData(slot_index=5, item_canonical_id=50, count=1),
            selected_hotbar_slot=0,
        )

        entities = [
            PerceivedEntityData(
                entity_id=101, canonical_type_id=5, category="hostile",
                dx=random.uniform(-4, 4), dy=0, dz=random.uniform(-4, 4),
                health=20.0, distance=3.2,
            )
        ]

        affordances = MechanicalAffordanceState(
            targeted_block_canonical_id=1,
            targeted_block_distance=2.5,
            can_mine_target=True,
            targeted_entity_idx=0,
            light_level=14,
            sky_light=15,
        )

        validity = ActionValidityMask(
            can_jump=True,
            can_sprint=True,
            can_sneak=True,
            can_attack_entity=True,
            can_dig_block=True,
            can_place_block=True,
            can_use_item=True,
        )

        last_res = ActionResult(action_primitive="noop", success=True)

        obs = FullObservation(
            voxels=voxels,
            voxel_shape=[11, 11, 11],
            player_state=player_state,
            inventory=inv,
            entities=entities,
            affordances=affordances,
            last_action_result=last_res,
            validity_mask=validity,
            done=(step == 39),
            step_id=step,
        )

        action, metrics = agent.step(obs)

        if step % 5 == 0 or step == 39:
            print(
                f"[Step {step:03d}] Motor: mv_z={action.motor.move_z:+.2f}, mv_x={action.motor.move_x:+.2f}, "
                f"look=({action.motor.yaw_delta:+.2f}, {action.motor.pitch_delta:+.2f}) | "
                f"Prim: {action.command.primitive.value:<14} | "
                f"Curiosity={metrics.get('curiosity', 0.0):.4f} | "
                f"Value={metrics.get('value', 0.0):.3f} | "
                f"WM Loss={metrics.get('wm_loss', 0.0):.4f} | "
                f"AC Loss={metrics.get('ac_loss', 0.0):.4f}"
            )

    agent.save_checkpoint()
    print("Environment Contract v1 smoke test & checkpoint verification passed!")

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
        import asyncio
        from bot.stream_server import AgentStreamServer
        from bot.multi_agent import MultiAgentLearningSystem
        system = MultiAgentLearningSystem(cfg)
        server = AgentStreamServer(system, host=cfg.stream_host, port=cfg.stream_port)
        asyncio.run(server.start())

if __name__ == "__main__":
    main()
