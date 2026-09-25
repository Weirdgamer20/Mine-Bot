import asyncio
import time
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = str(Path(__file__).parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from bot.config import Config
from bot.agent import LearningAgent
from bot.stream_server import AgentStreamServer
from bot.protocol.framing import MessageType, encode_frame, StreamFramingReader
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

async def run_protocol_v1_test():
    cfg = Config()
    cfg.stream_port = 9199
    agent = LearningAgent(cfg)
    server = AgentStreamServer(agent, host="127.0.0.1", port=cfg.stream_port)

    server_task = asyncio.create_task(server.start())
    await asyncio.sleep(0.5)

    reader, writer = await asyncio.open_connection("127.0.0.1", cfg.stream_port)
    stream_reader = StreamFramingReader(reader)

    # 1. Test HELLO Handshake
    manifest = EnvironmentManifest(minecraft_version="1.20.4", protocol_version=765)
    hello_frame = encode_frame(MessageType.HELLO, 1, {"agent_id": "LB-01", "manifest": manifest.to_dict()})
    writer.write(hello_frame)
    await writer.drain()

    welcome_msg = await stream_reader.read_frame()
    assert welcome_msg is not None
    msg_type, seq_id, payload = welcome_msg
    assert msg_type == MessageType.WELCOME
    assert payload.get("status") == "ready"
    print("HELLO -> WELCOME handshake test passed!")

    # 2. Test PING -> PONG
    ping_frame = encode_frame(MessageType.PING, 2, {})
    writer.write(ping_frame)
    await writer.drain()

    pong_msg = await stream_reader.read_frame()
    assert pong_msg is not None
    msg_type, seq_id, payload = pong_msg
    assert msg_type == MessageType.PONG
    print("PING -> PONG heartbeat test passed!")

    # 3. Test OBSERVATION -> ACTION (Environment Contract v1)
    obs = FullObservation(
        voxels=[0] * 1331,
        player_state=[20.0, 20.0, 5.0, 20.0, 0, 64, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        inventory=CompleteInventoryState(
            slots=[ItemSlotData(slot_index=0, item_canonical_id=1, count=1)],
        ),
        entities=[
            PerceivedEntityData(entity_id=1, canonical_type_id=1, dx=1.0, dy=0.0, dz=1.0, distance=1.4)
        ],
        affordances=MechanicalAffordanceState(targeted_block_canonical_id=1, can_mine_target=True),
        last_action_result=ActionResult(action_primitive="noop", success=True),
        validity_mask=ActionValidityMask(can_jump=True, can_dig_block=True),
        step_id=1,
    )

    obs_frame = encode_frame(
        MessageType.OBSERVATION,
        3,
        {
            "agent_id": "LB-01",
            "sequence": 1,
            "world_tick": 1,
            "timestamp_ns": time.perf_counter_ns(),
            "observation": obs.to_dict(),
        },
    )
    writer.write(obs_frame)
    await writer.drain()

    action_msg = await stream_reader.read_frame()
    assert action_msg is not None
    msg_type, seq_id, payload = action_msg
    assert msg_type == MessageType.ACTION
    assert "action" in payload
    action_dict = payload["action"]
    assert "motor" in action_dict
    assert "command" in action_dict
    print(f"OBSERVATION -> ACTION test passed!")
    print(f"Action command received: {action_dict['command']['primitive']}")
    print(f"Motor locomotion: move_z={action_dict['motor']['move_z']:+.2f}, move_x={action_dict['motor']['move_x']:+.2f}")

    writer.close()
    await writer.wait_closed()
    server.running = False
    server.server.close()
    await server.server.wait_closed()
    server_task.cancel()
    print("ALL ENVIRONMENT CONTRACT v1 TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    import os
    asyncio.run(run_protocol_v1_test())
    os._exit(0)
