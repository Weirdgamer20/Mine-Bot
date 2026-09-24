import asyncio
import struct
import json
import numpy as np

from bot.config import Config
from bot.agent import LearningAgent
from bot.stream_server import AgentStreamServer
from bot.schemas import Observation, Affordances

async def run_test():
    cfg = Config()
    cfg.stream_port = 9199  # test port
    agent = LearningAgent(cfg)
    server = AgentStreamServer(agent, host="127.0.0.1", port=cfg.stream_port)
    
    server_task = asyncio.create_task(server.start())
    await asyncio.sleep(0.5)

    # Connect client via raw TCP
    reader, writer = await asyncio.open_connection("127.0.0.1", cfg.stream_port)

    # 1. Test PING
    ping_payload = json.dumps({"type": "PING"}).encode("utf-8")
    writer.write(struct.pack("!I", len(ping_payload)) + ping_payload)
    await writer.drain()

    header = await reader.readexactly(4)
    length = struct.unpack("!I", header)[0]
    res = json.loads((await reader.readexactly(length)).decode("utf-8"))
    assert res["type"] == "PONG"
    print("PING test passed!")

    # 2. Test OBSERVE frame
    obs = Observation(
        voxels=[0] * 1331,
        player_state=[20.0, 20.0, 5.0, 20.0, 0, 64, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        affordances=Affordances(target_block_id=1, can_mine=True),
    )
    obs_dict = {
        "voxels": obs.voxels,
        "player_state": obs.player_state,
        "inventory": [],
        "entities": [],
        "affordances": {
            "target_block_id": 1,
            "target_block_distance": 2.0,
            "can_mine": True,
            "light_level": 15,
            "time_of_day": 0.5,
            "is_raining": False,
            "is_sleeping": False,
            "is_swimming": False,
            "equipped_item_id": 0,
        },
        "done": False,
    }
    msg_payload = json.dumps({"type": "OBSERVE", "observation": obs_dict}).encode("utf-8")
    writer.write(struct.pack("!I", len(msg_payload)) + msg_payload)
    await writer.drain()

    header = await reader.readexactly(4)
    length = struct.unpack("!I", header)[0]
    action_res = json.loads((await reader.readexactly(length)).decode("utf-8"))
    assert action_res["type"] == "ACTION"
    assert "action" in action_res
    assert "move_x" in action_res["action"]
    print(f"OBSERVE test passed! Action returned: {action_res['action']}")

    writer.close()
    await writer.wait_closed()
    server.server.close()
    await server.server.wait_closed()
    server_task.cancel()
    print("ALL STREAMING TESTS PASSED!")

if __name__ == "__main__":
    asyncio.run(run_test())
