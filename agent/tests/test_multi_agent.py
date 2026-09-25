from bot.personality import PERSONALITIES, get_personality
from bot.multi_agent import MultiAgentLearningSystem
from bot.config import Config
from bot.schemas import (
    FullObservation,
    ActionValidityMask,
    MechanicalAffordanceState,
    CompleteInventoryState,
)


def make_dummy_obs(x: float = 0.0, z: float = 0.0, yaw: float = 0.0) -> FullObservation:
    return FullObservation(
        voxels=[0] * 1331,
        voxel_shape=[11, 11, 11],
        player_state=[20.0, 20.0, 5.0, 20.0, x, 64.0, z, yaw, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        inventory=CompleteInventoryState(),
        entities=[],
        affordances=MechanicalAffordanceState(),
        validity_mask=ActionValidityMask(),
        done=False,
        step_id=1,
    )


def test_four_equal_peer_ids():
    assert list(PERSONALITIES) == ["LB-01", "LB-02", "LB-03", "LB-04"]
    assert set(PERSONALITIES) == {"LB-01", "LB-02", "LB-03", "LB-04"}


def test_personality_names_are_distinct():
    names = {profile.name for profile in PERSONALITIES.values()}
    assert names == {"EXPLORER", "SURVIVOR", "WARRIOR", "OPPORTUNIST"}


def test_warrior_is_confrontational_and_non_warriors_are_not_forced():
    assert get_personality("LB-03").warrior_confrontation is True
    assert all(
        not profile.warrior_confrontation
        for agent_id, profile in PERSONALITIES.items()
        if agent_id != "LB-03"
    )


def test_all_peers_have_independent_personality_profiles():
    profiles = list(PERSONALITIES.values())
    assert len(profiles) == 4
    assert len({id(profile) for profile in profiles}) == 4


def test_peer_agent_instances_are_distinct():
    cfg = Config()
    system = MultiAgentLearningSystem(cfg)

    # 4 distinct agent instances
    peers = [system.get_agent(aid) for aid in system.AGENT_IDS]
    assert len(peers) == 4
    assert len({id(p) for p in peers}) == 4

    # Spatial memory, skills, experience graph are isolated
    assert len({id(p.spatial_memory) for p in peers}) == 4
    assert len({id(p.skill_library) for p in peers}) == 4
    assert len({id(p.experience_graph) for p in peers}) == 4
    assert len({id(p.benchmark_suite) for p in peers}) == 4

    # Neural models are shared
    assert len({id(p.world_model) for p in peers}) == 1
    assert id(peers[0].world_model) == id(system.shared.world_model)
    assert len({id(p.encoder) for p in peers}) == 1
    assert len({id(p.actor_critic) for p in peers}) == 1
    assert len({id(p.rnd) for p in peers}) == 1

    # Replay buffer is shared
    assert len({id(p.memory) for p in peers}) == 1
    assert id(peers[0].memory) == id(system.shared.memory)


def test_peer_step_spatial_isolation():
    cfg = Config()
    system = MultiAgentLearningSystem(cfg)

    p1 = system.get_agent("LB-01")
    p2 = system.get_agent("LB-02")

    # Step p1 with observation at (100, 200, yaw=45)
    obs_p1 = make_dummy_obs(x=100.0, z=200.0, yaw=45.0)
    action1, metrics1 = system.step("LB-01", obs_p1)

    # Verify p1's spatial memory was updated, but p2's was not
    assert (6, 12) in p1.spatial_memory.regions  # 100 // 16 = 6, 200 // 16 = 12
    assert (6, 12) not in p2.spatial_memory.regions
    assert p1.spatial_memory.get_visitation_count(100.0, 200.0) == 1
    assert p2.spatial_memory.get_visitation_count(100.0, 200.0) == 0

    # p1 steps incremented, p2 remained 0
    assert p1.episode_steps == 1
    assert p2.episode_steps == 0


if __name__ == "__main__":
    test_four_equal_peer_ids()
    test_personality_names_are_distinct()
    test_warrior_is_confrontational_and_non_warriors_are_not_forced()
    test_all_peers_have_independent_personality_profiles()
    test_peer_agent_instances_are_distinct()
    test_peer_step_spatial_isolation()
    print("[TEST] ALL MULTI-AGENT ARCHITECTURE AND ISOLATION TESTS PASSED!")
