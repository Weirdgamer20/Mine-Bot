from bot.personality import PERSONALITIES, get_personality


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


if __name__ == "__main__":
    test_four_equal_peer_ids()
    test_personality_names_are_distinct()
    test_warrior_is_confrontational_and_non_warriors_are_not_forced()
    test_all_peers_have_independent_personality_profiles()
    print("[TEST] ALL 4 MULTI-AGENT PERSONALITY TESTS PASSED!")

