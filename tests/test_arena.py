import random

import pytest

from core.target import GenerationResult, Target
from playground import arena


class Echo(Target):
    license_tag = "test"

    def __init__(self, name):
        self.model_id = name

    def generate(self, prompt, **kwargs):
        return GenerationResult(prompt=prompt, response=f"{self.model_id} says hi", logprobs=None, latency_s=0.0)


def contenders():
    return {"alpha": Echo("alpha"), "beta": Echo("beta")}


def test_match_pairs_each_side_with_its_own_response():
    for seed in range(6):
        m = arena.make_match("hello", "greetings", contenders(), rng=random.Random(seed))
        assert m.left_response == f"{m.left_model} says hi"
        assert m.right_response == f"{m.right_model} says hi"
        assert {m.left_model, m.right_model} == {"alpha", "beta"}


def test_sides_are_shuffled():
    lefts = {arena.make_match("p", "t", contenders(), rng=random.Random(s)).left_model for s in range(20)}
    assert lefts == {"alpha", "beta"}


def test_match_validation():
    with pytest.raises(ValueError, match="exactly two"):
        arena.make_match("p", "t", {"alpha": Echo("alpha")})
    with pytest.raises(ValueError, match="prompt"):
        arena.make_match("  ", "t", contenders())


def test_votes_round_trip_and_reject_unknown(tmp_path):
    log = tmp_path / "votes.jsonl"
    m = arena.make_match("p", "t", contenders(), rng=random.Random(0))
    v = arena.record_vote(m, "left", log)
    assert v.winner == m.left_model
    assert arena.record_vote(m, "tie", log).winner is None
    assert [x.outcome for x in arena.load_votes(log)] == ["left", "tie"]
    with pytest.raises(ValueError):
        arena.record_vote(m, "middle", log)
    assert arena.load_votes(tmp_path / "missing.jsonl") == []


def _vote(a, b, winner, outcome=None, topic="t"):
    outcome = outcome or ("tie" if winner is None else "left")
    return arena.Vote("p", topic, a, b, winner, outcome, 0.0)


def test_elo_rewards_the_winner_and_is_zero_sum():
    ratings = arena.elo([_vote("a", "b", "a")] * 5)
    assert ratings["a"] > 1000 > ratings["b"]
    assert ratings["a"] + ratings["b"] == pytest.approx(2000)


def test_ties_between_equals_change_nothing_and_both_bad_is_skipped():
    assert arena.elo([_vote("a", "b", None)]) == {"a": 1000, "b": 1000}
    assert arena.elo([_vote("a", "b", None, "both_bad")]) == {}


def test_standings_and_topic_breakdown():
    votes = [_vote("a", "b", "a", topic="math"), _vote("b", "a", "a", topic="math"),
             _vote("a", "b", "b", topic="law"), _vote("a", "b", None, topic="law")]
    rows = {r["model"]: r for r in arena.standings(votes)}
    assert rows["a"]["matches"] == 4 and rows["a"]["wins"] == 2 and rows["a"]["ties"] == 1
    assert rows["a"]["win_rate"] == 62.5  # (2 + 0.5) / 4
    assert arena.standings(votes)[0]["model"] == "a"
    by_topic = arena.win_rate_by_topic(votes, "a")
    assert by_topic == [{"topic": "law", "matches": 2, "win_rate": 25.0},
                        {"topic": "math", "matches": 2, "win_rate": 100.0}]
