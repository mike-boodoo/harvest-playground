import random

import pytest

from core.peer_benchmark import Judgment, eigentrust, peer_judgments, spearman
from core.session_flow import build_community, run_session, topic_shares


@pytest.fixture(scope="module")
def community():
    return build_community(seed=0)


def test_arrival_order_rewards_the_moderator_and_shapley_does_not(community):
    g = community.graph
    for topic in g.topics():
        mod = g.moderator(topic)
        arrival = community.shares["arrival"][topic]
        assert arrival[mod] == max(arrival.values())
    # quant-volatility: ana covers 3 of 4 subthemes first, ben's only subtheme was already covered
    q_arr = community.shares["arrival"]["quant-volatility"]
    q_shap = community.shares["shapley"]["quant-volatility"]
    assert q_arr == pytest.approx({"ana": 0.75, "skewtrader": 0.25, "ben": 0.0})
    assert q_shap["ben"] > 0 and q_shap["ana"] < q_arr["ana"]


def test_same_component_same_access_different_stake(community):
    ana, ben = run_session(community, "ana"), run_session(community, "ben")
    assert ana.exploration.topic_degree == ben.exploration.topic_degree
    assert ana.takeaway.coverage == pytest.approx(ben.takeaway.coverage)
    assert ana.takeaway.stake["arrival"] > ben.takeaway.stake["arrival"] == 0


def test_other_component_is_unreachable(community):
    r = run_session(community, "hana")
    assert r.exploration.unreachable and "ana" in r.exploration.unreachable
    assert r.takeaway.coverage < run_session(community, "ana").takeaway.coverage


def test_unknown_share_rule(community):
    with pytest.raises(ValueError):
        topic_shares(community.graph, community.pools, "lottery")


def test_moderator_anchor_resists_colluders(community):
    g = community.graph
    true = {u: sum(community.skill[u][t] for t in g.topics_of(u)) / len(g.topics_of(u)) for u in g.users()}
    judgments = peer_judgments(g, community.skill, random.Random(0), colluders=community.colluders)
    mods = sorted({g.moderator(t) for t in g.topics()})
    anchored, _, _ = eigentrust(g.users(), judgments, pretrusted=mods)
    unanchored, _, _ = eigentrust(g.users(), judgments)
    rank = sorted(anchored, key=lambda u: -anchored[u])
    assert {rank.index(s) for s in community.colluders} == {len(rank) - 2, len(rank) - 1}
    assert spearman(anchored, true) > spearman(unanchored, true)


def test_eigentrust_is_a_distribution_and_flows_along_judgments():
    users = ["a", "b", "c"]
    js = [Judgment("a", "t", "b", "c")] * 3
    t, _, reach = eigentrust(users, js, pretrusted=["a"])
    assert sum(t.values()) == pytest.approx(1.0)
    assert t["b"] > t["c"] == 0
    assert reach == 1


def test_two_person_topics_get_no_peer_judgments(community):
    js = peer_judgments(community.graph, community.skill, random.Random(1))
    assert not any(j.topic == "bonsai" for j in js)
