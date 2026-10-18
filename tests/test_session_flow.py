import random

import pytest

from core.peer_benchmark import Judgment, eigentrust, peer_judgments, select_top_k, spearman
from core.session_flow import build_community, inference_accuracy, run_session, topic_shares
from core.topic_graph import Membership, TopicGraph


@pytest.fixture(scope="module")
def community():
    return build_community(seed=0)


def test_arrival_order_rewards_the_first_arrival_and_shapley_does_not(community):
    g = community.graph
    for topic in g.topics():
        first = g.first_arrival(topic)
        arrival = community.shares["arrival"][topic]
        assert arrival[first] == max(arrival.values())
    # quant-volatility: ana covers 3 of 4 subthemes first, ben's only subtheme was already covered
    q_arr = community.shares["arrival"]["quant-volatility"]
    q_shap = community.shares["shapley"]["quant-volatility"]
    assert q_arr == pytest.approx({"ana": 0.75, "skewtrader": 0.25, "ben": 0.0})
    assert q_shap["ben"] > 0 and q_shap["ana"] < q_arr["ana"]


def test_same_component_same_access_different_stake(community):
    ana, ben = run_session(community, "ana"), run_session(community, "ben")
    assert ana.exploration.topic_degree == ben.exploration.topic_degree
    assert ana.takeaway.coverage == pytest.approx(ben.takeaway.coverage)
    assert ana.takeaway.stake["selection"] > ben.takeaway.stake["selection"] == 0
    assert ben.takeaway.selected_on == []


def test_other_component_is_unreachable(community):
    r = run_session(community, "hana")
    assert r.exploration.unreachable and "ana" in r.exploration.unreachable
    assert r.takeaway.coverage < run_session(community, "ana").takeaway.coverage


def test_unknown_share_rule(community):
    with pytest.raises(ValueError):
        topic_shares(community.graph, community.pools, "lottery")


@pytest.mark.parametrize("seed", range(4))
def test_emergent_picks_settle_include_the_best_and_exclude_colluders(seed):
    c = build_community(seed=seed, top_k=2)
    assert c.selection.converged
    for topic in c.graph.topics():
        picks = c.selection.top_k[topic]
        best = max(c.graph.members_of(topic), key=lambda u: c.skill[u][topic])
        assert best in picks
        assert not set(picks) & c.colluders
        assert sum(c.shares["selection"][topic].values()) == pytest.approx(1.0)


def test_picks_anchor_the_benchmark_without_any_designation(community):
    g = community.graph
    true = {u: sum(community.skill[u][t] for t in g.topics_of(u)) / len(g.topics_of(u)) for u in g.users()}
    rank = sorted(community.trust, key=lambda u: -community.trust[u])
    assert {rank.index(s) for s in community.colluders} == {len(rank) - 2, len(rank) - 1}
    judgments = peer_judgments(g, community.skill, random.Random(0), colluders=community.colluders)
    unanchored, _, _ = eigentrust(g.users(), judgments)
    assert spearman(community.trust, true) > spearman(unanchored, true)


def test_two_person_topic_with_no_evidence_picks_both(community):
    assert community.selection.top_k["bonsai"] == ["hana", "ivo"]


def test_select_top_k_keeps_ties_and_ignores_zero_trust():
    g = TopicGraph([Membership(u, "t", i) for i, u in enumerate("abcd")])
    assert select_top_k(g, {"t": {"a": 0.4, "b": 0.4, "c": 0.2, "d": 0.0}}, k=1) == {"t": ["a", "b"]}
    assert select_top_k(g, {"t": {"a": 0.5, "b": 0.0, "c": 0.0, "d": 0.0}}, k=2) == {"t": ["a"]}
    assert select_top_k(g, {"t": {}}, k=1) == {"t": ["a", "b", "c", "d"]}


def test_members_can_infer_the_picks_from_the_agents_answers(community):
    few, many = inference_accuracy(community, 3), inference_accuracy(community, 20)
    assert many > few
    assert many > 0.9


def test_eigentrust_is_a_distribution_and_flows_along_judgments():
    users = ["a", "b", "c"]
    js = [Judgment("a", "t", "b", "c")] * 3
    t, _, reach = eigentrust(users, js, pretrusted=["a"])
    assert sum(t.values()) == pytest.approx(1.0)
    assert t["b"] > t["c"] == 0
    assert reach == 1
    weighted, _, _ = eigentrust(users, [], pretrusted={"a": 3, "b": 1})
    assert weighted == pytest.approx({"a": 0.75, "b": 0.25, "c": 0.0})


def test_two_person_topics_get_no_peer_judgments(community):
    js = peer_judgments(community.graph, community.skill, random.Random(1))
    assert not any(j.topic == "bonsai" for j in js)
