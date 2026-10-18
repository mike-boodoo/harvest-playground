from core.topic_graph import (
    Membership, TopicGraph, reidentification_risk, topic_pseudonym, visible_view,
)


def chain():
    # a -#x- b -#y- c -#z- d, plus e/f/g on #w (k-sized), and h/i on their own #iso
    rows = [("a", "x", 1), ("b", "x", 2), ("b", "y", 3), ("c", "y", 4), ("c", "z", 5), ("d", "z", 6),
            ("d", "w", 7), ("e", "w", 8), ("f", "w", 9), ("h", "iso", 1), ("i", "iso", 2)]
    return TopicGraph([Membership(u, t, at) for u, t, at in rows])


def test_explore_runs_until_exhausted():
    exp = chain().explore("a")
    assert [h.degree for h in exp.hops] == [1, 2, 3, 4]
    assert exp.user_degree == {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 4}
    assert exp.topic_degree == {"x": 0, "y": 1, "z": 2, "w": 3}
    assert exp.unreachable == ["h", "i"]
    assert exp.exhausted_after == 4


def test_exhaustion_gives_a_component_the_same_reach():
    g = chain()
    reached = {u: set(g.explore(u).user_degree) for u in "abcdef"}
    assert len({frozenset(r) for r in reached.values()}) == 1
    assert g.diameter() == 4


def test_first_arrival_and_arrival_order():
    g = chain()
    assert g.first_arrival("y") == "b"
    assert g.members_of("w") == ["d", "e", "f"]


def test_pseudonyms_differ_across_topics():
    assert topic_pseudonym("b", "x") == topic_pseudonym("b", "x")
    assert topic_pseudonym("b", "x") != topic_pseudonym("b", "y")


def test_visible_view_names_only_degree_one_and_hides_small_topics():
    g = chain()
    views = visible_view(g, g.explore("a"), k=3)
    assert views[0].degree == 1 and views[0].topics_shown == ["x"]   # own topic, shown despite 2 members
    assert views[0].people == [topic_pseudonym("b", "x")]
    assert all(v.people == [] for v in views[1:])
    assert views[1].topics_shown == [] and views[1].topics_suppressed == 1
    assert views[3].topics_shown == ["w"] and views[3].people_count == 2


def test_reidentification_risk_counts_matching_signatures():
    risk = reidentification_risk(chain(), chain().explore("a"))
    assert risk["e"] == risk["f"] == 2   # same degree, same single topic
    assert risk["b"] == 1
