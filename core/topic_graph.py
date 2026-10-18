"""
Topic graph: who can reach whom through shared #topics, explored until
exhausted, and what each user is allowed to see along the way.

The graph is bipartite (users <-> topics). A user's degree of separation
from you is the number of topic hops it takes to reach them:

  degree 0  you
  degree 1  people on one of your topics
  degree 2  people on a topic one of THOSE people is on
  ...

`explore()` runs a breadth-first search from one user and keeps going
until the frontier is empty: every person and topic in your connected
component has been reached and nothing new can be. Anyone left over is
in a different component and is unreachable however long you look.

Exhaustion flattens access: everyone in the same component reaches the
same set of people and topics, only at different distances. So within a
component, distance and ownership are the only things that can make one
user's takeaway differ from another's (see core/session_flow.py).

`visible_view()` is the transparency policy: what the UI may show about
each hop. It is deliberately separate from the search, because the search
needs the full graph and the UI must not get it.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Membership:
    user: str
    topic: str
    joined_at: float
    subthemes: tuple[str, ...] = ()


@dataclass
class Hop:
    degree: int                 # degree of separation of the people found on this hop
    via_topics: list[str]       # topics first entered on this hop
    new_users: list[str]        # people first reached on this hop


@dataclass
class Exploration:
    ego: str
    hops: list[Hop]
    user_degree: dict[str, int]     # everyone reached, including ego at 0
    topic_degree: dict[str, int]    # topic -> degree of the closest member you reached it through
    unreachable: list[str]          # people in other components

    @property
    def exhausted_after(self) -> int:
        """Hops it took until nothing new could be reached (this user's eccentricity)."""
        return len(self.hops)


class TopicGraph:
    def __init__(self, memberships: list[Membership]):
        self.memberships = list(memberships)
        self._by_topic: dict[str, list[Membership]] = defaultdict(list)
        self._by_user: dict[str, list[Membership]] = defaultdict(list)
        for m in self.memberships:
            self._by_topic[m.topic].append(m)
            self._by_user[m.user].append(m)
        for ms in self._by_topic.values():
            ms.sort(key=lambda m: m.joined_at)

    def users(self) -> list[str]:
        return sorted(self._by_user)

    def topics(self) -> list[str]:
        return sorted(self._by_topic)

    def topics_of(self, user: str) -> list[str]:
        return sorted(m.topic for m in self._by_user[user])

    def members_of(self, topic: str) -> list[str]:
        """Members in arrival order."""
        return [m.user for m in self._by_topic[topic]]

    def membership(self, user: str, topic: str) -> Membership:
        return next(m for m in self._by_topic[topic] if m.user == user)

    def moderator(self, topic: str) -> str:
        """First to arrive on a topic."""
        return self._by_topic[topic][0].user

    def moderated_by(self, user: str) -> list[str]:
        return [t for t in self.topics() if self.moderator(t) == user]

    def explore(self, ego: str) -> Exploration:
        user_degree = {ego: 0}
        topic_degree: dict[str, int] = {}
        frontier = [ego]
        hops: list[Hop] = []
        while frontier:
            degree = user_degree[frontier[0]] + 1
            via, found = [], []
            for user in frontier:
                for topic in self.topics_of(user):
                    if topic in topic_degree:
                        continue
                    topic_degree[topic] = degree - 1
                    via.append(topic)
                    for member in self.members_of(topic):
                        if member not in user_degree:
                            user_degree[member] = degree
                            found.append(member)
            if not found:
                break
            hops.append(Hop(degree=degree, via_topics=sorted(via), new_users=sorted(found)))
            frontier = sorted(found)
        return Exploration(
            ego=ego, hops=hops, user_degree=user_degree, topic_degree=topic_degree,
            unreachable=[u for u in self.users() if u not in user_degree],
        )

    def diameter(self) -> int:
        """Longest exhaustion depth over all users (the most hops anyone ever needs)."""
        return max(self.explore(u).exhausted_after for u in self.users())


# ---- transparency ----------------------------------------------------------

def topic_pseudonym(user: str, topic: str, salt: str = "session") -> str:
    """
    A handle that is stable within one topic but different across topics,
    so seeing someone on two shared topics doesn't tell you it's the same
    person. Writing style can still link them; this only stops the UI
    from doing it for you.
    """
    digest = hashlib.sha256(f"{salt}|{topic}|{user}".encode()).hexdigest()[:6]
    return f"{topic[:1]}·{digest}"


@dataclass
class HopView:
    degree: int
    topics_shown: list[str]              # topic names the UI may print
    topics_suppressed: int               # topics hidden because too few people are on them
    people: list[str] = field(default_factory=list)  # pseudonyms (degree 1 only)
    people_count: int = 0                # how many people this hop reached


def visible_view(graph: TopicGraph, exp: Exploration, k: int = 3) -> list[HopView]:
    """
    What the ego's UI may show, hop by hop:

      degree 1   per-topic pseudonyms of the people you share a topic with
                 (you're in the same room; their comments are already visible)
      degree 2+  counts only, never handles

    Beyond your own topics (degree 1), a topic name is shown only if at
    least `k` people are on it. A topic with fewer members can single
    someone out: "1 person on #bonsai, reached through someone you know"
    is close to naming them.
    """
    views = []
    for hop in exp.hops:
        shown = [t for t in hop.via_topics if hop.degree == 1 or len(graph.members_of(t)) >= k]
        people = []
        if hop.degree == 1:
            for topic in hop.via_topics:
                people += [topic_pseudonym(u, topic) for u in graph.members_of(topic) if u in hop.new_users]
        views.append(HopView(
            degree=hop.degree, topics_shown=shown,
            topics_suppressed=len(hop.via_topics) - len(shown),
            people=sorted(people), people_count=len(hop.new_users),
        ))
    return views


def reidentification_risk(graph: TopicGraph, exp: Exploration) -> dict[str, int]:
    """
    For each person reached, the size of their anonymity set: how many
    reached people share the same (degree, topic set) signature. If
    the ego's UI could ever be pushed into printing someone's degree and
    topics together, everyone with a set of size 1 would be identified.
    """
    sig = {u: (d, tuple(graph.topics_of(u))) for u, d in exp.user_degree.items() if u != exp.ego}
    counts: dict[tuple, int] = defaultdict(int)
    for s in sig.values():
        counts[s] += 1
    return {u: counts[s] for u, s in sig.items()}
