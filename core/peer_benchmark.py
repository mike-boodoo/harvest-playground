"""
Peer-derived benchmark: models are scored by each other, not by a service.

On every topic, members' models answer the same prompts in pairs and a
third member's model judges which answer is better (the arena, with a
peer as the judge instead of a person). Nobody holds an answer key, so
the judgments are only as good as the judges, and a judge's skill is
exactly what's being measured.

EigenTrust (Kamvar, Schlosser & Garcia-Molina, 2003; built for P2P
networks with no central authority) resolves that loop:

  1. Each judge i turns its verdicts into local trust c_ij in each
     model j it judged (wins minus losses, clipped at 0, normalized).
  2. Global trust t solves t = (1 - a) * C^T t + a * p: you're trusted
     if models that are trusted rate you well, recursively.
  3. p is spread over a small set of pre-trusted peers. Without it a
     colluding ring can vouch for itself; with it, trust has to flow in
     from somewhere already trusted.

Here the pre-trusted peers are topic moderators (first arrivals). That is
the second reason moderators matter: they anchor the benchmark as well as
owning the largest share of the topic agent.

Each power iteration pushes trust one judging hop further, so the number
of iterations before every reachable model has nonzero trust is a
degrees-of-separation count over the judging graph.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

from .topic_graph import TopicGraph


@dataclass
class Judgment:
    judge: str
    topic: str
    winner: str
    loser: str


def peer_judgments(
    graph: TopicGraph,
    skill: dict[str, dict[str, float]],
    rng: random.Random,
    matches_per_pair: int = 50,
    colluders: frozenset[str] = frozenset(),
) -> list[Judgment]:
    """
    Simulated peer arena. Each model's answer on a topic is good with
    probability equal to its true skill there; a judge picks the better
    answer with probability 0.5 + 0.5 * (judge's skill on the topic).
    Colluders always vote for a fellow colluder when one is in the match.

    A topic with only two members produces no judgments: there's no
    third party to judge them, which is a real limit of peer evaluation.
    """
    judgments = []
    for topic in graph.topics():
        members = graph.members_of(topic)
        for a, b in combinations(members, 2):
            judges = [j for j in members if j not in (a, b)]
            if not judges:
                continue
            for _ in range(matches_per_pair):
                judge = rng.choice(judges)
                good_a = rng.random() < skill[a][topic]
                good_b = rng.random() < skill[b][topic]
                if judge in colluders and {a, b} & colluders:
                    winner = a if a in colluders else b
                elif good_a == good_b:
                    continue
                else:
                    better = a if good_a else b
                    correct = rng.random() < 0.5 + 0.5 * skill[judge][topic]
                    winner = better if correct else (b if better == a else a)
                judgments.append(Judgment(judge, topic, winner, b if winner == a else a))
    return judgments


def eigentrust(
    users: list[str],
    judgments: list[Judgment],
    pretrusted: list[str] | None = None,
    alpha: float = 0.15,
    tol: float = 1e-10,
    max_iter: int = 500,
) -> tuple[dict[str, float], int, int]:
    """
    Returns (global trust per user, iterations to converge, iterations
    until the set of users with any trust stops growing, i.e. how many
    judging hops trust had to travel from the anchors).
    `pretrusted=None` spreads p uniformly, i.e. no anchor.
    """
    sat: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for j in judgments:
        sat[j.judge][j.winner] += 1
        sat[j.judge][j.loser] -= 1

    anchors = pretrusted or users
    p = {u: (1 / len(anchors) if u in anchors else 0.0) for u in users}
    local: dict[str, dict[str, float]] = {}
    for i in users:
        row = {j: max(v, 0.0) for j, v in sat[i].items() if v > 0}
        total = sum(row.values())
        # A judge with no positive verdicts defers to the anchors.
        local[i] = {j: v / total for j, v in row.items()} if total else dict(p)

    t = dict(p)
    reach_iter = None
    for it in range(1, max_iter + 1):
        nxt = {u: alpha * p[u] for u in users}
        for i, ti in t.items():
            if ti:
                for j, cij in local[i].items():
                    nxt[j] += (1 - alpha) * cij * ti
        if reach_iter is None and sum(v > 0 for v in nxt.values()) == sum(v > 0 for v in t.values()):
            reach_iter = it - 1
        delta = sum(abs(nxt[u] - t[u]) for u in users)
        t = nxt
        if delta < tol:
            return t, it, reach_iter if reach_iter is not None else it
    return t, max_iter, reach_iter if reach_iter is not None else max_iter


def win_rate(users: list[str], judgments: list[Judgment]) -> dict[str, float]:
    """Naive baseline: share of judged matches won, every judge counted equally."""
    wins, games = defaultdict(int), defaultdict(int)
    for j in judgments:
        wins[j.winner] += 1
        games[j.winner] += 1
        games[j.loser] += 1
    return {u: wins[u] / games[u] if games[u] else 0.0 for u in users}


def spearman(a: dict[str, float], b: dict[str, float]) -> float:
    """Rank correlation between two scorings of the same keys (average ranks on ties)."""
    keys = sorted(a)

    def ranks(d):
        order = sorted(keys, key=lambda k: d[k])
        r, i = {}, 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and d[order[j + 1]] == d[order[i]]:
                j += 1
            for k in order[i:j + 1]:
                r[k] = (i + j) / 2
            i = j + 1
        return r

    ra, rb = ranks(a), ranks(b)
    n = len(keys)
    ma, mb = sum(ra.values()) / n, sum(rb.values()) / n
    cov = sum((ra[k] - ma) * (rb[k] - mb) for k in keys)
    va = sum((ra[k] - ma) ** 2 for k in keys) ** 0.5
    vb = sum((rb[k] - mb) ** 2 for k in keys) ** 0.5
    return cov / (va * vb) if va and vb else 0.0
