"""
One user's session, end to end: join topics, explore outward through
topic overlap until nothing new can be reached, then leave with a model
learned from every topic agent in reach.

Three quantities decide what each user takes away, and they come from
three different places:

  access     how far away each topic is (core/topic_graph.py). Exhaustive
             exploration gives everyone in a component the same topics,
             so distance decay is the only access difference.
  stake      the user's ownership share of each topic agent
             (core/valuation.py). Under the arrival-order rule the
             moderator (first arrival) gets the largest share; under
             Shapley, order doesn't matter.
  quality    how good each topic agent is: members' models mixed by the
             trust their peers give them (core/peer_benchmark.py), so the
             benchmark that orders the models is derived from the models
             themselves, not from a service.

Models are simulated: each user's model has a hidden skill per topic.
Batches are really signed and pooled, with a separate key per user per
topic so the pool can't link one person across topics by pubkey.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from .peer_benchmark import eigentrust, peer_judgments
from .pool import CapturePool
from .provenance import ProvenanceSigner
from .target import GenerationResult, Target
from .topic_graph import (
    Exploration, HopView, Membership, TopicGraph, reidentification_risk, visible_view,
)
from .valuation import arrival_order_values, ownership_shares, shapley_values

BASE_SKILL = 0.25   # a model's skill on a topic its owner never joined


def demo_community() -> tuple[list[Membership], dict[str, dict[str, float]], frozenset[str]]:
    """
    Twelve people on seven topics: a five-topic component three hops
    wide, a separate two-person #bonsai component, and two colluders on
    #vector-search. Moderators are not always the most skilled member.
    Returns (memberships, true skill per user per topic, colluders).
    """
    rows = [
        # user, topic, joined_at, subthemes covered, true skill
        ("ana", "quant-volatility", 1, ("measurement", "regimes", "data"), 0.70),
        ("skewtrader", "quant-volatility", 3, ("regimes", "hedging"), 0.80),
        ("ben", "quant-volatility", 5, ("measurement",), 0.55),
        ("skewtrader", "agentic-orchestration", 2, ("aggregation", "verification", "interop"), 0.75),
        ("cleo", "agentic-orchestration", 4, ("verification", "routing"), 0.85),
        ("dev", "agentic-orchestration", 6, ("aggregation",), 0.50),
        ("cleo", "vector-search", 1.5, ("indexing", "recall", "hybrid"), 0.80),
        ("eli", "vector-search", 7, ("latency", "recall"), 0.70),
        ("sy1", "vector-search", 12, ("indexing",), 0.30),
        ("sy2", "vector-search", 13, ("indexing",), 0.30),
        ("eli", "rust-async", 8, ("runtimes", "cancellation"), 0.85),
        ("fay", "rust-async", 9, ("pinning", "runtimes"), 0.70),
        ("gus", "rust-async", 14, ("cancellation",), 0.45),
        ("ana", "options-tax", 10, ("washsales", "straddles"), 0.60),
        ("gus", "options-tax", 11, ("reporting",), 0.80),
        ("ben", "options-tax", 15, ("straddles",), 0.50),
        ("hana", "bonsai", 2, ("soil", "pruning"), 0.75),
        ("ivo", "bonsai", 3, ("wiring",), 0.60),
    ]
    memberships = [Membership(u, t, at, subs) for u, t, at, subs, _ in rows]
    users = sorted({r[0] for r in rows})
    topics = sorted({r[1] for r in rows})
    skill = {u: {t: BASE_SKILL for t in topics} for u in users}
    for u, t, _, _, s in rows:
        skill[u][t] = s
    return memberships, skill, frozenset({"sy1", "sy2"})


# ---- pools and ownership ---------------------------------------------------

class _SimulatedModel(Target):
    license_tag = "apache-2.0"

    def __init__(self, user: str):
        self.model_id = f"{user}-model"

    def generate(self, prompt, **kwargs):
        return GenerationResult(prompt=prompt, response=f"[{self.model_id}] {prompt}",
                                logprobs=None, latency_s=0.0)


def topic_signer(user: str, topic: str, secret: str = "demo") -> ProvenanceSigner:
    """A per-topic key, so pubkeys in different topic pools don't link the same person."""
    return ProvenanceSigner.from_hex(hashlib.sha256(f"{secret}|{user}|{topic}".encode()).hexdigest())


@dataclass
class TopicPool:
    topic: str
    pool: CapturePool
    vocab: list[str]
    pubkey_of: dict[str, str]     # user -> this topic's pubkey (held privately by the simulation)


def build_topic_pools(graph: TopicGraph) -> dict[str, TopicPool]:
    """Each member signs a batch of prompts over the subthemes they cover; batches are verified into the topic's pool."""
    pools = {}
    for topic in graph.topics():
        pool, pubkey_of, vocab = CapturePool(), {}, set()
        for user in graph.members_of(topic):
            m = graph.membership(user, topic)
            vocab.update(m.subthemes)
            prompts = [f"[{topic}] a question about {s}" for s in m.subthemes]
            signer = topic_signer(user, topic)
            model = _SimulatedModel(user)
            record = signer.sign_batch(model.generate_batch(prompts), model_id=model.model_id,
                                       license_tag=model.license_tag, strategy_name="topic-session")
            pool.add_batch(record)
            pubkey_of[user] = signer.export_pubkey_hex()
        pools[topic] = TopicPool(topic, pool, sorted(vocab), pubkey_of)
    return pools


def topic_shares(graph: TopicGraph, pools: dict[str, TopicPool], rule: str) -> dict[str, dict[str, float]]:
    """Ownership of each topic agent per user, under 'arrival' (priority) or 'shapley' (order-free)."""
    shares = {}
    for topic, tp in pools.items():
        if rule == "arrival":
            order = [tp.pubkey_of[u] for u in graph.members_of(topic)]
            raw = arrival_order_values(tp.pool, tp.vocab, order=order)
        elif rule == "shapley":
            raw = shapley_values(tp.pool, tp.vocab, n_permutations=400)
        else:
            raise ValueError(f"unknown rule {rule!r}")
        by_pubkey = ownership_shares(raw)
        shares[topic] = {u: by_pubkey[pk] for u, pk in tp.pubkey_of.items()}
    return shares


# ---- topic agents and takeaways --------------------------------------------

def topic_agent_quality(graph: TopicGraph, skill: dict, weight: dict) -> dict[str, float]:
    """
    Each topic agent learns from all its members' models, mixed by `weight`:
    either one score per user (peer-derived trust) or one per topic per user
    (ownership shares, for comparison).
    """
    quality = {}
    for topic in graph.topics():
        members = graph.members_of(topic)
        w = weight[topic] if isinstance(weight.get(topic), dict) else weight
        weights = {u: w.get(u, 0.0) for u in members}
        total = sum(weights.values())
        if not total:
            weights, total = {u: 1.0 for u in members}, len(members)
        quality[topic] = sum(weights[u] * skill[u][topic] for u in members) / total
    return quality


@dataclass
class Takeaway:
    user: str
    per_topic: dict[str, float]     # quality of the user's take-away model on each topic
    coverage: float                 # mean over every topic in the community
    stake: dict[str, float]         # rule -> sum of ownership shares across topic agents
    moderates: list[str]


def takeaway(graph: TopicGraph, exp: Exploration, skill: dict, agent_quality: dict,
             shares_by_rule: dict[str, dict[str, dict[str, float]]], decay: float = 0.85) -> Takeaway:
    """
    The user's model learns from every topic agent it reached, discounted by
    `decay` per degree of separation, and never gets worse than it started.
    """
    user = exp.ego
    per_topic = {}
    for topic in graph.topics():
        own = skill[user][topic]
        if topic in exp.topic_degree:
            per_topic[topic] = max(own, agent_quality[topic] * decay ** exp.topic_degree[topic])
        else:
            per_topic[topic] = own
    stake = {rule: sum(s.get(user, 0.0) for s in shares.values()) for rule, shares in shares_by_rule.items()}
    return Takeaway(user, per_topic, sum(per_topic.values()) / len(per_topic), stake, graph.moderated_by(user))


@dataclass
class SessionReport:
    exploration: Exploration
    views: list[HopView]
    smallest_anonymity_set: int | None
    takeaway: Takeaway


@dataclass
class Community:
    graph: TopicGraph
    skill: dict
    colluders: frozenset[str]
    pools: dict[str, TopicPool]
    shares: dict[str, dict[str, dict[str, float]]]
    trust: dict[str, float]
    agent_quality: dict[str, float]


def build_community(seed: int = 0) -> Community:
    memberships, skill, colluders = demo_community()
    graph = TopicGraph(memberships)
    pools = build_topic_pools(graph)
    shares = {rule: topic_shares(graph, pools, rule) for rule in ("arrival", "shapley")}
    judgments = peer_judgments(graph, skill, random.Random(seed), colluders=colluders)
    moderators = sorted({graph.moderator(t) for t in graph.topics()})
    trust, _, _ = eigentrust(graph.users(), judgments, pretrusted=moderators)
    return Community(graph, skill, colluders, pools, shares, trust,
                     topic_agent_quality(graph, skill, trust))


def run_session(c: Community, ego: str, k: int = 3, decay: float = 0.85) -> SessionReport:
    exp = c.graph.explore(ego)
    risk = reidentification_risk(c.graph, exp)
    return SessionReport(
        exploration=exp,
        views=visible_view(c.graph, exp, k=k),
        smallest_anonymity_set=min(risk.values()) if risk else None,
        takeaway=takeaway(c.graph, exp, c.skill, c.agent_quality, c.shares, decay=decay),
    )
