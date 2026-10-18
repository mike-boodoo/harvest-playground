"""
Blind side-by-side arena between two models you run yourself.

Both contenders go through the toolkit's open-weight adapters, so the
"flagship" side of a match is whichever open model you choose to stand in
for one (e.g. a large self-hosted model behind vLLM). Comparing against
closed flagship models is done in the Benchmark tab via their published
scores instead, which keeps this tool inside the open-weight-only
constraint documented in core/target.py.

Votes are appended to a JSONL log; standings (win rate, Elo) are always
recomputed from the full log, so the log is the only state.
"""

from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

VOTES = ("left", "right", "tie", "both_bad")


@dataclass
class Match:
    prompt: str
    topic: str
    left_model: str
    right_model: str
    left_response: str
    right_response: str


@dataclass
class Vote:
    prompt: str
    topic: str
    model_a: str
    model_b: str
    winner: str | None  # model label, or None for tie / both bad
    outcome: str        # one of VOTES
    ts: float


def make_match(prompt: str, topic: str, contenders: dict, rng: random.Random | None = None,
               max_new_tokens: int = 256) -> Match:
    """
    contenders: {label: Target} with exactly two entries. Sides are shuffled
    so the voter can't learn which side is which model.
    """
    if len(contenders) != 2:
        raise ValueError("an arena match needs exactly two models")
    if not prompt.strip():
        raise ValueError("enter a prompt")
    labels = list(contenders)
    (rng or random).shuffle(labels)
    responses = []
    for label in labels:
        target = contenders[label]
        key = "max_new_tokens" if hasattr(target, "max_new_tokens") else "max_tokens"
        responses.append(target.generate(prompt, **{key: max_new_tokens}).response)
    return Match(prompt, topic or "(none)", labels[0], labels[1], responses[0], responses[1])


def record_vote(match: Match, outcome: str, log_path: str | Path) -> Vote:
    if outcome not in VOTES:
        raise ValueError(f"vote must be one of {VOTES}")
    winner = {"left": match.left_model, "right": match.right_model}.get(outcome)
    vote = Vote(match.prompt, match.topic, match.left_model, match.right_model, winner, outcome, time.time())
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps(asdict(vote)) + "\n")
    return vote


def load_votes(log_path: str | Path) -> list[Vote]:
    path = Path(log_path)
    if not path.exists():
        return []
    return [Vote(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def elo(votes: list[Vote], k: float = 32.0, start: float = 1000.0) -> dict[str, float]:
    """Sequential Elo over the vote log. Ties count as a half win each; 'both bad' is skipped."""
    ratings: dict[str, float] = defaultdict(lambda: start)
    for v in votes:
        if v.outcome == "both_bad":
            continue
        ra, rb = ratings[v.model_a], ratings[v.model_b]
        expected_a = 1 / (1 + 10 ** ((rb - ra) / 400))
        score_a = 0.5 if v.winner is None else float(v.winner == v.model_a)
        ratings[v.model_a] = ra + k * (score_a - expected_a)
        ratings[v.model_b] = rb + k * ((1 - score_a) - (1 - expected_a))
    return dict(ratings)


def standings(votes: list[Vote]) -> list[dict]:
    """Per model: matches, wins, ties, both-bad, win rate (ties as half), Elo."""
    stats = defaultdict(lambda: {"matches": 0, "wins": 0, "ties": 0, "both_bad": 0})
    for v in votes:
        for m in (v.model_a, v.model_b):
            s = stats[m]
            s["matches"] += 1
            s["wins"] += v.winner == m
            s["ties"] += v.outcome == "tie"
            s["both_bad"] += v.outcome == "both_bad"
    ratings = elo(votes)
    rows = []
    for model, s in stats.items():
        rows.append({
            "model": model, **s,
            "win_rate": round(100 * (s["wins"] + 0.5 * s["ties"]) / s["matches"], 1),
            "elo": round(ratings.get(model, 1000.0)),
        })
    return sorted(rows, key=lambda r: -r["elo"])


def win_rate_by_topic(votes: list[Vote], model: str) -> list[dict]:
    """Where a model wins and loses: its win rate (ties as half) per topic."""
    per_topic = defaultdict(lambda: [0.0, 0])
    for v in votes:
        if model not in (v.model_a, v.model_b):
            continue
        cell = per_topic[v.topic]
        cell[0] += 1.0 if v.winner == model else 0.5 if v.outcome == "tie" else 0.0
        cell[1] += 1
    return sorted(({"topic": t, "matches": n, "win_rate": round(100 * w / n, 1)}
                   for t, (w, n) in per_topic.items()), key=lambda r: r["win_rate"])
