"""
Shapley-value contribution valuation over a CapturePool.

This is the missing piece for real co-ownership: a mechanism to answer
"who contributed how much value" rather than just "who contributed
what." Follows Ghorbani & Zou's Data Shapley approach — treat each
contributor as a player in a cooperative game, where the payoff of a
coalition (subset of contributors) is some utility function over their
combined data, and each contributor's value is their average marginal
contribution across all possible orderings.

Exact Shapley computation is O(2^N) in the number of contributors —
infeasible past ~15-20 contributors. We use the standard fix: Monte
Carlo permutation sampling, which converges to the true Shapley value
as the number of sampled permutations grows.

Utility function used here: topic-cluster coverage diversity (reusing
core/coverage.py's keyword clustering) — a contributor who introduces
prompts covering topics nobody else touched adds more marginal value
than one who duplicates existing coverage. Swap in a different utility
function (e.g. eval score improvement from training on the subset) if
you have a downstream task to measure against instead.
"""

from __future__ import annotations

import random
from collections import defaultdict

from .pool import CapturePool
from .coverage import default_keyword_cluster


def coverage_utility(prompts: list[str], vocab: list[str]) -> float:
    """
    Utility of a set of prompts = number of distinct topic clusters it
    covers (out of the given vocab), as a fraction of vocab size.
    This rewards DIVERSITY of coverage, not raw volume — 100 prompts
    all about "billing" score the same as 1 prompt about "billing".
    """
    if not prompts or not vocab:
        return 0.0
    clusters_hit = {default_keyword_cluster(p, vocab) for p in prompts}
    clusters_hit.discard("uncategorized")
    return len(clusters_hit) / len(vocab)


def shapley_values(
    pool: CapturePool,
    vocab: list[str],
    n_permutations: int = 500,
    seed: int = 0,
    utility_fn=coverage_utility,
) -> dict[str, float]:
    """
    Returns {contributor_pubkey: shapley_value} — each value is that
    contributor's average marginal contribution to coverage_utility
    across n_permutations random join orderings.

    Higher n_permutations = more accurate estimate, more compute. A few
    hundred is usually enough for a handful of contributors; scale up
    for finer-grained estimates or more contributors.
    """
    rng = random.Random(seed)
    contributors = pool.contributors()

    # Cache each contributor's full prompt list once — avoids
    # re-querying the pool inside the hot permutation loop.
    prompts_by_contributor = {c: pool.all_prompts_by(c) for c in contributors}

    marginal_totals = defaultdict(float)

    for _ in range(n_permutations):
        order = contributors[:]
        rng.shuffle(order)
        for contributor, marginal in _marginals(order, prompts_by_contributor, vocab, utility_fn).items():
            marginal_totals[contributor] += marginal

    # Average marginal contribution across all sampled permutations
    # IS the Monte Carlo estimate of the Shapley value.
    return {c: marginal_totals[c] / n_permutations for c in contributors}


def _marginals(order, prompts_by_contributor, vocab, utility_fn) -> dict[str, float]:
    accumulated_prompts: list[str] = []
    prev_utility = 0.0
    marginals = {}
    for contributor in order:
        accumulated_prompts = accumulated_prompts + prompts_by_contributor[contributor]
        new_utility = utility_fn(accumulated_prompts, vocab)
        # Marginal contribution = utility gained by this contributor
        # joining the coalition that already includes everyone
        # before them in this ordering.
        marginals[contributor] = new_utility - prev_utility
        prev_utility = new_utility
    return marginals


def arrival_order_values(
    pool: CapturePool,
    vocab: list[str],
    order: list[str] | None = None,
    utility_fn=coverage_utility,
) -> dict[str, float]:
    """
    Marginal contribution in the ONE order contributors actually arrived
    (default: by batch created_at), instead of averaged over all orders.

    This is a priority rule: whoever covers a cluster first gets all of
    its credit, and a later contributor covering the same cluster gets
    none. It is what makes a topic's first arrival (its moderator) end
    up with the largest share. Shapley deliberately removes that
    advantage; compare the two on the same pool before choosing.
    """
    if order is None:
        first_seen = {}
        for c in sorted(pool.contributions, key=lambda c: c.created_at):
            first_seen.setdefault(c.contributor_pubkey, c.created_at)
        order = list(first_seen)
    prompts_by_contributor = {c: pool.all_prompts_by(c) for c in order}
    return _marginals(order, prompts_by_contributor, vocab, utility_fn)


def ownership_shares(shapley_vals: dict[str, float]) -> dict[str, float]:
    """
    Normalizes raw Shapley values into ownership percentages summing to
    1.0 — this is the number you'd actually use to split revenue,
    credit, or governance weight in a co-owned pool.
    """
    total = sum(shapley_vals.values())
    if total <= 0:
        # No contributor added measurable marginal value (e.g. everyone
        # covered identical topics) — fall back to equal split rather
        # than dividing by zero.
        n = len(shapley_vals)
        return {c: 1.0 / n for c in shapley_vals} if n else {}
    return {c: v / total for c, v in shapley_vals.items()}
