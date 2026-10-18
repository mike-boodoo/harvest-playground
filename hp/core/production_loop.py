"""
Closed production loop: how a co-owned model gets cheaper through use.

The loop, one cycle per month:

  1. SERVE    Each topic cluster is routed to whichever is cheaper in
              expectation: the co-owned model or a paid flagship API.
  2. METER    Unresolved inputs per cluster are the coverage gaps
              (core/coverage.py's job in a real deployment).
  3. LABEL    A fraction of human-resolved escalations become labeled
              training examples. These come from YOUR OWN traffic and
              YOUR OWN staff. Flagship API outputs are never used as
              training data: they are a paid service, not a data source.
  4. CONTRIBUTE  Contributors add examples (own data, or captures from
              open-weight models under their licenses) and aim them
              using the coverage gaps from step 2.
  5. POOL + FINE-TUNE  Examples land in a cluster's pool (signed and
              valued as in pool.py / valuation.py); performance on that
              cluster rises on a diminishing-returns curve.
  6. REPEAT   Higher coverage flips clusters from API to co-owned
              routing, which lowers cost, which funds the next cycle.

Three policies isolate what each piece of the loop is worth:
  open_loop      uniform contributor effort, no production labels
  feedback_only  uniform contributor effort + production labels
  full_loop      gap-targeted contributor effort + production labels

Every parameter is an assumption. The output is a shape, not a forecast.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .economics import (
    pooled_performance, coordination_cost, valuation_cost,
)


@dataclass
class LoopParams:
    requests_per_month: float = 1_000_000
    tokens_per_request: float = 1_500
    api_price_per_mtok: float = 6.0
    coop_marginal_per_mtok: float = 0.40
    fixed_hosting: float = 2_500.0
    human_cost_per_escalation: float = 0.15
    flagship_resolution: float = 0.92
    n_contributors: int = 10
    contributor_examples_per_month: float = 8_000
    label_rate: float = 0.05          # share of human escalations kept as training data
    base_perf: float = 0.55
    ceiling_perf: float = 0.93
    shares: tuple = (0.30, 0.22, 0.18, 0.14, 0.10, 0.06)
    scales: tuple = (20_000, 30_000, 40_000, 60_000, 90_000, 150_000)  # harder/rarer = more data


@dataclass
class MonthResult:
    month: int
    cost: float
    resolved_rate: float
    clusters_on_coop: int
    perf: list = field(default_factory=list)


def _per_request_costs(p: LoopParams):
    api = p.tokens_per_request / 1e6 * p.api_price_per_mtok
    coop = p.tokens_per_request / 1e6 * p.coop_marginal_per_mtok
    return api, coop


def simulate(policy: str, months: int = 12, p: LoopParams | None = None) -> list[MonthResult]:
    assert policy in ("open_loop", "feedback_only", "full_loop")
    p = p or LoopParams()
    shares = np.array(p.shares)
    scales = np.array(p.scales, dtype=float)
    n = np.zeros(len(shares))                      # pooled examples per cluster
    api_cost, coop_cost = _per_request_costs(p)
    h = p.human_cost_per_escalation

    # Fixed monthly overhead of being a co-owned model at all.
    overhead = (p.fixed_hosting + coordination_cost(p.n_contributors)
                + valuation_cost(p.n_contributors))

    results = []
    for m in range(months):
        r = np.array([
            pooled_performance(n[i], p.base_perf, p.ceiling_perf, scales[i])
            for i in range(len(shares))
        ])

        # 1. SERVE: route each cluster by expected cost per request.
        flagship_per_req = api_cost + (1 - p.flagship_resolution) * h
        coop_per_req = coop_cost + (1 - r) * h
        on_coop = coop_per_req < flagship_per_req
        per_req = np.where(on_coop, coop_per_req, flagship_per_req)
        cost = overhead + float(np.sum(shares * p.requests_per_month * per_req))

        resolved = np.where(on_coop, r, p.flagship_resolution)
        results.append(MonthResult(
            month=m, cost=cost,
            resolved_rate=float(np.sum(shares * resolved)),
            clusters_on_coop=int(on_coop.sum()),
            perf=[float(x) for x in r],
        ))

        # 2-3. METER + LABEL: human escalations on co-owned-routed traffic
        # become training examples for that cluster.
        if policy != "open_loop":
            escalations = shares * p.requests_per_month * (1 - r) * on_coop
            n += escalations * p.label_rate

        # 4. CONTRIBUTE: split effort uniformly, or by traffic-weighted gap.
        if policy == "full_loop":
            gap = shares * np.maximum(p.ceiling_perf - r, 1e-9)
            weights = gap / gap.sum()
        else:
            weights = np.full(len(shares), 1 / len(shares))
        n += p.contributor_examples_per_month * weights

    return results


def baseline_cost(months: int = 12, p: LoopParams | None = None) -> float:
    """Cumulative cost of sending everything to the flagship API."""
    p = p or LoopParams()
    api_cost, _ = _per_request_costs(p)
    per_req = api_cost + (1 - p.flagship_resolution) * p.human_cost_per_escalation
    return months * p.requests_per_month * per_req
