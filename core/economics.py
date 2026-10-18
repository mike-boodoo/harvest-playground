"""
Economic models for the effects discussed re: harvesting from open
models — cost-gap collapse via distillation, pricing power decay as
clone quality approaches the original, and the free-rider dynamic in
frontier investment incentives.

These are stylized simulations, not calibrated forecasts. Useful for
reasoning about qualitative shape (does the curve bend, does the
equilibrium collapse) rather than precise numbers.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------------------
# 1. Cost-gap collapse: pretraining cost vs. distillation cost to reach
#    a given fraction of teacher capability.
# ---------------------------------------------------------------------

def capability_retention(compute_ratio: np.ndarray, k: float = 8.0, midpoint: float = 0.05) -> np.ndarray:
    """
    Fraction of teacher-model capability a distilled student retains,
    as a function of compute_ratio = student_compute / teacher_compute.

    Modeled as a logistic saturation curve: retention rises steeply at
    low compute ratios (most of the "easy" behavior is cheap to
    capture) then flattens well below 1.0 (the last few points of
    capability are disproportionately expensive to reach via
    distillation alone — this is empirically what's observed with
    real distillation runs).

    k: steepness of the logistic transition. Higher k = sharper knee
       (retention snaps from ~0% to ~100% over a narrower compute-ratio
       range). Lower k = a more gradual, smeared-out transition.
    midpoint: the compute_ratio at which retention crosses 50%. This is
       the single most important knob — it's where the "knee" of the
       curve sits. 0.05 means half of teacher capability is reached at
       just 5% of teacher compute; tune this against real distillation
       benchmarks if you want calibrated (not just qualitative) output.
    """
    # Work in log10 space because compute ratios span many orders of
    # magnitude (0.0001x to 1x) — a logistic curve on raw linear ratios
    # would look like a step function jammed against zero.
    # np.clip guards against log10(0) when compute_ratio includes 0.
    x = np.log10(np.clip(compute_ratio, 1e-6, None))
    m = np.log10(midpoint)

    # Standard logistic function shifted so it's centered on `midpoint`
    # (in log space) and scaled by steepness `k`. Output range is (0, 1).
    return 1 / (1 + np.exp(-k * (x - m)))


def cost_gap_collapse(teacher_pretrain_cost: float, compute_ratios: np.ndarray) -> dict:
    """
    Returns student cost and retained capability across a sweep of
    compute ratios, illustrating how little it costs to reach "close
    enough" capability relative to the teacher's pretraining spend.
    """
    # Dollar cost of the student run scales linearly with the compute
    # ratio applied to the teacher's known pretraining spend — a
    # simplifying assumption (real compute-to-dollar conversion varies
    # by hardware/efficiency), but fine for showing relative shape.
    student_cost = teacher_pretrain_cost * compute_ratios

    retention = capability_retention(compute_ratios)

    return {
        "compute_ratio": compute_ratios,
        "student_cost": student_cost,
        "retention": retention,
        # "Bang for buck" metric: dollars spent per percentage point of
        # capability retained. This number gets huge at very low
        # retention (dividing by near-zero) and shrinks fast once the
        # curve's knee is reached — showing the same threshold effect
        # from a cost-efficiency angle instead of a raw-cost angle.
        "cost_per_retention_point": student_cost / np.clip(retention, 1e-6, None),
    }


# ---------------------------------------------------------------------
# 2. Pricing power decay: willingness-to-pay for the original as a
#    cheap substitute's quality approaches it.
# ---------------------------------------------------------------------

def substitute_pricing_pressure(
    original_price: float, retention: np.ndarray, substitute_price: float,
    switching_cost: float = 0.1,
) -> np.ndarray:
    """
    Simple substitution-pressure model: the max sustainable price for
    the original is bounded by the substitute's price plus the
    remaining quality gap's dollar value, minus a switching-cost buffer
    that lets the original retain some premium even at high retention
    (integration cost, brand trust, support SLAs, etc).

    max_price = substitute_price + (1 - retention) * original_price + switching_cost * original_price
    """
    # The remaining quality gap (1 - retention) is worth some fraction
    # of the original's price — a buyer will pay extra for the original
    # only up to the dollar value of the capability the substitute is
    # still missing.
    quality_gap_value = (1 - retention) * original_price

    # Total ceiling = what the substitute costs, plus what the residual
    # quality gap is worth, plus a fixed switching-cost buffer (covers
    # integration effort, brand trust, support SLAs — non-quality
    # reasons someone stays on the original even at 100% retention).
    return substitute_price + quality_gap_value + switching_cost * original_price


# ---------------------------------------------------------------------
# 3. Free-rider investment dynamic: does frontier investment shrink
#    over successive rounds if captured value share decays?
# ---------------------------------------------------------------------

@dataclass
class InvestmentRound:
    round_num: int
    investment: float
    captured_value_share: float
    total_value_created: float
    captured_value: float


def simulate_investment_dynamic(
    initial_investment: float,
    n_rounds: int,
    capture_decay_rate: float = 0.15,
    value_multiplier: float = 4.0,
    reinvestment_fraction: float = 0.6,
    min_capture_share: float = 0.05,
) -> list[InvestmentRound]:
    """
    Simulates a lab that invests in a new frontier model each round,
    creates `value_multiplier`x the investment in total ecosystem
    value, but captures a shrinking share of that value each round
    (harvesting/distillation erodes the premium faster each cycle).
    Reinvests `reinvestment_fraction` of what it actually captures.

    This is a toy model for exploring: does investment stabilize at
    some floor, or does it spiral toward zero? Depends on whether
    capture_share decays faster than reinvestment_fraction compounds.
    """
    rounds = []
    investment = initial_investment
    capture_share = 1.0  # round 0 starts at full capture, before any erosion

    for i in range(n_rounds):
        # Erode capture share geometrically each round — this represents
        # harvesting/distillation catching up faster with each successive
        # model generation (competitors have more practice, more tooling,
        # and a known target to approximate). Floored at min_capture_share
        # so the model doesn't erode to literally zero.
        capture_share = max(min_capture_share, capture_share * (1 - capture_decay_rate))

        # Total ecosystem value created by this round's model (captured
        # + given away to distillers/downstream users/consumer surplus).
        total_value = investment * value_multiplier

        # Only this slice of total value actually flows back to the lab
        # that funded the round.
        captured_value = total_value * capture_share

        rounds.append(InvestmentRound(
            round_num=i,
            investment=investment,
            captured_value_share=capture_share,
            total_value_created=total_value,
            captured_value=captured_value,
        ))

        # Next round's investment is a fraction of what was actually
        # captured this round — NOT a fraction of total value created.
        # This is the mechanism that produces the overshoot-then-collapse
        # shape: early on, captured_value can still exceed investment
        # (funding growth) even while capture_share is already declining;
        # eventually decay outpaces the multiplier and investment craters.
        investment = captured_value * reinvestment_fraction

    return rounds


# ---------------------------------------------------------------------
# 4. Co-owned model break-even: when does a pooled, self-hosted model
#    beat paying per-token for a flagship API?
#
#    Every default below is an ASSUMPTION chosen to make the shape
#    visible, not a measured figure. Replace them with your own quotes
#    (GPU hosting, API price sheet, contributor counts) before reading
#    any dollar number as a forecast.
# ---------------------------------------------------------------------

def pooled_performance(
    pooled_examples: float,
    base_perf: float = 0.60,
    ceiling_perf: float = 0.85,
    saturation_scale: float = 50_000.0,
) -> float:
    """
    Task performance (0-1) of the co-owned model as a function of how
    many pooled training examples it has absorbed.

    Exponential saturation: each additional example adds less than the
    last (the diminishing-returns point from Data Shapley — duplicate
    data adds little). It starts at the open base model's own
    performance and approaches a ceiling below flagship quality,
    because pooling data does not add parameters or pretraining scale.
    """
    return base_perf + (ceiling_perf - base_perf) * (1 - np.exp(-pooled_examples / saturation_scale))


def coordination_cost(
    n_contributors: int,
    cost_per_link: float = 40.0,
    exponent: float = 1.3,
) -> float:
    """
    Monthly overhead of keeping N contributors aligned: verification,
    merge review, dispute handling, payout logistics.

    Scales superlinearly in N (exponent > 1) because the number of
    pairwise relationships grows faster than headcount. This is the
    coordination bucket from the Ostrom/commons-governance and
    blockchain-AI literature; the exponent is an assumption.
    """
    return cost_per_link * (n_contributors ** exponent)


def valuation_cost(
    n_contributors: int,
    n_permutations: int = 500,
    cost_per_utility_eval: float = 0.02,
    refreshes_per_month: float = 1.0,
) -> float:
    """
    Monthly cost of recomputing Shapley ownership shares.

    Monte Carlo Shapley needs n_permutations * n_contributors utility
    evaluations per refresh. With the cheap coverage utility used in
    valuation.py the per-eval cost is tiny; if the utility were a real
    fine-tune and eval run, cost_per_utility_eval could be orders of
    magnitude larger, and this term could dominate.
    """
    return n_permutations * n_contributors * cost_per_utility_eval * refreshes_per_month


def api_monthly_cost(tokens_per_month: float, price_per_mtok: float = 6.0) -> float:
    """Flagship API: cost is linear in volume, no fixed component."""
    return (tokens_per_month / 1e6) * price_per_mtok


def co_owned_monthly_cost(
    tokens_per_month: float,
    n_contributors: int,
    fixed_hosting: float = 2_500.0,
    marginal_per_mtok: float = 0.40,
    **valuation_kwargs,
) -> float:
    """
    Co-owned, self-hosted model: a large fixed component (hosting,
    coordination, valuation) plus a small marginal inference cost.
    """
    return (
        fixed_hosting
        + coordination_cost(n_contributors)
        + valuation_cost(n_contributors, **valuation_kwargs)
        + (tokens_per_month / 1e6) * marginal_per_mtok
    )


def quality_adjusted_cost(
    tokens_per_month: float,
    model_perf: float,
    flagship_perf: float = 0.92,
    gap_penalty_per_mtok: float = 30.0,
) -> float:
    """
    Dollar cost of the performance shortfall versus the flagship.

    Every million tokens served at lower quality costs the buyer
    something (rework, escalations, lost conversions). The penalty rate
    is an assumption, and it is the single most sensitive input here:
    it decides whether a cheaper-but-worse model is actually cheaper.
    """
    shortfall = max(0.0, flagship_perf - model_perf)
    return (tokens_per_month / 1e6) * shortfall * gap_penalty_per_mtok


def break_even_volume(
    n_contributors: int,
    pooled_examples: float,
    quality_adjust: bool = True,
    max_tokens: float = 5e10,
    **kwargs,
) -> float | None:
    """
    Smallest monthly token volume at which the co-owned model costs
    less than the flagship API, or None if it never does within
    max_tokens. Found by bisection on (co-owned cost - API cost),
    which is monotone because the co-owned slope (marginal inference
    plus quality penalty) is below the API slope in the sweeps here.
    """
    perf = pooled_performance(pooled_examples)

    def gap(tokens: float) -> float:
        co = co_owned_monthly_cost(tokens, n_contributors, **kwargs)
        if quality_adjust:
            co += quality_adjusted_cost(tokens, perf)
        return co - api_monthly_cost(tokens)

    if gap(max_tokens) > 0:
        return None
    lo, hi = 0.0, max_tokens
    for _ in range(100):
        mid = (lo + hi) / 2
        if gap(mid) > 0:
            lo = mid
        else:
            hi = mid
    return hi
