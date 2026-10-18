"""
Break-even sweep for a co-owned model versus a flagship API.

Run: python examples/breakeven_demo.py

All inputs are stylized assumptions (see core/economics.py section 4).
The output shows the SHAPE of the trade-off, not a price forecast.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.economics import (
    pooled_performance, coordination_cost, valuation_cost,
    api_monthly_cost, co_owned_monthly_cost, quality_adjusted_cost,
    break_even_volume,
)


def fmt_tokens(t):
    return "never (within 50B tok/mo)" if t is None else f"{t / 1e6:,.0f}M tok/mo"


def main():
    print("=== 1. Diminishing returns on pooled data ===")
    for n in [0, 10_000, 50_000, 100_000, 250_000, 1_000_000]:
        print(f"  {n:>9,} pooled examples -> performance {pooled_performance(n):.3f}")

    print("\n=== 2. Overhead grows faster than headcount ===")
    print("  contributors  coordination/mo  valuation/mo  total overhead/mo")
    for n in [3, 10, 30, 100, 300]:
        c, v = coordination_cost(n), valuation_cost(n)
        print(f"  {n:>12}  ${c:>13,.0f}  ${v:>10,.0f}  ${c + v:>14,.0f}")

    print("\n=== 3. Break-even volume vs. flagship API ===")
    print("  (pooled examples fixed at 100,000; quality penalty applied)")
    for n in [3, 10, 30, 100, 300]:
        be = break_even_volume(n_contributors=n, pooled_examples=100_000)
        print(f"  {n:>4} contributors -> break-even at {fmt_tokens(be)}")

    print("\n=== 4. Does more pooled data move break-even? (10 contributors) ===")
    for ex in [10_000, 50_000, 100_000, 250_000, 1_000_000]:
        be = break_even_volume(n_contributors=10, pooled_examples=ex)
        print(f"  {ex:>9,} examples (perf {pooled_performance(ex):.3f}) -> {fmt_tokens(be)}")

    print("\n=== 5. Sensitivity to the quality-gap penalty (10 contributors, 100k examples) ===")
    perf = pooled_performance(100_000)
    for pen in [0, 10, 30, 60, 100]:
        def gap(tokens, pen=pen):
            return (co_owned_monthly_cost(tokens, 10)
                    + quality_adjusted_cost(tokens, perf, gap_penalty_per_mtok=pen)
                    - api_monthly_cost(tokens))
        # coarse scan then report first crossing
        crossing = None
        t = 1e6
        while t <= 5e10:
            if gap(t) <= 0:
                crossing = t
                break
            t *= 1.05
        print(f"  penalty ${pen:>3}/Mtok -> {fmt_tokens(crossing)}")

    # --- sanity checks: the model must behave the way its docstrings claim ---
    assert pooled_performance(0) < pooled_performance(1e5) <= pooled_performance(1e7) <= 0.85
    gains = [pooled_performance(x + 50_000) - pooled_performance(x) for x in (0, 100_000, 300_000)]
    assert gains[0] > gains[1] > gains[2] > 0, "returns must diminish"
    assert coordination_cost(100) / coordination_cost(10) > 10, "overhead must be superlinear"
    be_small = break_even_volume(3, 100_000)
    be_large = break_even_volume(300, 100_000)
    assert be_small is not None and (be_large is None or be_large > be_small), \
        "more contributors must raise the break-even bar"
    print("\nAll sanity checks passed.")


if __name__ == "__main__":
    main()
