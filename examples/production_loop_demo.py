"""
Run: python examples/production_loop_demo.py

Compares three loop policies against an all-flagship baseline over 12
months, then sweeps contributor throughput to find where the loop
ignites. Stylized assumptions from core/production_loop.py.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.production_loop import simulate, baseline_cost, LoopParams

MONTHS = 12
POLICIES = ("open_loop", "feedback_only", "full_loop")


def total(res):
    return sum(r.cost for r in res)


def first_coop_month(res):
    return next((r.month + 1 for r in res if r.clusters_on_coop > 0), None)


def main():
    base_total = baseline_cost(MONTHS)
    print(f"All-flagship baseline: ${base_total / MONTHS:,.0f}/mo, ${base_total:,.0f} over {MONTHS} mo\n")

    print("=== Throughput sweep: contributed examples per month (whole pool) ===")
    print(f"{'ex/mo':>8} | " + " | ".join(f"{pol:>31}" for pol in POLICIES))
    for ex in [8_000, 20_000, 40_000, 80_000, 150_000, 300_000]:
        p = LoopParams(contributor_examples_per_month=ex)
        cells = []
        for pol in POLICIES:
            res = simulate(pol, MONTHS, p)
            t = total(res)
            fm = first_coop_month(res)
            cells.append(f"{(t - base_total) / base_total:+6.1%}  first co-op mo={fm if fm else '-'}")
        print(f"{ex:>8,} | " + " | ".join(f"{c:>31}" for c in cells))

    print("\n=== Month by month at 150,000 ex/mo (full loop) ===")
    p = LoopParams(contributor_examples_per_month=150_000)
    for r in simulate("full_loop", MONTHS, p):
        print(f"  month {r.month + 1:>2}: ${r.cost:>8,.0f}  clusters on co-op={r.clusters_on_coop}/6  "
              f"resolved={r.resolved_rate:.3f}")

    # --- checks: non-vacuous, at a throughput where the loop ignites ---
    low = {pol: total(simulate(pol, MONTHS, LoopParams(contributor_examples_per_month=8_000))) for pol in POLICIES}
    assert len({round(v) for v in low.values()}) == 1 and low["full_loop"] > base_total, \
        "at 8k ex/mo nothing ignites; overhead makes every policy worse than the baseline"

    high = {pol: total(simulate(pol, MONTHS, p)) for pol in POLICIES}
    assert high["full_loop"] < high["feedback_only"] < high["open_loop"] < base_total, \
        "at 150k ex/mo each loop component must strictly lower cost"
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
