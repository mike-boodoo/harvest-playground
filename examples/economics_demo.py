"""
Run: python examples/economics_demo.py
Produces economics_demo.png with three panels.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from core.economics import cost_gap_collapse, substitute_pricing_pressure, simulate_investment_dynamic


def main():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # --- Panel 1: cost gap collapse ---
    teacher_cost = 100_000_000  # $100M pretrain run
    ratios = np.logspace(-4, 0, 200)
    result = cost_gap_collapse(teacher_cost, ratios)

    ax = axes[0]
    ax.plot(ratios, result["retention"] * 100, color="#2563eb")
    ax.set_xscale("log")
    ax.set_xlabel("Student compute / teacher compute")
    ax.set_ylabel("Capability retained (%)")
    ax.set_title("Cost-gap collapse\n(retention vs. relative spend)")
    ax.axhline(90, color="gray", linestyle="--", linewidth=0.8)
    ax.grid(alpha=0.3)

    # --- Panel 2: pricing pressure ---
    original_price = 20.0  # $/unit (e.g. per-1M-token price)
    substitute_price = 2.0
    pressure = substitute_pricing_pressure(original_price, result["retention"], substitute_price)

    ax = axes[1]
    ax.plot(ratios, pressure, color="#dc2626", label="Max sustainable price")
    ax.axhline(original_price, color="gray", linestyle="--", linewidth=0.8, label="Original list price")
    ax.set_xscale("log")
    ax.set_xlabel("Student compute / teacher compute")
    ax.set_ylabel("Price ($/unit)")
    ax.set_title("Pricing power decay\nas substitute quality rises")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- Panel 3: investment dynamic under two decay rates ---
    ax = axes[2]
    for decay, label, color in [(0.05, "slow erosion (5%/round)", "#16a34a"),
                                  (0.25, "fast erosion (25%/round)", "#dc2626")]:
        rounds = simulate_investment_dynamic(
            initial_investment=100_000_000, n_rounds=12, capture_decay_rate=decay,
        )
        ax.plot([r.round_num for r in rounds], [r.investment for r in rounds],
                 label=label, color=color, marker="o", markersize=3)

    ax.set_xlabel("Investment round")
    ax.set_ylabel("Investment ($)")
    ax.set_yscale("log")
    ax.set_title("Frontier investment trajectory\nunder value-capture erosion")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path = Path(__file__).resolve().parent / "economics_demo.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

    # print a few numbers too
    print("\n--- Cost gap collapse ---")
    for r in [0.001, 0.01, 0.05, 0.1]:
        idx = (np.abs(ratios - r)).argmin()
        print(f"  compute_ratio={r:.3f}  cost=${result['student_cost'][idx]:,.0f}  "
              f"retention={result['retention'][idx]*100:.1f}%")

    print("\n--- Investment dynamic (fast erosion) ---")
    rounds = simulate_investment_dynamic(100_000_000, 12, capture_decay_rate=0.25)
    for r in rounds[::3]:
        print(f"  round {r.round_num:2d}: investment=${r.investment:,.0f}  "
              f"capture_share={r.captured_value_share:.1%}")


if __name__ == "__main__":
    main()
