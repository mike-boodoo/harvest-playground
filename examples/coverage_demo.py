"""
Demo: coverage metering over synthetic B2C traffic, including a
deliberate drift scenario (a new input cluster surges and the model
handles it poorly) to show flagged_drift() catching it.

Run: python examples/coverage_demo.py
"""

import sys
import random
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.coverage import CoverageTracker


VOCAB = ["billing", "password", "shipping", "returns", "login", "refund"]

# Response quality per cluster in "window A" (normal operation)
RESOLVE_RATE_A = {
    "billing": 0.85, "password": 0.92, "shipping": 0.80,
    "returns": 0.75, "login": 0.90, "refund": 0.70, "uncategorized": 0.40,
}

# Window B: a new "crypto_payment" feature launched — high volume,
# low resolution rate because the model wasn't trained/validated on it
RESOLVE_RATE_B = dict(RESOLVE_RATE_A)
RESOLVE_RATE_B["crypto_payment"] = 0.30


def simulate_window(tracker: CoverageTracker, start_ts: float, n: int,
                     resolve_rates: dict, extra_vocab: list[str] | None = None, rng=None):
    rng = rng or random.Random(0)
    vocab = VOCAB + (extra_vocab or [])
    for i in range(n):
        cluster = rng.choice(vocab)
        rate = resolve_rates.get(cluster, 0.5)
        resolved = rng.random() < rate
        event = tracker.log(
            input_text=f"user question about {cluster} #{i}",
            resolved=resolved,
            confidence=rng.uniform(0.4, 0.99) if resolved else rng.uniform(0.1, 0.5),
            cluster=cluster,
        )
        event.timestamp = start_ts + rng.uniform(0, 3600)


def main():
    tracker = CoverageTracker(cluster_vocab=VOCAB, confidence_threshold=0.5)
    rng = random.Random(42)

    t0 = time.time() - 7200  # window A: 2h ago -> 1h ago
    t1 = time.time() - 3600  # window B: 1h ago -> now

    simulate_window(tracker, t0, n=300, resolve_rates=RESOLVE_RATE_A, rng=rng)
    # window B: crypto_payment cluster surges in volume with poor coverage
    simulate_window(
        tracker, t1, n=300, resolve_rates=RESOLVE_RATE_B,
        extra_vocab=["crypto_payment"] * 4,  # weight it heavily to simulate a surge
        rng=rng,
    )

    print("=== Overall coverage ===")
    print(tracker.overall_coverage())

    print("\n=== Coverage by cluster ===")
    for cluster, stats in sorted(tracker.coverage_by_cluster().items()):
        rc = stats["resolved_coverage"]
        print(f"  {cluster:16s} n={stats['n']:4d}  resolved={rc:.1%}" if rc is not None else f"  {cluster}: n=0")

    print("\n=== Lowest-coverage clusters ===")
    for cluster, rate in tracker.lowest_coverage_clusters(top_k=5):
        print(f"  {cluster:16s} {rate:.1%}")

    print("\n=== Flagged drift (window A -> window B) ===")
    flagged = tracker.flagged_drift((t0, t0 + 3600), (t1, t1 + 3600))
    for f in flagged:
        print(f"  [{f['cluster']}] {', '.join(f['reasons'])}")
        print(f"      window_a: n={f['window_a']['n']}, coverage={f['window_a']['resolved_coverage']}")
        print(f"      window_b: n={f['window_b']['n']}, coverage={f['window_b']['resolved_coverage']}")


if __name__ == "__main__":
    main()
