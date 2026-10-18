"""
Proves the co-ownership mechanism end-to-end:
  1. Three independent users, each with their OWN signing key (no
     shared secrets — this is the multi-party part)
  2. Each captures a batch on a DIFFERENT topic mix (simulating real
     independent contributors who don't coordinate)
  3. Batches get pooled — signatures verified on the way in
  4. Shapley valuation computes who actually added marginal value
  5. Ownership shares come out proportional to real contribution, not
     just batch count

Run: python examples/co_ownership_demo.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.target import Target, GenerationResult
from core.provenance import ProvenanceSigner
from core.capture import run_capture
from core.pool import CapturePool
from core.valuation import shapley_values, ownership_shares
from strategies.base import SamplingStrategy


class MockTarget(Target):
    model_id = "mock-open-model-7b"
    license_tag = "apache-2.0"

    def generate(self, prompt, **kwargs):
        return GenerationResult(prompt=prompt, response=f"[response to: {prompt[:30]}]",
                                 logprobs=None, latency_s=0.001)


class FixedPromptStrategy(SamplingStrategy):
    """Returns a fixed prompt list — lets us control exactly what each contributor covers."""
    name = "fixed"

    def __init__(self, prompts: list[str]):
        self.prompts = prompts

    def generate_prompts(self, n, seed_corpus=None):
        return self.prompts[:n]


def main():
    target = MockTarget()
    vocab = ["billing", "shipping", "returns", "login", "refund"]

    # Three INDEPENDENT contributors, each with their own keypair —
    # nobody shares a secret, nobody trusts anyone else's key.
    alice_signer = ProvenanceSigner()
    bob_signer = ProvenanceSigner()
    carol_signer = ProvenanceSigner()

    # Alice covers 3 distinct topics broadly.
    alice_strategy = FixedPromptStrategy([
        "billing question one", "shipping question one", "returns question one",
    ])
    # Bob duplicates Alice's exact topics — no new coverage, just volume.
    bob_strategy = FixedPromptStrategy([
        "billing question two", "billing question three", "shipping question two",
    ])
    # Carol covers two topics NOBODY else touched.
    carol_strategy = FixedPromptStrategy([
        "login question one", "refund question one",
    ])

    alice_record = run_capture(target, alice_strategy, 3, alice_signer, out_dir="./examples/_co_out")
    bob_record = run_capture(target, bob_strategy, 3, bob_signer, out_dir="./examples/_co_out")
    carol_record = run_capture(target, carol_strategy, 2, carol_signer, out_dir="./examples/_co_out")

    # --- Pooling: each batch verified independently before acceptance ---
    pool = CapturePool()
    pool.add_batch(alice_record)
    pool.add_batch(bob_record)
    pool.add_batch(carol_record)

    print(f"Pool contains {pool.total_size()} prompts from {len(pool.contributors())} contributors")

    # --- Naive split: equal ownership by contributor count ---
    naive_split = {c: 1 / len(pool.contributors()) for c in pool.contributors()}

    # --- Shapley split: ownership weighted by actual marginal coverage added ---
    shap = shapley_values(pool, vocab=vocab, n_permutations=1000)
    shares = ownership_shares(shap)

    pubkey_to_name = {
        alice_signer.export_pubkey_hex(): "Alice",
        bob_signer.export_pubkey_hex(): "Bob",
        carol_signer.export_pubkey_hex(): "Carol",
    }

    print("\n=== Naive (equal-split) ownership ===")
    for pk, share in naive_split.items():
        print(f"  {pubkey_to_name[pk]:8s}: {share:.1%}")

    print("\n=== Shapley (contribution-weighted) ownership ===")
    for pk, share in shares.items():
        print(f"  {pubkey_to_name[pk]:8s}: {share:.1%}")

    print("\nBob contributed the same NUMBER of prompts as Alice, but they")
    print("all duplicate topics Alice already covered — Shapley correctly")
    print("assigns him a smaller ownership share than the naive equal split.")


if __name__ == "__main__":
    main()
