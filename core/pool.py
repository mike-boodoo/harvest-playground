"""
CapturePool — the aggregation layer that was missing.

Previously: each ProvenanceSigner produced one signed batch, verified
1:1, never combined with anyone else's. This module is the multi-party
analog to FedAvg's aggregation step: instead of averaging gradient
updates from N parties into one model, this merges N independently
signed capture batches into one pooled dataset — while, unlike a naive
merge, retaining per-contributor attribution instead of erasing it.

This is what makes it "co-owned" rather than just "co-located": every
prompt/response pair in the pool still traces back to exactly which
signer produced it, which is what valuation.py needs to compute
contribution shares afterward.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .provenance import ProvenanceSigner


@dataclass
class Contribution:
    """One verified batch from one contributor, indexed by their pubkey."""
    contributor_pubkey: str
    model_id: str
    strategy_name: str
    prompts: list[str]
    responses: list[str]
    created_at: float


class CapturePool:
    """
    Accumulates verified batches from multiple contributors into one
    shared pool. Rejects unverifiable batches outright — co-ownership
    requires that every party's contribution is provably theirs, or
    the whole valuation step downstream is meaningless.
    """

    def __init__(self):
        self.contributions: list[Contribution] = []

    def add_batch(self, signed_record: dict) -> Contribution:
        """
        Verify a signed capture batch (as produced by
        ProvenanceSigner.sign_batch) and add it to the pool.
        Raises ValueError if the signature doesn't check out — a pool
        can only be as trustworthy as its worst accepted contribution.
        """
        if not ProvenanceSigner.verify(signed_record):
            raise ValueError("Batch failed signature verification — refusing to pool untrusted data")

        batch = signed_record["batch"]
        contribution = Contribution(
            contributor_pubkey=signed_record["pubkey_hex"],
            model_id=batch["model_id"],
            strategy_name=batch["strategy_name"],
            prompts=[r["prompt"] for r in batch["results"]],
            responses=[r["response"] for r in batch["results"]],
            created_at=batch["created_at"],
        )
        self.contributions.append(contribution)
        return contribution

    def contributors(self) -> list[str]:
        """Unique contributor pubkeys currently in the pool."""
        seen = []
        for c in self.contributions:
            if c.contributor_pubkey not in seen:
                seen.append(c.contributor_pubkey)
        return seen

    def contributions_by(self, pubkey: str) -> list[Contribution]:
        return [c for c in self.contributions if c.contributor_pubkey == pubkey]

    def all_prompts_by(self, pubkey: str) -> list[str]:
        prompts = []
        for c in self.contributions_by(pubkey):
            prompts.extend(c.prompts)
        return prompts

    def total_size(self) -> int:
        return sum(len(c.prompts) for c in self.contributions)
