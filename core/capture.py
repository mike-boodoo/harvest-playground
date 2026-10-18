"""
Capture orchestrator — runs a sampling strategy against a target,
collects results, signs the batch, writes to disk.

This is the one function most users of the playground will actually call.
"""

from __future__ import annotations

from pathlib import Path

from .target import Target
from .provenance import ProvenanceSigner
from strategies.base import SamplingStrategy


def run_capture(
    target: Target,
    strategy: SamplingStrategy,
    n_prompts: int,
    signer: ProvenanceSigner,
    out_dir: str | Path = "./captures",
    seed_corpus: list[str] | None = None,
    gen_kwargs: dict | None = None,
) -> dict:
    """
    Runs `strategy` against `target` for `n_prompts` samples, signs the
    resulting batch, writes it to `out_dir`, and returns the signed record.
    """
    gen_kwargs = gen_kwargs or {}

    prompts = strategy.generate_prompts(n_prompts, seed_corpus=seed_corpus)
    results = target.generate_batch(prompts, **gen_kwargs)

    signed_record = signer.sign_batch(
        results=results,
        model_id=target.model_id,
        license_tag=target.license_tag,
        strategy_name=strategy.name,
        batch_meta={"n_prompts": n_prompts, "strategy_config": strategy.describe_config()},
    )

    path = signer.save_batch(signed_record, out_dir)
    signed_record["_written_to"] = str(path)
    return signed_record
