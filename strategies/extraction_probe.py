"""
Extraction-resistance / membership-inference probing.

Intended use: point this at a model YOU deployed, to test how easily an
adversary could reconstruct training data or distill your fine-tune via
black-box queries. This is the red-team side — you're harvesting from
your own model to find out what a hostile harvester would learn first.

Two probe families:
  - RepetitionProbe: feeds partial sequences from a candidate training
    set and checks for high-confidence verbatim continuation (a classic
    memorization signal).
  - PerturbationProbe: sends near-duplicate prompts and measures
    response-distribution drift, useful for estimating how much of the
    model's decision boundary an attacker could map per query.
"""

from __future__ import annotations

import random

from .base import SamplingStrategy


class RepetitionProbe(SamplingStrategy):
    name = "extraction_repetition_probe"

    def __init__(self, candidate_snippets: list[str], prefix_ratio: float = 0.5, seed: int = 0):
        """
        candidate_snippets: strings suspected to be in the model's training
                             data (e.g. your own fine-tuning set, to check
                             for over-memorization before you ship).
        prefix_ratio: fraction of each snippet to use as the prompt prefix;
                      the rest is held out to compare against the model's
                      continuation for verbatim overlap.
        """
        self.candidate_snippets = candidate_snippets
        self.prefix_ratio = prefix_ratio
        self.rng = random.Random(seed)

    def generate_prompts(self, n: int, seed_corpus: list[str] | None = None) -> list[str]:
        pool = seed_corpus or self.candidate_snippets
        chosen = self.rng.sample(pool, min(n, len(pool)))
        prefixes = []
        for snippet in chosen:
            cut = max(1, int(len(snippet) * self.prefix_ratio))
            prefixes.append(snippet[:cut])
        return prefixes

    def describe_config(self) -> dict:
        return {"prefix_ratio": self.prefix_ratio, "n_candidates": len(self.candidate_snippets)}


class PerturbationProbe(SamplingStrategy):
    name = "extraction_perturbation_probe"

    def __init__(self, base_prompts: list[str], perturbations: list[str] | None = None, seed: int = 0):
        """
        base_prompts: anchor prompts to probe around.
        perturbations: small edits applied to each anchor (paraphrase
                        markers, typo injection, synonym swap hints) —
                        provide your own or use the light default set.
        """
        self.base_prompts = base_prompts
        self.perturbations = perturbations or [
            "{p}",
            "{p} (rephrase if needed)",
            "In other words: {p}",
            "{p}?",
            "Question: {p}",
        ]
        self.rng = random.Random(seed)

    def generate_prompts(self, n: int, seed_corpus: list[str] | None = None) -> list[str]:
        pool = seed_corpus or self.base_prompts
        prompts = []
        while len(prompts) < n:
            base = self.rng.choice(pool)
            pert = self.rng.choice(self.perturbations)
            prompts.append(pert.format(p=base))
        return prompts[:n]

    def describe_config(self) -> dict:
        return {"n_base_prompts": len(self.base_prompts), "n_perturbations": len(self.perturbations)}
