"""
Diverse-coverage sampler for building (prompt, response) training pairs
from an open-weight model — e.g. for distilling a larger open model
into a smaller one you're training, where you already hold rights to
use the teacher model's outputs this way (check the teacher model's
license — most permissive open-weight licenses, e.g. Apache-2.0, MIT,
Llama Community License for non-competing use, allow this; some
(certain Llama tiers, some research-only releases) restrict using
outputs to train other models — read the license before running this
in distillation mode).
"""

from __future__ import annotations

import random

from .base import SamplingStrategy


class DistillationSampler(SamplingStrategy):
    name = "distillation_coverage"

    def __init__(self, topics: list[str], templates: list[str] | None = None,
                 length_bins: tuple[int, ...] = (20, 60, 150), seed: int = 0):
        """
        topics: subject areas to stratify over, e.g. ["math", "coding", "writing"]
        templates: prompt templates with {topic} and optional {length} placeholders.
                   Falls back to a generic template set if not provided.
        length_bins: rough target word counts to stratify prompt complexity over.
        """
        self.topics = topics
        self.templates = templates or [
            "Explain {topic} to someone with no background, in about {length} words.",
            "Write a short example demonstrating {topic}.",
            "What's a common misconception about {topic}?",
            "Compare two approaches within {topic}.",
            "Give a step-by-step walkthrough related to {topic}.",
        ]
        self.length_bins = length_bins
        self.rng = random.Random(seed)

    def generate_prompts(self, n: int, seed_corpus: list[str] | None = None) -> list[str]:
        prompts = []
        if seed_corpus:
            prompts.extend(seed_corpus[:n])

        while len(prompts) < n:
            topic = self.rng.choice(self.topics)
            template = self.rng.choice(self.templates)
            length = self.rng.choice(self.length_bins)
            prompts.append(template.format(topic=topic, length=length))

        return prompts[:n]

    def describe_config(self) -> dict:
        return {"topics": self.topics, "n_templates": len(self.templates), "length_bins": self.length_bins}
