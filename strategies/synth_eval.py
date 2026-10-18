"""
Eval-set generation: boundary / edge-case prompt sampler.

Useful for building a held-out eval suite by sampling adversarial and
boundary-condition style prompts from a model, rather than curating them
by hand.
"""

from __future__ import annotations

import random

from .base import SamplingStrategy


class SynthEvalSampler(SamplingStrategy):
    name = "synth_eval"

    EDGE_CASE_FRAMES = [
        "What happens at the edge case where {topic} breaks down?",
        "Describe a scenario where standard advice about {topic} fails.",
        "What's an example of {topic} that most people get wrong?",
        "Contrast the textbook version of {topic} with a real-world exception.",
        "Under what conditions does {topic} produce a counterintuitive result?",
    ]

    def __init__(self, topics: list[str], seed: int = 0):
        self.topics = topics
        self.rng = random.Random(seed)

    def generate_prompts(self, n: int, seed_corpus: list[str] | None = None) -> list[str]:
        prompts = []
        if seed_corpus:
            prompts.extend(seed_corpus[:n])

        while len(prompts) < n:
            topic = self.rng.choice(self.topics)
            frame = self.rng.choice(self.EDGE_CASE_FRAMES)
            prompts.append(frame.format(topic=topic))

        return prompts[:n]

    def describe_config(self) -> dict:
        return {"topics": self.topics, "n_frames": len(self.EDGE_CASE_FRAMES)}
