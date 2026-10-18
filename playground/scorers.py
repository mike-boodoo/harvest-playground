"""
Multiple-choice scorers: given an MMLU prompt ending in "Answer:", pick A-D.

Two real strategies, matching how MMLU is usually run:

  LogprobScorer     reads the model's next-token distribution and picks the
                    letter with the highest probability. This is the
                    reference eval's method; it needs direct access to the
                    model (HF transformers), and never fails to answer.
  GenerationScorer  generates a short completion through any core.target
                    Target and parses the first A-D letter out of it. Works
                    with OpenAI-compatible servers; an unparseable reply is
                    scored as wrong, as a harness would.

MockScorer answers from a seeded RNG. It exists so the UI and pipeline can
be exercised with no model at all, and labels itself as such everywhere.
"""

from __future__ import annotations

import random
import re
from abc import ABC, abstractmethod

from .mmlu import LETTERS

_LETTER_RE = re.compile(r"\b([ABCD])\b")


class ChoiceScorer(ABC):
    name: str = "base"

    @abstractmethod
    def choose(self, prompt: str) -> tuple[str | None, dict[str, float] | None]:
        """Return (chosen letter or None if unparseable, per-letter probabilities if known)."""


class LogprobScorer(ChoiceScorer):
    name = "next-token logprob"

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        # " A" is how the letter appears after "Answer:"; fall back to "A"
        # for tokenizers that don't fold the leading space into the token.
        self.letter_ids = []
        for letter in LETTERS:
            ids = tokenizer.encode(" " + letter, add_special_tokens=False) or \
                  tokenizer.encode(letter, add_special_tokens=False)
            self.letter_ids.append(ids[-1])
        cfg = model.config
        self.max_len = getattr(cfg, "max_position_embeddings", None) or getattr(cfg, "n_positions", 2048)

    def choose(self, prompt):
        import torch

        ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        # Keep the end of the prompt (the actual question) if it is too long
        # for the context window; earlier few-shot examples go first.
        ids = ids[:, -self.max_len:].to(self.model.device)
        with torch.no_grad():
            logits = self.model(input_ids=ids).logits[0, -1]
        probs = torch.softmax(logits[self.letter_ids].float(), dim=-1).tolist()
        dist = dict(zip(LETTERS, probs))
        return max(dist, key=dist.get), dist


class GenerationScorer(ChoiceScorer):
    name = "generate + parse"

    def __init__(self, target, max_new_tokens: int = 8):
        self.target = target
        self.max_new_tokens = max_new_tokens

    def choose(self, prompt):
        # HFLocalTarget takes max_new_tokens; the OpenAI-style adapters take max_tokens.
        key = "max_new_tokens" if hasattr(self.target, "max_new_tokens") else "max_tokens"
        result = self.target.generate(prompt, **{key: self.max_new_tokens})
        match = _LETTER_RE.search(result.response)
        return (match.group(1) if match else None), None


class MockScorer(ChoiceScorer):
    name = "mock (random guesses, no model)"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def choose(self, prompt):
        return self.rng.choice(LETTERS), None
