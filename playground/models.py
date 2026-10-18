"""
One place to turn UI choices into a (Target, ChoiceScorer) pair.

Backends are the toolkit's own open-weight adapters (core/target.py) plus
a no-model mock for trying the UI out. Loaded models are cached by their
settings, because reloading weights on every button press would make the
UI unusable with anything larger than a toy model.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from core.target import GenerationResult, HFLocalTarget, OpenAICompatLocalTarget, Target

from .scorers import ChoiceScorer, GenerationScorer, LogprobScorer, MockScorer

BACKENDS = ["mock", "hf_local", "openai_compat"]


class MockTarget(Target):
    """Canned, clearly-labelled responses. For exercising the UI without a model."""

    model_id = "mock (no model)"
    license_tag = "mock-no-model"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        topic_hint = prompt.rstrip("?.").split()[-3:]
        response = (f"[mock response — no model loaded] A placeholder answer about "
                    f"{' '.join(topic_hint)}. Load a real model to see real output.")
        return GenerationResult(prompt=prompt, response=response, logprobs=None,
                                latency_s=round(self.rng.uniform(0.01, 0.05), 3))


@dataclass
class LoadedModel:
    label: str
    target: Target
    scorer: ChoiceScorer


_cache: dict[tuple, LoadedModel] = {}


def load(backend: str, model: str = "", base_url: str = "", license_tag: str = "check-model-card",
         max_new_tokens: int = 256) -> LoadedModel:
    key = (backend, model.strip(), base_url.strip(), license_tag.strip(), max_new_tokens)
    if key in _cache:
        return _cache[key]

    if backend == "mock":
        target = MockTarget()
        loaded = LoadedModel("mock (no model)", target, MockScorer())
    elif backend == "hf_local":
        if not model.strip():
            raise ValueError("hf_local needs a model path or Hugging Face model name")
        target = HFLocalTarget(model.strip(), license_tag=license_tag.strip(), dtype="float32",
                               max_new_tokens=max_new_tokens)
        # A local checkpoint folder is labelled by its name, not its full path.
        label = Path(model.strip()).name if Path(model.strip()).is_dir() else model.strip()
        loaded = LoadedModel(label, target, LogprobScorer(target.model, target.tokenizer))
    elif backend == "openai_compat":
        if not (base_url.strip() and model.strip()):
            raise ValueError("openai_compat needs a server URL and the model id it serves")
        target = OpenAICompatLocalTarget(base_url.strip(), model.strip(), license_tag=license_tag.strip(),
                                         max_tokens=max_new_tokens)
        loaded = LoadedModel(model.strip(), target, GenerationScorer(target))
    else:
        raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")

    _cache[key] = loaded
    return loaded


def clear_cache() -> None:
    _cache.clear()
