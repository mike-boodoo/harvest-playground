"""
Sampling strategy interface — the customizable core.

This is the part every prompt engineer porting this playground is
expected to extend. A strategy's only job is: given a budget `n` and an
optional seed corpus, produce a list of prompt strings to send to the
target model.

Subclass this for anything: templated task suites, adversarial probes,
distributional coverage samplers, curriculum generators, whatever your
use case needs. Nothing here is target-specific, so the same strategy
can be pointed at any open-weight model via the target adapters in
core/target.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class SamplingStrategy(ABC):
    name: str = "base"

    @abstractmethod
    def generate_prompts(self, n: int, seed_corpus: list[str] | None = None) -> list[str]:
        """Return up to n prompt strings."""
        ...

    def describe_config(self) -> dict:
        """Override to surface strategy parameters into provenance logs."""
        return {}
