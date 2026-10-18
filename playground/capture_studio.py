"""
Capture studio backend: preview prompts, run signed captures, browse and
verify batches. A thin layer over core.capture / core.provenance so the UI
and the CLI produce byte-for-byte the same kind of batch files.
"""

from __future__ import annotations

import json
from pathlib import Path

from cli import get_or_create_signer, parse_topics
from core.capture import run_capture
from core.provenance import ProvenanceSigner
from core.target import Target
from strategies.distill import DistillationSampler
from strategies.synth_eval import SynthEvalSampler

STRATEGIES = {
    "distill": DistillationSampler,
    "syntheval": SynthEvalSampler,
}

TOPIC_DIR = Path(__file__).resolve().parent.parent / "config" / "topics"


def topic_sets() -> dict[str, Path]:
    return {p.stem: p for p in sorted(TOPIC_DIR.glob("*.txt"))}


def resolve_topics(topic_set: str | None, custom: str = "") -> list[str]:
    """Custom comma-separated topics win over a named topic set."""
    if custom.strip():
        return [t.strip() for t in custom.split(",") if t.strip()]
    if topic_set:
        return parse_topics(f"@{topic_sets()[topic_set]}")
    raise ValueError("choose a topic set or type some topics")


def make_strategy(strategy: str, topics: list[str], seed: int = 0):
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {list(STRATEGIES)}")
    return STRATEGIES[strategy](topics=topics, seed=seed)


def preview(strategy: str, topics: list[str], n: int, seed: int = 0) -> list[str]:
    return make_strategy(strategy, topics, seed).generate_prompts(n)


def capture(target: Target, strategy: str, topics: list[str], n: int, out_dir: str | Path,
            key_file: str | Path, seed: int = 0, max_new_tokens: int | None = None) -> dict:
    signer = get_or_create_signer(str(key_file))
    gen_kwargs = {}
    if max_new_tokens:
        key = "max_new_tokens" if hasattr(target, "max_new_tokens") else "max_tokens"
        gen_kwargs[key] = max_new_tokens
    return run_capture(target, make_strategy(strategy, topics, seed), n, signer,
                       out_dir=out_dir, gen_kwargs=gen_kwargs)


def list_batches(out_dir: str | Path) -> list[dict]:
    """Summary rows for every batch file in out_dir, newest first."""
    rows = []
    for path in sorted(Path(out_dir).glob("batch_*.json"), reverse=True):
        try:
            record = json.loads(path.read_text())
            batch = record["batch"]
        except (json.JSONDecodeError, KeyError):
            rows.append({"file": path.name, "model": "?", "strategy": "?", "license": "?",
                         "samples": 0, "signature": "unreadable"})
            continue
        rows.append({
            "file": path.name,
            "model": batch["model_id"],
            "strategy": batch["strategy_name"],
            "license": batch["license_tag"],
            "samples": len(batch["results"]),
            "signature": "valid" if ProvenanceSigner.verify(record) else "INVALID",
        })
    return rows


def load_batch(out_dir: str | Path, file_name: str) -> dict:
    path = Path(out_dir) / Path(file_name).name  # never read outside out_dir
    return json.loads(path.read_text())
