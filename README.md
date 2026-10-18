# Harvest Playground

A portable, customizable toolkit for capturing structured (prompt →
response) data from **open-weight models** — for distillation dataset
building, synthetic eval generation, or extraction-resistance
red-teaming of your own deployments.

## Scope, on purpose

This tool only ships adapters for:
1. Local Hugging Face `transformers` checkpoints (`HFLocalTarget`)
2. Self-hosted OpenAI-compatible servers — vLLM, TGI, Ollama, llama.cpp
   server, text-generation-webui (`OpenAICompatLocalTarget`)

There's no adapter for closed commercial APIs. That's intentional — it
means anyone you hand this to can point it at any open-weight model
without having to first work out whether doing so is allowed. If you
add your own adapter for a different backend, that check becomes your
responsibility.

**Before running in distillation mode**, check the specific license of
whatever model you're targeting. Apache-2.0 / MIT-licensed models are
generally fine. Some releases (certain Llama tiers, research-only
checkpoints) restrict using their outputs to train other models —
`license_tag` is threaded through captures and provenance records
specifically so you have a paper trail of what you checked.

## Structure

```
harvest-playground/
├── core/
│   ├── target.py       # backend adapters (HF local, OpenAI-compat local)
│   ├── capture.py       # orchestrates strategy -> target -> signed batch
│   ├── provenance.py    # ed25519 signing/verification of capture batches
│   └── coverage.py      # coverage metering: manual-input replacement tracking + drift detection
├── strategies/
│   ├── base.py               # SamplingStrategy interface — extend this
│   ├── distill.py             # diverse-coverage sampler
│   ├── synth_eval.py          # boundary/edge-case eval sampler
│   └── extraction_probe.py    # repetition + perturbation probes (red-team your own model)
├── config/
│   └── example_targets.yaml
└── cli.py
```

## Quickstart

```bash
pip install transformers torch httpx pynacl

# Distillation-style capture from a local checkpoint
python cli.py distill --model ./llama-3.1-8b-instruct \
    --topics math,coding,writing --n 100

# Same, stratified over the 57 MMLU subjects (config/topics/mmlu.txt)
python cli.py distill --model ./llama-3.1-8b-instruct \
    --topics @config/topics/mmlu.txt --n 100

# Synthetic eval generation against a self-hosted vLLM server
python cli.py syntheval --backend openai_compat \
    --base-url http://localhost:8000 --model-id my-finetune \
    --topics physics,history --n 50

# Verify a capture batch's signature
python cli.py verify --batch-file ./captures/batch_1712345678000.json
```

## Customizing: writing your own strategy

The whole point of the playground is that `strategies/base.py` is a
two-method interface. Anyone porting this can drop in a new sampler
without touching `core/`:

```python
from strategies.base import SamplingStrategy

class MyStrategy(SamplingStrategy):
    name = "my_strategy"

    def generate_prompts(self, n, seed_corpus=None):
        return [f"Custom prompt #{i}" for i in range(n)]
```

Then wire it into `cli.py` (or just call `core.capture.run_capture`
directly from a notebook/script — the CLI is a convenience layer, not
a requirement).

## Provenance

Every capture batch is ed25519-signed (`core/provenance.py`) — same
signing primitive as a model lineage/checkpoint provenance system, just
applied to (prompt, response) batches. This gives you:

- A tamper-evident record of exactly which model, strategy, and config
  produced a given dataset
- Something you can hand downstream (e.g. to someone who'll train on
  data you captured) so they can verify it wasn't altered after capture
- An audit trail of the `license_tag` you attached at capture time

Keys are stored at `--key-file` (default `./captures/.signing_key`) —
treat this like any other private key.

## Coverage metering

`core/coverage.py` is a separate concern from harvesting/capture — it
measures how much of **your own** manual/human input volume your
**own** deployed model now resolves, broken down by topic cluster, with
drift detection between time windows. It doesn't query any external
model.

```python
from core.coverage import CoverageTracker

tracker = CoverageTracker(cluster_vocab=["billing", "shipping", "returns"])

tracker.log(input_text="where's my order", resolved=True, confidence=0.91)
tracker.log(input_text="refund not processed", resolved=False, confidence=0.3)

tracker.overall_coverage()          # {'n': 2, 'resolved_coverage': 0.5, ...}
tracker.coverage_by_cluster()       # per-cluster breakdown
tracker.lowest_coverage_clusters()  # worst-covered clusters, for triage

# Compare two time windows and flag clusters with dropped coverage
# or a surge in input share (e.g. a new feature launch the model
# wasn't validated against):
tracker.flagged_drift(window_a=(t0, t0 + 3600), window_b=(t1, t1 + 3600))
```

Default clustering is a lightweight keyword matcher — pass `cluster_fn`
to plug in embedding-based clustering if you need something finer
grained. See `examples/coverage_demo.py` for a full synthetic run,
including a simulated feature-launch drift scenario.

## Extending backends

If you add an adapter for another backend in `core/target.py`, keep
the same pattern: set `model_id` and `license_tag` on the instance, and
make sure `license_tag` reflects something you've actually verified
permits the use case (distillation vs. eval-only vs. internal
red-teaming) rather than leaving it as a placeholder.
