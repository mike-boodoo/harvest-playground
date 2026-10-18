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
│   ├── coverage.py      # coverage metering: manual-input replacement tracking + drift detection
│   ├── topic_graph.py   # topic overlap graph: exhaustive exploration, visibility policy
│   ├── session_flow.py  # per-user session: pools, ownership, takeaway model
│   └── peer_benchmark.py # peer-judged arena + EigenTrust, anchored on moderators
├── strategies/
│   ├── base.py               # SamplingStrategy interface — extend this
│   ├── distill.py             # diverse-coverage sampler
│   ├── synth_eval.py          # boundary/edge-case eval sampler
│   └── extraction_probe.py    # repetition + perturbation probes (red-team your own model)
├── playground/                # interactive UI (Gradio) + the logic behind it
│   ├── app.py                 # five tabs; handlers are plain functions
│   ├── mmlu.py / scorers.py / benchmark.py   # MMLU loading, A-D scoring, comparison
│   ├── capture_studio.py      # preview / capture / browse / verify batches
│   ├── arena.py               # blind side-by-side votes, win rates, Elo
│   ├── coverage_view.py       # CSV traffic logs -> coverage + drift
│   └── charts.py
├── config/
│   ├── example_targets.yaml
│   ├── reference_scores.json  # published MMLU scores to compare against
│   └── topics/                # topic sets for --topics @file
├── tests/                     # pytest suite (runs offline)
└── cli.py
```

## Quickstart

```bash
pip install -r requirements.txt

# Distillation-style capture from a local checkpoint
python cli.py distill --model ./llama-3.1-8b-instruct \
    --topics math,coding,writing --n 100

# Same, stratified over the 57 MMLU subjects (config/topics/mmlu.txt)
python cli.py distill --model ./llama-3.1-8b-instruct \
    --topics @config/topics/mmlu.txt --n 100

# Other topic sets in config/topics/: mmlu_pro.txt (13 broad domains),
# programming_languages.txt (25 popular languages, from GitHub Linguist)

# Synthetic eval generation against a self-hosted vLLM server
python cli.py syntheval --backend openai_compat \
    --base-url http://localhost:8000 --model-id my-finetune \
    --topics physics,history --n 50

# Verify a capture batch's signature
python cli.py verify --batch-file ./captures/batch_1712345678000.json
```

## Interactive UI

```bash
python cli.py ui            # http://127.0.0.1:7860
```

Every tab takes a backend: `hf_local` (a checkpoint path or Hugging Face
name), `openai_compat` (a server you host), or `mock`, which runs the UI
with no model at all and labels its output as fake.

| Tab | What it does |
|---|---|
| **Benchmark** | Runs MMLU (5-shot, next-token scoring like the reference eval) on your model, per subject and per category, and charts it against `config/reference_scores.json`: the MMLU leaderboard plus vendor-reported flagship scores, each tagged with its prompting setup. **Download MMLU** fetches the questions from a GitHub mirror into `data/mmlu/`. |
| **Capture studio** | Pick a strategy and topic set, preview prompts, run a signed capture, then browse batches; each batch's signature is checked when it is listed or opened. |
| **Arena** | Two open models answer the same prompt with sides hidden; vote, then see which was which. Standings (win rate, Elo) and per-topic win rates come from `runs/arena_votes.jsonl`. Arena outputs are never written to capture batches. |
| **Coverage** | Upload a CSV traffic log (`input_text`, `resolved`, optional `confidence`, `timestamp`, `cluster`) or use demo traffic: coverage per cluster plus drift between the first and second half of the log. |
| **Economics** | Sliders over `core/economics.py` (break-even vs. a flagship API) and `core/production_loop.py` (cumulative cost per policy). |

Flagship models are compared through their **published** scores rather
than queried. Querying a closed API would need an adapter this toolkit
deliberately doesn't ship (see "Scope, on purpose"), and published scores
are how labs report these comparisons anyway. Only the 5-shot vendor rows
are like-for-like with the default run; the setting column says which.

The benchmark also runs headless:

```bash
python cli.py fetch-mmlu                          # all 57 subjects, ~6.5 MB
python cli.py benchmark --model ./my-model --n-per-subject 20
```

Screenshots from a run with a tiny randomly-initialised model (so it
scores at chance, 24.4% on 570 questions) are in `docs/screenshots/`.

## Tests

```bash
python -m pytest
```

The suite runs offline: MMLU is replaced by a small fixture in the same
file format, and the one test that needs a real model builds a ~200K
parameter random GPT-2 on the fly.

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

## Topic graph sessions: degrees of separation, takeaways, peer benchmark

`examples/degrees_of_separation_demo.py` simulates one user flow per
person in a small community of #topics (the OQOS connector's model of
topics, curators and overlap):

1. **Explore** (`core/topic_graph.py`): breadth-first through shared
   topics until nothing new can be reached. Everyone in one connected
   component ends up reaching the same people and topics, so access is
   equal and only distance differs.
2. **Show** (`visible_view`): per-topic pseudonyms for people on your own
   topics, counts only beyond that, and topic names hidden when fewer
   than `k` people are on them.
3. **Take away** (`core/session_flow.py`): each user's model learns from
   every topic agent in reach, discounted per hop. Their stake in each
   topic agent comes from `valuation.arrival_order_values` (priority:
   the moderator, i.e. the first arrival, gets the largest share) or
   `valuation.shapley_values` (order-free).
4. **Benchmark** (`core/peer_benchmark.py`): members' models judge each
   other in a peer arena, and EigenTrust turns the verdicts into global
   trust, with moderators as the pre-trusted anchor. The demo compares it
   with an unanchored run and a naive win rate when two models collude.

```bash
python examples/degrees_of_separation_demo.py
```

Models are simulated (a hidden skill per topic); batches are really
signed, with a separate key per user per topic, and pooled.

## Extending backends

If you add an adapter for another backend in `core/target.py`, keep
the same pattern: set `model_id` and `license_tag` on the instance, and
make sure `license_tag` reflects something you've actually verified
permits the use case (distillation vs. eval-only vs. internal
red-teaming) rather than leaving it as a placeholder.
