"""
Harvest Playground UI.

    python cli.py ui                 # http://127.0.0.1:7860
    python -m playground.app --port 7861

Tabs: Benchmark (MMLU vs published scores), Capture studio (signed
captures), Arena (blind side-by-side between two open models you run),
Coverage, Economics. Event handlers are plain module-level functions so
tests can call them without a browser.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

from core.economics import api_monthly_cost, co_owned_monthly_cost, pooled_performance, quality_adjusted_cost
from core.production_loop import LoopParams, baseline_cost, simulate

from . import arena, benchmark, capture_studio, charts, coverage_view, mmlu, models

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "mmlu"
RUNS_DIR = ROOT / "runs"
CAPTURE_DIR = ROOT / "captures"
REFERENCES = ROOT / "config" / "reference_scores.json"
ARENA_LOG = RUNS_DIR / "arena_votes.jsonl"

COLUMNS = benchmark.CATEGORY_COLUMNS + ["Average"]
DEFAULT_COMPARE = ["GPT-3 (175B, few-shot)", "Random baseline"]


def _references() -> list[dict]:
    return benchmark.load_references(REFERENCES)


def _category_reference_names() -> list[str]:
    return [r["name"] for r in _references() if all(r.get(c) is not None for c in benchmark.CATEGORY_COLUMNS)]


# ---------------------------------------------------------------- Benchmark

def bench_status(data_dir: str) -> str:
    have = mmlu.available_subjects(data_dir)
    if not have:
        return (f"No MMLU data in `{data_dir}` yet. Pick subjects (or none for all 57) and press "
                "**Download MMLU**.")
    return f"{len(have)} of 57 MMLU subjects available in `{data_dir}`."


def bench_download(data_dir: str, subjects: list[str], progress=gr.Progress()):
    subjects = subjects or mmlu.SUBJECTS
    done = []
    for i, subject in enumerate(subjects):
        progress((i, len(subjects)), desc=f"downloading {subject}")
        mmlu.fetch(data_dir, [subject], log=lambda *_: None)
        done.append(subject)
    return f"Downloaded {len(done)} subject(s). " + bench_status(data_dir)


def bench_select(category: str) -> list[str]:
    if category == "All 57":
        return list(mmlu.SUBJECTS)
    return [s for s in mmlu.SUBJECTS if mmlu.category_of(s) == category]


def bench_run(backend, model, base_url, license_tag, data_dir, subjects, n_per_subject, k_shot,
              compare_names, progress=gr.Progress()):
    if not subjects:
        raise gr.Error("Pick at least one subject.")
    missing = sorted(set(subjects) - set(mmlu.available_subjects(data_dir)))
    if missing:
        raise gr.Error(f"Not downloaded yet: {', '.join(missing[:5])}{'…' if len(missing) > 5 else ''}. "
                       "Press Download MMLU first.")
    try:
        loaded = models.load(backend, model, base_url, license_tag)
    except Exception as e:  # surface load errors in the UI instead of a stack trace
        raise gr.Error(f"Could not load model: {e}")

    result = benchmark.run_benchmark(
        loaded.scorer, data_dir, list(subjects), loaded.label, k_shot=int(k_shot),
        n_per_subject=int(n_per_subject) or None,
        progress=lambda i, n, s: progress((i, n), desc=f"{mmlu.pretty(s)}"),
    )
    saved = result.save(RUNS_DIR / "benchmarks")
    return bench_render(result, compare_names) + (f"Saved to `{saved.relative_to(ROOT)}`",)


def bench_render(result: benchmark.BenchmarkResult, compare_names: list[str]):
    refs = _references()
    table = benchmark.comparison_table(result, refs)
    mine = next(r for r in table if r["kind"] == "local")
    picked = [r for r in table if r["name"] in (compare_names or [])][:2]
    cats = result.by_category()

    n_total = len(result.questions)
    avg = cats["Average"]["accuracy"]
    lines = [
        f"### {result.model_label}: **{avg:.1f}%** on {n_total} questions",
        f"{len(result.by_subject())} subject(s) · {result.k_shot}-shot · scorer: {result.scorer} · "
        f"{result.duration_s:.1f}s",
    ]
    if n_total < 500:
        # ±1.96·sqrt(p(1-p)/n): rough 95% interval, to stop small runs being over-read
        p = avg / 100
        lines.append(f"Small sample: the 95% interval on the average is about ±{196 * (p * (1 - p) / n_total) ** 0.5:.1f} "
                     "points. Use more questions per subject for a firmer comparison.")
    if result.unparsed:
        lines.append(f"{result.unparsed} answer(s) could not be parsed and were scored wrong.")
    if result.scorer.startswith("mock"):
        lines.append("**Mock backend:** these are random guesses, not a model. Expect about 25%.")
    lines.append("Vendor rows report overall scores only, and some use different prompting (shown after the name).")

    cat_fig = charts.category_chart([mine] + picked, COLUMNS, "Accuracy by MMLU category")
    rank_fig = charts.ranking_chart(table, "Where this run lands among published MMLU scores")
    comparison = pd.DataFrame(table)[["name", "kind", "setting"] + COLUMNS]
    subjects = pd.DataFrame([
        {"subject": mmlu.pretty(s), "category": d["category"], "questions": d["n"],
         "accuracy %": round(d["accuracy"], 1)}
        for s, d in result.by_subject().items()
    ]).sort_values("accuracy %")
    return "\n\n".join(lines), cat_fig, rank_fig, comparison, subjects


# ---------------------------------------------------------------- Capture studio

def cap_preview(strategy, topic_set, custom_topics, n, seed):
    try:
        topics = capture_studio.resolve_topics(topic_set, custom_topics)
    except ValueError as e:
        raise gr.Error(str(e))
    prompts = capture_studio.preview(strategy, topics, int(n), int(seed))
    return pd.DataFrame({"#": range(1, len(prompts) + 1), "prompt": prompts})


def cap_run(backend, model, base_url, license_tag, strategy, topic_set, custom_topics, n, seed,
            max_new_tokens, out_dir):
    try:
        topics = capture_studio.resolve_topics(topic_set, custom_topics)
        loaded = models.load(backend, model, base_url, license_tag)
    except Exception as e:
        raise gr.Error(str(e))
    record = capture_studio.capture(loaded.target, strategy, topics, int(n), out_dir,
                                    Path(out_dir) / ".signing_key", int(seed), int(max_new_tokens))
    path = Path(record["_written_to"])
    ok = capture_studio.ProvenanceSigner.verify({k: v for k, v in record.items() if k != "_written_to"})
    status = (f"Captured **{len(record['batch']['results'])}** samples from `{loaded.label}` "
              f"(license tag `{record['batch']['license_tag']}`) → `{path.name}` · signature "
              f"{'valid ✓' if ok else 'INVALID'}")
    batches, dropdown = cap_refresh(out_dir, select=path.name)
    return status, _results_frame(record), batches, dropdown


def cap_refresh(out_dir, select=None):
    rows = capture_studio.list_batches(out_dir)
    frame = pd.DataFrame(rows, columns=["file", "model", "strategy", "license", "samples", "signature"])
    files = [r["file"] for r in rows]
    return frame, gr.update(choices=files, value=select or (files[0] if files else None))


def cap_view(out_dir, file_name):
    if not file_name:
        return "Pick a batch.", pd.DataFrame()
    record = capture_studio.load_batch(out_dir, file_name)
    batch = record["batch"]
    ok = capture_studio.ProvenanceSigner.verify(record)
    meta = (f"**{file_name}** · model `{batch['model_id']}` · strategy `{batch['strategy_name']}` · "
            f"license `{batch['license_tag']}` · signer `{record['pubkey_hex'][:16]}…` · "
            f"signature **{'valid ✓' if ok else 'INVALID ✗ (modified after signing)'}**")
    return meta, _results_frame(record)


def _results_frame(record):
    return pd.DataFrame([{"prompt": r["prompt"], "response": r["response"], "latency s": round(r["latency_s"], 2)}
                         for r in record["batch"]["results"]])


# ---------------------------------------------------------------- Arena

def arena_random_prompt(topic_set):
    topics = capture_studio.resolve_topics(topic_set)
    topic = random.choice(topics)
    return capture_studio.preview("distill", [topic], 1, seed=random.randrange(10**6))[0], topic


def _contenders(a, b):
    labels = [models.load(*a).label, models.load(*b).label]
    if labels[0] == labels[1]:
        labels = [f"A: {labels[0]}", f"B: {labels[1]}"]
    return {labels[0]: models.load(*a).target, labels[1]: models.load(*b).target}


def arena_generate(a_backend, a_model, a_url, b_backend, b_model, b_url, prompt, topic, max_new_tokens):
    try:
        contenders = _contenders((a_backend, a_model, a_url, "arena-eval-only"),
                                 (b_backend, b_model, b_url, "arena-eval-only"))
        match = arena.make_match(prompt, topic, contenders, max_new_tokens=int(max_new_tokens))
    except Exception as e:
        raise gr.Error(str(e))
    on = gr.update(interactive=True)
    return (match.left_response, match.right_response, match, "Which answer is better? Models are hidden until you vote.",
            on, on, on, on)


def arena_vote(outcome, match, log_path=ARENA_LOG):
    if match is None:
        raise gr.Error("Generate a match first.")
    arena.record_vote(match, outcome, log_path)
    verdict = {"left": f"You picked **{match.left_model}**", "right": f"You picked **{match.right_model}**",
               "tie": "You called it a **tie**", "both_bad": "You marked **both bad**"}[outcome]
    reveal = f"{verdict}. Left was `{match.left_model}`, right was `{match.right_model}`."
    off = gr.update(interactive=False)
    table, topics = arena_tables(log_path)
    return (reveal, None, table, topics, off, off, off, off)


def arena_tables(log_path=ARENA_LOG):
    """Standings, plus per-topic win rates for the current top-rated model."""
    votes = arena.load_votes(log_path)
    table = pd.DataFrame(arena.standings(votes),
                         columns=["model", "elo", "win_rate", "matches", "wins", "ties", "both_bad"])
    focus = table["model"].iloc[0] if len(table) else None
    topics = pd.DataFrame(arena.win_rate_by_topic(votes, focus) if focus else [],
                          columns=["topic", "matches", "win_rate"])
    return table, topics


# ---------------------------------------------------------------- Coverage

def cov_load(file, vocab, threshold, use_demo):
    try:
        if use_demo or file is None:
            tracker = coverage_view.demo_tracker(threshold=float(threshold))
            source = "synthetic demo traffic over two hours"
        else:
            words = [w.strip() for w in vocab.split(",") if w.strip()]
            tracker = coverage_view.load_csv(file, words, float(threshold))
            source = f"`{Path(file).name}`"
    except ValueError as e:
        raise gr.Error(str(e))
    overall = tracker.overall_coverage()
    if not overall["n"]:
        raise gr.Error("No events in the file.")
    clusters = coverage_view.cluster_rows(tracker)
    drift = coverage_view.drift_rows(tracker)
    summary = (f"### {100 * overall['resolved_coverage']:.1f}% resolved without a human · "
               f"{100 * overall['confident_coverage']:.1f}% resolved confidently\n\n"
               f"{overall['n']} inputs from {source}. Drift compares the first and second half of the log by time"
               f"{f'; **{len(drift)} cluster(s) flagged**' if drift else '; nothing flagged'}.")
    fig = charts.bar_chart([r["cluster"] for r in clusters], [r["resolved %"] for r in clusters],
                           "Resolved coverage by cluster (worst first)", "resolved without escalation (%)")
    return (summary, fig, pd.DataFrame(clusters),
            pd.DataFrame(drift, columns=["cluster", "why", "before n", "after n"]))


# ---------------------------------------------------------------- Economics

def econ_breakeven(n_contributors, pooled_examples, api_price, penalty):
    n_contributors, pooled_examples = int(n_contributors), float(pooled_examples)
    perf = pooled_performance(pooled_examples)
    volumes = np.logspace(6, np.log10(5e10), 200)
    api = np.array([api_monthly_cost(v, api_price) for v in volumes])
    co = np.array([co_owned_monthly_cost(v, n_contributors)
                   + quality_adjusted_cost(v, perf, gap_penalty_per_mtok=penalty) for v in volumes])
    crossing = next((v for v, a, c in zip(volumes, api, co) if c <= a), None)
    if crossing is None:
        verdict = "The co-owned model **never** beats the flagship API below 50B tokens/month with these inputs."
        vline = None
    else:
        verdict = f"Break-even at about **{crossing / 1e6:,.0f}M tokens/month**; above that the co-owned model is cheaper."
        vline = (crossing / 1e6, f"break-even {crossing / 1e6:,.0f}M")
    summary = (f"{verdict}\n\nPooled performance {perf:.3f} vs. flagship 0.92 · {n_contributors} contributors · "
               f"API ${api_price:.2f}/Mtok · quality penalty ${penalty:.0f}/Mtok of shortfall. "
               "All inputs are stylised assumptions (core/economics.py): read the shape, not the dollar figure.")
    fig = charts.line_chart(volumes / 1e6, {"flagship API": list(api / 1e3),
                                            "co-owned (quality-adjusted)": list(co / 1e3)},
                            "Monthly cost vs. volume", "million tokens / month (log)", "cost ($k / month, log)",
                            vline=vline, logx=True, logy=True)
    return summary, fig


def econ_loop(contributor_examples, months, api_price):
    p = LoopParams(contributor_examples_per_month=float(contributor_examples), api_price_per_mtok=float(api_price))
    months = int(months)
    xs = list(range(1, months + 1))
    series, rows = {}, []
    for policy in ("open_loop", "feedback_only", "full_loop"):
        cum, total = [], 0.0
        for m in simulate(policy, months, p):
            total += m.cost
            cum.append(total / 1e3)
        series[policy] = cum
        rows.append({"policy": policy, f"{months}-month cost ($k)": round(total / 1e3, 1)})
    base = baseline_cost(months, p)
    rows.append({"policy": "all-flagship baseline", f"{months}-month cost ($k)": round(base / 1e3, 1)})
    fig = charts.line_chart(xs, series, "Cumulative cost by policy", "month", "cumulative cost ($k)",
                            context={"all-flagship baseline": [base / months * m / 1e3 for m in xs]})
    if len({r[f"{months}-month cost ($k)"] for r in rows[:3]}) == 1:
        note = ("At this contribution rate no cluster ever gets good enough to move off the flagship API, so all "
                "three policies cost the same: the overhead of running a co-owned model, with no savings. "
                "Raise contributor examples to see the loop take off.")
    else:
        note = ("Lower is better. The gap between policies is what each part of the loop is worth: production "
                "labels (feedback_only vs. open_loop) and aiming contributions at coverage gaps (full_loop).")
    return fig, pd.DataFrame(rows), note


# ---------------------------------------------------------------- Layout

def _model_picker(prefix: str, default_backend: str = "mock"):
    with gr.Row():
        backend = gr.Radio(models.BACKENDS, value=default_backend, label=f"{prefix}backend")
        model = gr.Textbox(label="model path / name (hf_local) or model id (openai_compat)",
                           placeholder="./models/SmolLM2-135M-Instruct", scale=2)
        base_url = gr.Textbox(label="server URL (openai_compat)", placeholder="http://localhost:8000")
    return backend, model, base_url


INTRO = """# Harvest Playground
Benchmark an open-weight model against published flagship scores, capture signed datasets, and compare models
side by side. Backends are open-weight only (local Hugging Face checkpoints or a server you host); **mock** runs
the UI with no model at all."""


def build() -> gr.Blocks:
    topic_names = list(capture_studio.topic_sets())
    with gr.Blocks(title="Harvest Playground") as demo:
        gr.Markdown(INTRO)
        with gr.Tabs():
            with gr.Tab("Benchmark"):
                b_backend, b_model, b_url = _model_picker("")
                b_license = gr.State("benchmark-eval-only")
                with gr.Row():
                    b_data = gr.Textbox(value=str(DATA_DIR), label="MMLU data folder", scale=3)
                    b_download = gr.Button("Download MMLU", scale=1)
                b_status = gr.Markdown(bench_status(str(DATA_DIR)))
                with gr.Row():
                    b_quick = gr.Radio(["All 57"] + benchmark.CATEGORY_COLUMNS, label="quick select", value=None)
                b_subjects = gr.Dropdown(mmlu.SUBJECTS, multiselect=True, label="subjects",
                                         value=["high_school_computer_science", "world_religions",
                                                "high_school_microeconomics", "nutrition"])
                with gr.Row():
                    b_n = gr.Slider(0, 200, value=20, step=5, label="questions per subject (0 = all)")
                    b_k = gr.Slider(0, 5, value=5, step=1, label="few-shot examples")
                    b_compare = gr.Dropdown(_category_reference_names(), multiselect=True, max_choices=2,
                                            value=DEFAULT_COMPARE, label="compare categories with (up to 2)")
                b_run = gr.Button("Run benchmark", variant="primary")
                b_summary = gr.Markdown()
                with gr.Row():
                    b_cat_plot = gr.Plot(label="by category", show_label=False)
                    b_rank_plot = gr.Plot(label="ranking", show_label=False)
                b_table = gr.Dataframe(label="comparison (published scores + this run)", interactive=False)
                b_subject_table = gr.Dataframe(label="per subject (worst first)", interactive=False)
                b_saved = gr.Markdown()

                b_download.click(bench_download, [b_data, b_subjects], b_status)
                b_quick.change(bench_select, b_quick, b_subjects)
                b_run.click(bench_run, [b_backend, b_model, b_url, b_license, b_data, b_subjects, b_n, b_k, b_compare],
                            [b_summary, b_cat_plot, b_rank_plot, b_table, b_subject_table, b_saved])

            with gr.Tab("Capture studio"):
                c_backend, c_model, c_url = _model_picker("")
                with gr.Row():
                    c_license = gr.Textbox(value="check-model-card", label="license tag (what you checked)")
                    c_strategy = gr.Radio(list(capture_studio.STRATEGIES), value="distill", label="strategy")
                    c_topic_set = gr.Dropdown(topic_names, value="mmlu" if "mmlu" in topic_names else None,
                                              label="topic set")
                    c_custom = gr.Textbox(label="or custom topics (comma-separated)")
                with gr.Row():
                    c_n = gr.Slider(1, 100, value=5, step=1, label="samples")
                    c_seed = gr.Number(value=0, precision=0, label="seed")
                    c_tokens = gr.Slider(8, 1024, value=128, step=8, label="max new tokens")
                    c_out = gr.Textbox(value=str(CAPTURE_DIR), label="output folder")
                with gr.Row():
                    c_preview_btn = gr.Button("Preview prompts")
                    c_run_btn = gr.Button("Run capture", variant="primary")
                c_status = gr.Markdown()
                c_preview = gr.Dataframe(label="prompts", interactive=False)
                c_results = gr.Dataframe(label="captured samples", interactive=False, wrap=True)
                gr.Markdown("### Batches")
                with gr.Row():
                    c_refresh = gr.Button("Refresh", scale=0)
                    c_file = gr.Dropdown([], label="open batch", scale=3)
                c_batches = gr.Dataframe(label="all batches (signature checked on load)", interactive=False)
                c_meta = gr.Markdown()
                c_batch_rows = gr.Dataframe(label="batch contents", interactive=False, wrap=True)

                c_preview_btn.click(cap_preview, [c_strategy, c_topic_set, c_custom, c_n, c_seed], c_preview)
                c_run_btn.click(cap_run, [c_backend, c_model, c_url, c_license, c_strategy, c_topic_set, c_custom,
                                          c_n, c_seed, c_tokens, c_out], [c_status, c_results, c_batches, c_file])
                c_refresh.click(cap_refresh, c_out, [c_batches, c_file])
                c_file.change(cap_view, [c_out, c_file], [c_meta, c_batch_rows])

            with gr.Tab("Arena"):
                gr.Markdown("Two open models you run, answering the same prompt with sides hidden. "
                            "Vote, then the models are revealed. Arena outputs are for evaluation only and are "
                            "never written to capture batches.")
                with gr.Accordion("Model A", open=True):
                    a_backend, a_model, a_url = _model_picker("A · ")
                with gr.Accordion("Model B", open=True):
                    z_backend, z_model, z_url = _model_picker("B · ")
                with gr.Row():
                    r_topic_set = gr.Dropdown(topic_names, value=topic_names[0] if topic_names else None,
                                              label="topic set for random prompts")
                    r_random = gr.Button("Random prompt")
                    r_tokens = gr.Slider(8, 1024, value=192, step=8, label="max new tokens")
                r_prompt = gr.Textbox(label="prompt", lines=2)
                r_topic = gr.Textbox(label="topic (for per-topic win rates)")
                r_go = gr.Button("Generate both", variant="primary")
                r_match = gr.State(None)
                with gr.Row():
                    r_left = gr.Textbox(label="Left", lines=8, interactive=False)
                    r_right = gr.Textbox(label="Right", lines=8, interactive=False)
                with gr.Row():
                    v_left = gr.Button("◀ Left is better", interactive=False)
                    v_tie = gr.Button("Tie", interactive=False)
                    v_bad = gr.Button("Both bad", interactive=False)
                    v_right = gr.Button("Right is better ▶", interactive=False)
                r_reveal = gr.Markdown()
                init_table, init_topics = arena_tables()
                r_table = gr.Dataframe(init_table, label="standings", interactive=False)
                r_topics = gr.Dataframe(init_topics, label="per-topic win rate of the top-rated model (worst topics first)", interactive=False)

                r_random.click(arena_random_prompt, r_topic_set, [r_prompt, r_topic])
                vote_buttons = [v_left, v_tie, v_bad, v_right]
                r_go.click(arena_generate, [a_backend, a_model, a_url, z_backend, z_model, z_url, r_prompt, r_topic,
                                            r_tokens], [r_left, r_right, r_match, r_reveal, *vote_buttons])
                for btn, outcome in zip(vote_buttons, ("left", "tie", "both_bad", "right")):
                    btn.click(lambda m, o=outcome: arena_vote(o, m), r_match,
                              [r_reveal, r_match, r_table, r_topics, *vote_buttons])

            with gr.Tab("Coverage"):
                gr.Markdown("How much of **your own** traffic your deployed model resolves, by cluster, with drift "
                            "between the first and second half of the log. CSV columns: `input_text`, `resolved`, "
                            "optional `confidence`, `timestamp`, `cluster`.")
                with gr.Row():
                    v_file = gr.File(label="traffic log (CSV)", file_types=[".csv"], type="filepath")
                    with gr.Column():
                        v_vocab = gr.Textbox(value="billing, password, shipping, returns, login, refund",
                                             label="cluster keywords (used when a row has no cluster)")
                        v_threshold = gr.Slider(0, 1, value=0.5, step=0.05, label="confidence threshold")
                        v_demo = gr.Checkbox(value=True, label="use synthetic demo traffic instead")
                        v_go = gr.Button("Analyse", variant="primary")
                v_summary = gr.Markdown()
                v_plot = gr.Plot(label="coverage by cluster", show_label=False)
                with gr.Row():
                    v_clusters = gr.Dataframe(label="clusters", interactive=False)
                    v_drift = gr.Dataframe(label="flagged drift", interactive=False)
                v_go.click(cov_load, [v_file, v_vocab, v_threshold, v_demo], [v_summary, v_plot, v_clusters, v_drift])

            with gr.Tab("Economics"):
                gr.Markdown("Stylised cost models from `core/economics.py` and `core/production_loop.py`. "
                            "Every input is an assumption; the curves show trade-offs, not forecasts.")
                with gr.Row():
                    e_n = gr.Slider(2, 300, value=10, step=1, label="contributors")
                    e_ex = gr.Slider(10_000, 1_000_000, value=100_000, step=10_000, label="pooled examples")
                    e_price = gr.Slider(0.5, 30, value=6.0, step=0.5, label="flagship API $ / Mtok")
                    e_pen = gr.Slider(0, 100, value=30, step=5, label="quality penalty $ / Mtok of shortfall")
                e_summary = gr.Markdown()
                e_plot = gr.Plot(label="break-even", show_label=False)
                gr.Markdown("### Production loop")
                with gr.Row():
                    l_ex = gr.Slider(5_000, 300_000, value=150_000, step=5_000, label="contributor examples / month")
                    l_months = gr.Slider(3, 36, value=12, step=1, label="months")
                l_note = gr.Markdown()
                l_plot = gr.Plot(label="cumulative cost", show_label=False)
                l_table = gr.Dataframe(label="totals", interactive=False)

                e_inputs = [e_n, e_ex, e_price, e_pen]
                for comp in e_inputs:
                    comp.release(econ_breakeven, e_inputs, [e_summary, e_plot])
                for comp in (l_ex, l_months, e_price):
                    comp.release(econ_loop, [l_ex, l_months, e_price], [l_plot, l_table, l_note])
                demo.load(econ_breakeven, e_inputs, [e_summary, e_plot])
                demo.load(econ_loop, [l_ex, l_months, e_price], [l_plot, l_table, l_note])
    return demo


def main(argv=None):
    parser = argparse.ArgumentParser(description="Harvest Playground UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="create a public Gradio link")
    args = parser.parse_args(argv)
    build().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
