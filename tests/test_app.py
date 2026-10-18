"""Handler-level tests for the Gradio app: every tab's callbacks, no browser."""

import gradio as gr
import pandas as pd
import pytest

from playground import app, models


@pytest.fixture(autouse=True)
def isolated_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(app, "ROOT", tmp_path)
    monkeypatch.setattr(app, "ARENA_LOG", tmp_path / "runs" / "arena_votes.jsonl")
    models.clear_cache()


def test_build_constructs_all_tabs():
    demo = app.build()
    assert isinstance(demo, gr.Blocks)
    labels = {getattr(b, "label", None) for b in demo.blocks.values()}
    assert {"Benchmark", "Capture studio", "Arena", "Coverage", "Economics"} <= labels


def test_bench_select():
    assert len(app.bench_select("All 57")) == 57
    stem = app.bench_select("STEM")
    assert "abstract_algebra" in stem and "world_religions" not in stem


def test_bench_status(mmlu_dir, tmp_path):
    assert "2 of 57" in app.bench_status(str(mmlu_dir))
    assert "No MMLU data" in app.bench_status(str(tmp_path / "nothing"))


def test_bench_run_mock_end_to_end(mmlu_dir, tmp_path):
    summary, cat_fig, rank_fig, comparison, subjects, saved = app.bench_run(
        "mock", "", "", "x", str(mmlu_dir), ["abstract_algebra", "world_religions"], 0, 5,
        ["GPT-3 (175B, few-shot)"], progress=lambda *a, **k: None)
    assert "mock (no model)" in summary and "random guesses" in summary
    assert "95% interval" in summary  # 8 questions is a small sample
    assert cat_fig is not None and rank_fig is not None
    assert isinstance(comparison, pd.DataFrame)
    assert comparison["name"].str.contains("this run").sum() == 1
    assert {"GPT-4", "Claude 3 Opus", "Random baseline"} <= set(comparison["name"])
    assert subjects["accuracy %"].is_monotonic_increasing  # worst subject first
    assert "runs/benchmarks/mmlu_" in saved and list((tmp_path / "runs" / "benchmarks").glob("mmlu_*.json"))


def test_bench_run_errors(mmlu_dir):
    with pytest.raises(gr.Error, match="at least one subject"):
        app.bench_run("mock", "", "", "x", str(mmlu_dir), [], 0, 5, [], progress=lambda *a, **k: None)
    with pytest.raises(gr.Error, match="Not downloaded"):
        app.bench_run("mock", "", "", "x", str(mmlu_dir), ["anatomy"], 0, 5, [], progress=lambda *a, **k: None)
    with pytest.raises(gr.Error, match="Could not load model"):
        app.bench_run("hf_local", "", "", "x", str(mmlu_dir), ["world_religions"], 0, 5, [],
                      progress=lambda *a, **k: None)


def test_capture_tab_flow(tmp_path):
    out = str(tmp_path / "captures")
    preview = app.cap_preview("distill", "programming_languages", "", 4, 0)
    assert len(preview) == 4
    status, results, batches, dropdown = app.cap_run("mock", "", "", "apache-2.0", "syntheval", None,
                                                      "physics, history", 3, 0, 32, out)
    assert "Captured **3**" in status and "valid ✓" in status
    assert len(results) == 3 and "mock response" in results["response"][0]
    assert batches["signature"].tolist() == ["valid"]
    meta, rows = app.cap_view(out, dropdown["value"])
    assert "valid ✓" in meta and len(rows) == 3
    assert app.cap_view(out, None)[0] == "Pick a batch."
    with pytest.raises(gr.Error):
        app.cap_preview("distill", None, "", 3, 0)


def test_arena_tab_flow(tmp_path):
    prompt, topic = app.arena_random_prompt("mmlu_pro")
    assert topic in prompt
    left, right, match, note, *buttons = app.arena_generate("mock", "", "", "mock", "", "", prompt, topic, 32)
    assert "mock response" in left and "mock response" in right
    assert {match.left_model, match.right_model} == {"A: mock (no model)", "B: mock (no model)"}
    assert all(b["interactive"] for b in buttons)

    reveal, state, table, topics, *buttons = app.arena_vote("left", match, tmp_path / "votes.jsonl")
    assert match.left_model in reveal and state is None
    assert not any(b["interactive"] for b in buttons)
    assert table["matches"].tolist() == [1, 1] and table.iloc[0]["model"] == match.left_model
    assert topics.iloc[0]["topic"] == topic
    with pytest.raises(gr.Error, match="Generate a match first"):
        app.arena_vote("left", None)


def test_coverage_tab_demo_and_csv(tmp_path):
    summary, fig, clusters, drift = app.cov_load(None, "", 0.5, True)
    assert "600 inputs" in summary and "flagged" in summary
    assert "crypto_payment" in drift["cluster"].tolist()

    csv = tmp_path / "t.csv"
    csv.write_text("input_text,resolved\nbilling help,true\nlogin broken,false\n")
    summary, _, clusters, drift = app.cov_load(str(csv), "billing, login", 0.5, False)
    assert "50.0% resolved" in summary and "t.csv" in summary
    assert drift.empty

    csv.write_text("input_text\nx\n")
    with pytest.raises(gr.Error, match="missing column"):
        app.cov_load(str(csv), "", 0.5, False)


def test_economics_tab():
    summary, fig = app.econ_breakeven(10, 100_000, 6.0, 30)
    assert "Break-even at about" in summary and fig is not None
    never, _ = app.econ_breakeven(300, 10_000, 1.0, 100)
    assert "never" in never

    # 8k examples/month: nothing ignites, every policy pays overhead for nothing (core's documented behaviour)
    fig, table, note = app.econ_loop(8_000, 12, 6.0)
    totals = dict(zip(table["policy"], table["12-month cost ($k)"]))
    assert set(totals) == {"open_loop", "feedback_only", "full_loop", "all-flagship baseline"}
    assert totals["full_loop"] == totals["open_loop"] > totals["all-flagship baseline"]
    assert "cost the same" in note

    # 150k examples/month (the UI default): each loop component strictly lowers cost
    _, table, note = app.econ_loop(150_000, 12, 6.0)
    totals = dict(zip(table["policy"], table["12-month cost ($k)"]))
    assert totals["full_loop"] < totals["feedback_only"] < totals["open_loop"] < totals["all-flagship baseline"]
    assert "Lower is better" in note
