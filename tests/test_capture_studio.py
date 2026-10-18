import json

import pytest

from core.provenance import ProvenanceSigner
from playground import capture_studio
from playground.models import MockTarget


def test_topic_sets_include_shipped_files():
    sets = capture_studio.topic_sets()
    assert {"mmlu", "mmlu_pro", "programming_languages"} <= set(sets)


def test_resolve_topics():
    assert capture_studio.resolve_topics("mmlu", " a , b ,, ") == ["a", "b"]  # custom wins
    mmlu = capture_studio.resolve_topics("mmlu")
    assert len(mmlu) == 57 and "abstract algebra" in mmlu
    with pytest.raises(ValueError):
        capture_studio.resolve_topics(None, "")


def test_preview_is_deterministic_per_seed():
    a = capture_studio.preview("syntheval", ["physics"], 4, seed=3)
    assert a == capture_studio.preview("syntheval", ["physics"], 4, seed=3)
    assert len(a) == 4 and all("physics" in p for p in a)
    with pytest.raises(ValueError, match="unknown strategy"):
        capture_studio.preview("nope", ["x"], 1)


def test_capture_writes_a_verifiable_batch(tmp_path):
    record = capture_studio.capture(MockTarget(), "distill", ["math"], 3, tmp_path, tmp_path / ".signing_key")
    assert (tmp_path / ".signing_key").exists()
    rows = capture_studio.list_batches(tmp_path)
    assert len(rows) == 1
    assert rows[0]["file"].startswith("batch_")
    assert {k: v for k, v in rows[0].items() if k != "file"} == {
        "model": "mock (no model)", "strategy": "distillation_coverage",
        "license": "mock-no-model", "samples": 3, "signature": "valid"}
    loaded = capture_studio.load_batch(tmp_path, rows[0]["file"])
    assert ProvenanceSigner.verify(loaded)
    assert loaded["batch"]["results"][0]["prompt"] == record["batch"]["results"][0]["prompt"]


def test_list_batches_flags_tampered_and_unreadable(tmp_path):
    capture_studio.capture(MockTarget(), "distill", ["math"], 2, tmp_path, tmp_path / ".signing_key")
    path = next(tmp_path.glob("batch_*.json"))
    record = json.loads(path.read_text())
    record["batch"]["results"][0]["response"] = "edited after signing"
    path.write_text(json.dumps(record))
    (tmp_path / "batch_0.json").write_text("{not json")

    by_file = {r["file"]: r["signature"] for r in capture_studio.list_batches(tmp_path)}
    assert by_file[path.name] == "INVALID"
    assert by_file["batch_0.json"] == "unreadable"


def test_load_batch_cannot_escape_out_dir(tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "secret.json").write_text("{}")
    with pytest.raises(FileNotFoundError):
        capture_studio.load_batch(tmp_path / "out", "../secret.json")
