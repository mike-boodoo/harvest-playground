import json
import subprocess
import sys
from pathlib import Path

import pytest

import cli

ROOT = Path(__file__).resolve().parent.parent


def run_cli(*args, cwd=ROOT):
    return subprocess.run([sys.executable, str(ROOT / "cli.py"), *args], cwd=cwd,
                          capture_output=True, text=True, timeout=300)


def test_parse_topics_inline_and_file(tmp_path):
    assert cli.parse_topics("math,coding") == ["math", "coding"]
    f = tmp_path / "t.txt"
    f.write_text("# header\n\nalgebra\n  # indented comment\n geometry \n")
    assert cli.parse_topics(f"@{f}") == ["algebra", "geometry"]


def test_shipped_topic_files_parse():
    for path in (ROOT / "config" / "topics").glob("*.txt"):
        topics = cli.parse_topics(f"@{path}")
        assert topics and all(t and not t.startswith("#") for t in topics)


def test_benchmark_command_with_mock(mmlu_dir, tmp_path):
    out = tmp_path / "runs"
    proc = run_cli("benchmark", "--backend", "mock", "--data-dir", str(mmlu_dir),
                   "--out-dir", str(out), "--k-shot", "2")
    assert proc.returncode == 0, proc.stderr
    assert "▶ mock (no model) (this run)" in proc.stdout
    assert "GPT-4" in proc.stdout
    [saved] = out.glob("mmlu_*.json")
    assert json.loads(saved.read_text())["k_shot"] == 2


def test_benchmark_command_without_data(tmp_path):
    proc = run_cli("benchmark", "--backend", "mock", "--data-dir", str(tmp_path / "none"))
    assert proc.returncode != 0 and "fetch-mmlu" in proc.stderr


@pytest.mark.parametrize("cmd", ["ui", "fetch-mmlu", "benchmark"])
def test_new_subcommands_have_help(cmd):
    proc = run_cli(cmd, "--help")
    assert proc.returncode == 0 and "usage" in proc.stdout
