import json

import pytest

from playground import benchmark, mmlu
from playground.scorers import ChoiceScorer
from tests.conftest import FIXTURE


class Oracle(ChoiceScorer):
    """Answers correctly by looking the question up; records prompts it saw."""
    name = "oracle"

    def __init__(self):
        self.answers = {row[0]: row[5] for rows in FIXTURE.values() for row in rows}
        self.prompts = []

    def choose(self, prompt):
        self.prompts.append(prompt)
        question = prompt.split("\n\n")[-1].split("\n")[0]
        return self.answers[question], None


class Always(ChoiceScorer):
    name = "always"

    def __init__(self, letter):
        self.letter = letter

    def choose(self, prompt):
        return self.letter, None


SUBJECTS = ["abstract_algebra", "world_religions"]


def test_oracle_scores_100_everywhere_it_ran(mmlu_dir):
    result = benchmark.run_benchmark(Oracle(), mmlu_dir, SUBJECTS, "oracle")
    cats = result.by_category()
    assert cats["STEM"] == {"n": 4, "accuracy": 100.0}
    assert cats["Humanities"] == {"n": 4, "accuracy": 100.0}
    assert cats["Average"] == {"n": 8, "accuracy": 100.0}
    # categories with no questions are blank, not zero
    assert cats["Social Sciences"]["accuracy"] is None
    assert cats["Other"]["accuracy"] is None


def test_constant_guess_scores_its_share_of_answers(mmlu_dir):
    result = benchmark.run_benchmark(Always("B"), mmlu_dir, SUBJECTS, "always-B")
    # test answers: algebra A,B,C,D (1 B) and religions B,B,D,A (2 B)
    subj = result.by_subject()
    assert subj["abstract_algebra"]["accuracy"] == 25.0
    assert subj["world_religions"]["accuracy"] == 50.0
    assert result.by_category()["Average"]["accuracy"] == pytest.approx(37.5)


def test_k_shot_and_subset_options(mmlu_dir):
    scorer = Oracle()
    result = benchmark.run_benchmark(scorer, mmlu_dir, ["world_religions"], "o", k_shot=2, n_per_subject=3)
    assert len(result.questions) == 3
    assert all(p.count("Answer: ") == 2 for p in scorer.prompts)

    scorer = Oracle()
    benchmark.run_benchmark(scorer, mmlu_dir, ["world_religions"], "o", k_shot=0)
    assert all("Answer: " not in p for p in scorer.prompts)


@pytest.mark.parametrize("kwargs,msg", [
    ({"subjects": []}, "at least one subject"),
    ({"k_shot": 6}, "k_shot"),
])
def test_bad_arguments(mmlu_dir, kwargs, msg):
    args = {"subjects": SUBJECTS, "k_shot": 5, **kwargs}
    with pytest.raises(ValueError, match=msg):
        benchmark.run_benchmark(Oracle(), mmlu_dir, args["subjects"], "x", k_shot=args["k_shot"])


def test_unparsed_answers_count_as_wrong(mmlu_dir):
    result = benchmark.run_benchmark(Always(None), mmlu_dir, SUBJECTS, "mute")
    assert result.unparsed == 8
    assert result.by_category()["Average"]["accuracy"] == 0.0


def test_progress_callback(mmlu_dir):
    seen = []
    benchmark.run_benchmark(Oracle(), mmlu_dir, SUBJECTS, "o", progress=lambda i, n, s: seen.append((i, n)))
    assert seen[0] == (1, 8) and seen[-1] == (8, 8)


def test_save_round_trips(mmlu_dir, tmp_path):
    result = benchmark.run_benchmark(Always("A"), mmlu_dir, SUBJECTS, "always-A")
    data = json.loads(result.save(tmp_path).read_text())
    assert data["model_label"] == "always-A"
    assert len(data["questions"]) == 8
    assert data["by_category"]["Average"]["n"] == 8


def test_comparison_table_inserts_run_and_sorts(mmlu_dir):
    refs = [
        {"name": "Big", "kind": "vendor", "setting": "5-shot", "Average": 90.0},
        {"name": "Small", "kind": "leaderboard", "setting": "few-shot",
         "STEM": 20.0, "Humanities": 20.0, "Social Sciences": 20.0, "Other": 20.0, "Average": 20.0},
    ]
    result = benchmark.run_benchmark(Always("B"), mmlu_dir, SUBJECTS, "mine")
    table = benchmark.comparison_table(result, refs)
    assert [r["name"] for r in table] == ["Big", "▶ mine (this run)", "Small"]
    mine = table[1]
    assert mine["kind"] == "local" and mine["Average"] == 37.5
    assert mine["Other"] is None
    assert table[0]["STEM"] is None  # vendor rows have no category split
    assert [r["name"] for r in benchmark.comparison_table(result, refs, kinds=("vendor",))] == \
        ["Big", "▶ mine (this run)"]


def test_shipped_reference_file_is_well_formed():
    refs = benchmark.load_references("config/reference_scores.json")
    assert any(r["kind"] == "vendor" for r in refs)
    for r in refs:
        assert r["kind"] in ("leaderboard", "vendor")
        assert r["source"].startswith("https://")
        assert 0 <= r["Average"] <= 100
        if r["kind"] == "leaderboard":
            assert all(0 <= r[c] <= 100 for c in benchmark.CATEGORY_COLUMNS)
    assert mmlu.category_of("abstract_algebra") in benchmark.CATEGORY_COLUMNS
