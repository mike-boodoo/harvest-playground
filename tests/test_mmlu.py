import pytest

from playground import mmlu
from tests.conftest import FIXTURE, write_fixture


def test_every_subject_maps_to_one_of_the_four_categories():
    assert len(mmlu.SUBJECTS) == 57
    cats = {mmlu.category_of(s) for s in mmlu.SUBJECTS}
    assert cats == {"STEM", "Humanities", "Social Sciences", "Other"}
    assert mmlu.category_of("abstract_algebra") == "STEM"
    assert mmlu.category_of("world_religions") == "Humanities"
    assert mmlu.category_of("high_school_microeconomics") == "Social Sciences"
    assert mmlu.category_of("nutrition") == "Other"


def test_load_subject_parses_quoted_fields(mmlu_dir):
    test = mmlu.load_subject(mmlu_dir, "abstract_algebra", "test")
    assert len(test) == 4
    q = test[2]
    assert q.question == "What is 2 ^ 3, with a comma, in it?"
    assert q.choices == ("6", "9", "8", '"12"')
    assert q.answer == "C"


def test_load_subject_rejects_malformed_rows(tmp_path):
    bad = tmp_path / "test" / "anatomy_test.csv"
    bad.parent.mkdir(parents=True)
    bad.write_text("question,a,b,c,d,E\n")
    with pytest.raises(ValueError, match="malformed"):
        mmlu.load_subject(tmp_path, "anatomy")


def test_available_subjects(mmlu_dir, tmp_path):
    assert mmlu.available_subjects(mmlu_dir) == ["abstract_algebra", "world_religions"]
    assert mmlu.available_subjects(tmp_path / "missing") == []


def test_build_prompt_matches_reference_layout(mmlu_dir):
    shots = mmlu.load_subject(mmlu_dir, "abstract_algebra", "dev")[:2]
    q = mmlu.load_subject(mmlu_dir, "abstract_algebra", "test")[0]
    prompt = mmlu.build_prompt(q, shots)
    assert prompt.startswith("The following are multiple choice questions (with answers) about abstract algebra.\n\n")
    assert "What is 2 + 2?\nA. 3\nB. 4\nC. 5\nD. 6\nAnswer: B\n\n" in prompt
    assert prompt.endswith("What is 1 + 1?\nA. 2\nB. 3\nC. 1\nD. 0\nAnswer:")
    # the test question's answer must never leak into its own prompt
    assert prompt.count("Answer: ") == 2


def test_fetch_downloads_both_splits_and_skips_existing(tmp_path):
    mirror = write_fixture(tmp_path / "mirror")
    dest = tmp_path / "data"
    written = mmlu.fetch(dest, ["world_religions"], base_url=mirror.as_uri(), log=lambda *_: None)
    assert sorted(p.name for p in written) == ["world_religions_dev.csv", "world_religions_test.csv"]
    assert len(mmlu.load_subject(dest, "world_religions")) == len(FIXTURE["world_religions"]) - 5
    assert mmlu.fetch(dest, ["world_religions"], base_url=mirror.as_uri(), log=lambda *_: None) == []


def test_fetch_rejects_unknown_subject(tmp_path):
    with pytest.raises(ValueError, match="unknown MMLU subject"):
        mmlu.fetch(tmp_path, ["astrology"])
