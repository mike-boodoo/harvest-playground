"""
MMLU (Hendrycks et al., "Measuring Massive Multitask Language
Understanding", ICLR 2021) loading and prompt formatting.

The subject -> subcategory -> category map below is copied from
https://github.com/hendrycks/test/blob/master/categories.py (MIT License,
Copyright (c) 2020 Dan Hendrycks). The four top-level categories are the
same ones the MMLU leaderboard reports, which is what lets a local run be
lined up against published scores in config/reference_scores.json.

Data layout is the original release's: <data_dir>/{dev,test}/<subject>_{dev,test}.csv,
one question per row: question, A, B, C, D, answer letter. No header.
"""

from __future__ import annotations

import csv
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# Hugging Face and the original Berkeley host are often unreachable from
# locked-down environments; this GitHub mirror (MIT) carries the same CSVs.
MIRROR_BASE = "https://raw.githubusercontent.com/FranxYao/chain-of-thought-hub/main/MMLU/data"

LETTERS = ("A", "B", "C", "D")

SUBCATEGORIES = {
    "abstract_algebra": "math", "anatomy": "health", "astronomy": "physics",
    "business_ethics": "business", "clinical_knowledge": "health",
    "college_biology": "biology", "college_chemistry": "chemistry",
    "college_computer_science": "computer science", "college_mathematics": "math",
    "college_medicine": "health", "college_physics": "physics",
    "computer_security": "computer science", "conceptual_physics": "physics",
    "econometrics": "economics", "electrical_engineering": "engineering",
    "elementary_mathematics": "math", "formal_logic": "philosophy",
    "global_facts": "other", "high_school_biology": "biology",
    "high_school_chemistry": "chemistry",
    "high_school_computer_science": "computer science",
    "high_school_european_history": "history", "high_school_geography": "geography",
    "high_school_government_and_politics": "politics",
    "high_school_macroeconomics": "economics", "high_school_mathematics": "math",
    "high_school_microeconomics": "economics", "high_school_physics": "physics",
    "high_school_psychology": "psychology", "high_school_statistics": "math",
    "high_school_us_history": "history", "high_school_world_history": "history",
    "human_aging": "health", "human_sexuality": "culture",
    "international_law": "law", "jurisprudence": "law",
    "logical_fallacies": "philosophy", "machine_learning": "computer science",
    "management": "business", "marketing": "business", "medical_genetics": "health",
    "miscellaneous": "other", "moral_disputes": "philosophy",
    "moral_scenarios": "philosophy", "nutrition": "health", "philosophy": "philosophy",
    "prehistory": "history", "professional_accounting": "other",
    "professional_law": "law", "professional_medicine": "health",
    "professional_psychology": "psychology", "public_relations": "politics",
    "security_studies": "politics", "sociology": "culture",
    "us_foreign_policy": "politics", "virology": "health",
    "world_religions": "philosophy",
}

CATEGORIES = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "Humanities": ["history", "philosophy", "law"],
    "Social Sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "Other": ["other", "business", "health"],
}

SUBJECTS = sorted(SUBCATEGORIES)


def category_of(subject: str) -> str:
    sub = SUBCATEGORIES[subject]
    return next(cat for cat, subs in CATEGORIES.items() if sub in subs)


def pretty(subject: str) -> str:
    return subject.replace("_", " ")


@dataclass
class Question:
    subject: str
    question: str
    choices: tuple[str, str, str, str]
    answer: str  # "A".."D"

    def format(self, include_answer: bool) -> str:
        lines = [self.question]
        lines += [f"{letter}. {choice}" for letter, choice in zip(LETTERS, self.choices)]
        lines.append(f"Answer: {self.answer}" if include_answer else "Answer:")
        return "\n".join(lines)


def load_subject(data_dir: str | Path, subject: str, split: str = "test") -> list[Question]:
    path = Path(data_dir) / split / f"{subject}_{split}.csv"
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    questions = []
    for row in rows:
        if len(row) != 6 or row[5].strip() not in LETTERS:
            raise ValueError(f"{path}: malformed MMLU row {row!r}")
        questions.append(Question(subject, row[0], tuple(row[1:5]), row[5].strip()))
    return questions


def available_subjects(data_dir: str | Path) -> list[str]:
    """Subjects with a test split present in data_dir."""
    test_dir = Path(data_dir) / "test"
    if not test_dir.is_dir():
        return []
    return [s for s in SUBJECTS if (test_dir / f"{s}_test.csv").exists()]


def build_prompt(question: Question, shots: list[Question]) -> str:
    """Standard MMLU k-shot prompt (same header and layout as the reference eval)."""
    header = f"The following are multiple choice questions (with answers) about {pretty(question.subject)}.\n\n"
    body = "".join(s.format(include_answer=True) + "\n\n" for s in shots)
    return header + body + question.format(include_answer=False)


def fetch(data_dir: str | Path, subjects: list[str] | None = None,
          base_url: str = MIRROR_BASE, log=print) -> list[Path]:
    """Download the dev (few-shot) and test splits for `subjects` (default: all 57)."""
    written = []
    for subject in subjects or SUBJECTS:
        if subject not in SUBCATEGORIES:
            raise ValueError(f"unknown MMLU subject: {subject}")
        for split in ("dev", "test"):
            dest = Path(data_dir) / split / f"{subject}_{split}.csv"
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(f"{base_url}/{split}/{subject}_{split}.csv", timeout=60) as resp:
                dest.write_bytes(resp.read())
            written.append(dest)
        log(f"[+] {subject}")
    return written
