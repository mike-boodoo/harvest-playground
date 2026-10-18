"""
Run MMLU against a local model and line the result up with published scores.

Accuracy is aggregated the way the MMLU leaderboard reports it: per
subject, then per top-level category (STEM / Humanities / Social
Sciences / Other), with "Average" as accuracy over every question
answered. Runs on a subset (`n_per_subject`) are fine for quick
comparisons but are noisy; the result records how many questions each
number is based on so the UI can say so.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from . import mmlu
from .scorers import ChoiceScorer

CATEGORY_COLUMNS = ["STEM", "Humanities", "Social Sciences", "Other"]


@dataclass
class QuestionResult:
    subject: str
    category: str
    question: str
    answer: str
    predicted: str | None
    correct: bool
    probs: dict[str, float] | None = None


@dataclass
class BenchmarkResult:
    model_label: str
    scorer: str
    k_shot: int
    n_per_subject: int | None
    started_at: float
    duration_s: float
    questions: list[QuestionResult] = field(default_factory=list)

    @staticmethod
    def _acc(rows: list[QuestionResult]) -> float | None:
        return 100.0 * sum(r.correct for r in rows) / len(rows) if rows else None

    def by_subject(self) -> dict[str, dict]:
        groups = defaultdict(list)
        for q in self.questions:
            groups[q.subject].append(q)
        return {s: {"n": len(rows), "accuracy": self._acc(rows), "category": mmlu.category_of(s)}
                for s, rows in sorted(groups.items())}

    def by_category(self) -> dict[str, dict]:
        groups = defaultdict(list)
        for q in self.questions:
            groups[q.category].append(q)
        out = {c: {"n": len(groups[c]), "accuracy": self._acc(groups[c])} for c in CATEGORY_COLUMNS}
        out["Average"] = {"n": len(self.questions), "accuracy": self._acc(self.questions)}
        return out

    @property
    def unparsed(self) -> int:
        return sum(q.predicted is None for q in self.questions)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["by_subject"] = self.by_subject()
        d["by_category"] = self.by_category()
        return d

    def save(self, out_dir: str | Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"mmlu_{int(self.started_at * 1000)}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


def run_benchmark(
    scorer: ChoiceScorer,
    data_dir: str | Path,
    subjects: list[str],
    model_label: str,
    k_shot: int = 5,
    n_per_subject: int | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> BenchmarkResult:
    """
    Score `subjects` from `data_dir` with `scorer`. Few-shot examples come
    from each subject's dev split, as in the reference eval; k_shot=0
    skips them (and the dev files are then not needed).
    """
    if not subjects:
        raise ValueError("pick at least one subject")
    if not 0 <= k_shot <= 5:
        raise ValueError("k_shot must be 0-5 (MMLU dev splits hold 5 examples)")

    plan = []
    for subject in subjects:
        test = mmlu.load_subject(data_dir, subject, "test")
        if n_per_subject:
            test = test[:n_per_subject]
        shots = mmlu.load_subject(data_dir, subject, "dev")[:k_shot] if k_shot else []
        plan += [(q, shots) for q in test]

    result = BenchmarkResult(model_label, scorer.name, k_shot, n_per_subject, time.time(), 0.0)
    for i, (q, shots) in enumerate(plan):
        predicted, probs = scorer.choose(mmlu.build_prompt(q, shots))
        result.questions.append(QuestionResult(
            q.subject, mmlu.category_of(q.subject), q.question, q.answer,
            predicted, predicted == q.answer, probs,
        ))
        if progress:
            progress(i + 1, len(plan), q.subject)
    result.duration_s = time.time() - result.started_at
    return result


def load_references(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text())["models"]


def comparison_table(result: BenchmarkResult, references: list[dict],
                     kinds: tuple[str, ...] = ("leaderboard", "vendor")) -> list[dict]:
    """
    One row per model, columns STEM / Humanities / Social Sciences / Other /
    Average, with the local run inserted and everything sorted by Average.
    Categories the local run didn't touch are left blank (None), not 0.
    """
    cats = result.by_category()
    mine = {"name": f"▶ {result.model_label} (this run)", "kind": "local",
            "setting": f"{result.k_shot}-shot, {result.scorer}"}
    for col in CATEGORY_COLUMNS + ["Average"]:
        acc = cats[col]["accuracy"]
        mine[col] = None if acc is None else round(acc, 1)

    rows = [mine] + [
        {"name": r["name"], "kind": r["kind"], "setting": r.get("setting", ""),
         **{col: r.get(col) for col in CATEGORY_COLUMNS + ["Average"]}}
        for r in references if r["kind"] in kinds
    ]
    return sorted(rows, key=lambda r: -(r["Average"] if r["Average"] is not None else -1))
