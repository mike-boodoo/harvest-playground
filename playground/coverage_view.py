"""
Coverage tab backend: build a core.coverage.CoverageTracker from an
uploaded traffic log, or from synthetic demo traffic, and summarise it.

Uploaded CSV columns (header row required):
  input_text   required
  resolved     required; true/false, yes/no or 1/0
  confidence   optional float 0-1
  timestamp    optional unix seconds (needed for drift)
  cluster      optional; otherwise assigned by keyword match on input_text
"""

from __future__ import annotations

import csv
import random
import time
from pathlib import Path

from core.coverage import CoverageTracker

TRUE = {"true", "1", "yes", "y", "t"}
FALSE = {"false", "0", "no", "n", "f"}

DEMO_VOCAB = ["billing", "password", "shipping", "returns", "login", "refund"]


def _bool(value: str, line: int) -> bool:
    v = value.strip().lower()
    if v in TRUE:
        return True
    if v in FALSE:
        return False
    raise ValueError(f"line {line}: 'resolved' must be true/false, got {value!r}")


def load_csv(path: str | Path, vocab: list[str], threshold: float = 0.5) -> CoverageTracker:
    tracker = CoverageTracker(cluster_vocab=vocab, confidence_threshold=threshold)
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = {"input_text", "resolved"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing column(s): {', '.join(sorted(missing))}")
        has_time = "timestamp" in reader.fieldnames
        for line, row in enumerate(reader, start=2):
            conf = row.get("confidence", "").strip()
            event = tracker.log(
                input_text=row["input_text"],
                resolved=_bool(row["resolved"], line),
                confidence=float(conf) if conf else None,
                cluster=(row.get("cluster") or "").strip() or None,
            )
            # Without a timestamp column every event gets the same time, so
            # drift_rows() sees no time span and reports nothing, instead of
            # comparing two arbitrary halves of the file by load order.
            if not has_time:
                event.timestamp = 0.0
            elif not (row["timestamp"] or "").strip():
                raise ValueError(f"line {line}: 'timestamp' is empty (fill it in or drop the column)")
            else:
                event.timestamp = float(row["timestamp"])
    return tracker


def demo_tracker(seed: int = 42, threshold: float = 0.5) -> CoverageTracker:
    """Two hours of synthetic support traffic; in the second hour a new
    'crypto_payment' cluster surges and is poorly handled (the drift case)."""
    rng = random.Random(seed)
    rates = {"billing": 0.85, "password": 0.92, "shipping": 0.80,
             "returns": 0.75, "login": 0.90, "refund": 0.70}
    tracker = CoverageTracker(cluster_vocab=DEMO_VOCAB + ["crypto_payment"], confidence_threshold=threshold)
    now = time.time()
    for start, clusters in ((now - 7200, DEMO_VOCAB), (now - 3600, DEMO_VOCAB + ["crypto_payment"] * 3)):
        for i in range(300):
            cluster = rng.choice(clusters)
            resolved = rng.random() < rates.get(cluster, 0.30)
            event = tracker.log(
                input_text=f"user question about {cluster} #{i}",
                resolved=resolved,
                confidence=rng.uniform(0.4, 0.99) if resolved else rng.uniform(0.1, 0.5),
                cluster=cluster,
            )
            event.timestamp = start + rng.uniform(0, 3600)
    return tracker


def cluster_rows(tracker: CoverageTracker) -> list[dict]:
    rows = [{"cluster": c, "inputs": s["n"],
             "resolved %": round(100 * s["resolved_coverage"], 1),
             "confident %": round(100 * s["confident_coverage"], 1)}
            for c, s in tracker.coverage_by_cluster().items()]
    return sorted(rows, key=lambda r: r["resolved %"])


def drift_rows(tracker: CoverageTracker) -> list[dict]:
    """Split the log at its time midpoint and compare the two halves."""
    stamps = [e.timestamp for e in tracker.events]
    if len(stamps) < 2 or min(stamps) == max(stamps):
        return []
    lo, hi = min(stamps), max(stamps)
    mid = (lo + hi) / 2
    flagged = tracker.flagged_drift((lo, mid), (mid, hi + 1))
    return [{"cluster": f["cluster"], "why": "; ".join(f["reasons"]),
             "before n": f["window_a"]["n"], "after n": f["window_b"]["n"]} for f in flagged]
