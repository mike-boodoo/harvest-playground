"""
Coverage metering — measures what fraction of previously-manual input a
model resolves without human escalation, broken down by topic cluster
and tracked over time for distribution drift.

This is a product/ops metric, not a model-harvesting tool: it logs
*your own* B2C traffic and *your own* model's resolution outcomes. No
querying of third-party models is involved anywhere in this module.

Core concepts:
  - InputEvent: one (input, model_response, resolved?) record
  - CoverageTracker: accumulates events, computes coverage stats
  - Clustering: lightweight keyword-bucket clustering by default;
    swap in embedding-based clustering via `cluster_fn` if you have one
  - Drift: compares cluster-level coverage between two time windows
"""

from __future__ import annotations

import time
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class InputEvent:
    input_text: str
    resolved: bool               # did the model handle it without escalation?
    confidence: float | None = None   # optional model confidence/logprob-derived score
    cluster: str | None = None   # filled in by CoverageTracker if not provided
    timestamp: float = field(default_factory=time.time)
    meta: dict = field(default_factory=dict)


def default_keyword_cluster(text: str, vocab: list[str]) -> str:
    """
    Minimal, dependency-free clustering: assigns an input to whichever
    vocab term it contains first, else 'uncategorized'. Good enough for
    a first pass; replace with embedding-based clustering for anything
    beyond quick triage (see `cluster_fn` param on CoverageTracker).
    """
    lowered = text.lower()
    for term in vocab:
        # \b...\b enforces whole-word matching so "login" doesn't match
        # inside "logins" ambiguously or "billing" inside "hairbilling".
        # re.escape guards against vocab terms containing regex
        # metacharacters (unlikely but cheap to guard against).
        if re.search(r"\b" + re.escape(term.lower()) + r"\b", lowered):
            return term  # first vocab match wins — order your vocab list by priority if terms can co-occur
    return "uncategorized"


class CoverageTracker:
    def __init__(
        self,
        cluster_vocab: list[str] | None = None,
        cluster_fn: Callable[[str], str] | None = None,
        confidence_threshold: float = 0.5,
    ):
        """
        cluster_vocab: keyword list used by the default clusterer.
        cluster_fn: override clusterer, e.g. an embedding-based one:
                    `lambda text: my_embedding_cluster(text)`
        confidence_threshold: minimum confidence to count as "confident coverage"
        """
        self.cluster_vocab = cluster_vocab or []
        self.cluster_fn = cluster_fn or (
            lambda text: default_keyword_cluster(text, self.cluster_vocab)
        )
        self.confidence_threshold = confidence_threshold
        self.events: list[InputEvent] = []

    def log(
        self,
        input_text: str,
        resolved: bool,
        confidence: float | None = None,
        cluster: str | None = None,
        meta: dict | None = None,
    ) -> InputEvent:
        event = InputEvent(
            input_text=input_text,
            resolved=resolved,
            confidence=confidence,
            cluster=cluster or self.cluster_fn(input_text),
            meta=meta or {},
        )
        self.events.append(event)
        return event

    # ---- aggregate metrics ----

    def overall_coverage(self, events: list[InputEvent] | None = None) -> dict:
        evs = events if events is not None else self.events
        if not evs:
            return {"n": 0, "resolved_coverage": None, "confident_coverage": None}

        n = len(evs)

        # resolved_coverage: plain fraction of inputs the model handled
        # without human escalation, no quality bar beyond that flag.
        resolved = sum(1 for e in evs if e.resolved)

        # confident_coverage: stricter — only counts as covered if it
        # was BOTH resolved AND the model's own confidence score cleared
        # the threshold. Events with confidence=None are treated as
        # passing this check by default (no confidence signal available
        # shouldn't silently zero out coverage — that's a separate data
        # quality problem to flag elsewhere, not conflate with low
        # confidence).
        confident = sum(
            1 for e in evs
            if e.resolved and (e.confidence is None or e.confidence >= self.confidence_threshold)
        )
        return {
            "n": n,
            "resolved_coverage": resolved / n,
            "confident_coverage": confident / n,
        }

    def coverage_by_cluster(self, events: list[InputEvent] | None = None) -> dict[str, dict]:
        evs = events if events is not None else self.events
        buckets: dict[str, list[InputEvent]] = defaultdict(list)
        for e in evs:
            buckets[e.cluster].append(e)

        return {
            cluster: self.overall_coverage(cluster_events)
            for cluster, cluster_events in buckets.items()
        }

    def lowest_coverage_clusters(self, top_k: int = 5, min_n: int = 5) -> list[tuple[str, float]]:
        """
        Surfaces the worst-covered clusters with enough volume to be
        meaningful (min_n guards against a single unlucky sample looking
        like a systemic gap).
        """
        by_cluster = self.coverage_by_cluster()

        # min_n filters out clusters with too little volume to trust —
        # a cluster with n=2 and 0% coverage looks alarming but is
        # statistical noise, not a real gap. Only clusters with enough
        # samples to be meaningful are considered.
        scored = [
            (cluster, stats["resolved_coverage"])
            for cluster, stats in by_cluster.items()
            if stats["n"] >= min_n and stats["resolved_coverage"] is not None
        ]
        # Ascending sort by coverage rate — worst clusters first.
        return sorted(scored, key=lambda x: x[1])[:top_k]

    # ---- drift between time windows ----

    def events_in_window(self, start_ts: float, end_ts: float) -> list[InputEvent]:
        return [e for e in self.events if start_ts <= e.timestamp < end_ts]

    def drift_report(self, window_a: tuple[float, float], window_b: tuple[float, float]) -> dict:
        """
        Compares per-cluster coverage and input-share between two time
        windows. Flags clusters where coverage dropped meaningfully or
        where a previously-rare cluster surged in volume (a common
        precursor to a coverage cliff — new input distribution the
        model wasn't validated against).
        """
        evs_a = self.events_in_window(*window_a)
        evs_b = self.events_in_window(*window_b)

        cov_a = self.coverage_by_cluster(evs_a)
        cov_b = self.coverage_by_cluster(evs_b)

        n_a, n_b = len(evs_a), len(evs_b)

        # Union of clusters seen in either window — a cluster that only
        # appears in window B (e.g. a brand-new feature) still needs to
        # show up in the report, not get silently dropped.
        all_clusters = set(cov_a) | set(cov_b)

        report = {}
        for cluster in all_clusters:
            # Default to n=0/no-coverage for clusters absent from one window
            # (e.g. a cluster that existed in A but vanished in B, or vice versa).
            stats_a = cov_a.get(cluster, {"n": 0, "resolved_coverage": None})
            stats_b = cov_b.get(cluster, {"n": 0, "resolved_coverage": None})

            # input_share = what fraction of ALL traffic in that window
            # this cluster represents — this is what catches a volume
            # surge (a cluster going from 2% to 40% of traffic), which is
            # a distinct signal from coverage dropping.
            share_a = stats_a["n"] / n_a if n_a else 0.0
            share_b = stats_b["n"] / n_b if n_b else 0.0

            cov_delta = None
            if stats_a["resolved_coverage"] is not None and stats_b["resolved_coverage"] is not None:
                cov_delta = stats_b["resolved_coverage"] - stats_a["resolved_coverage"]

            report[cluster] = {
                "window_a": stats_a,
                "window_b": stats_b,
                "input_share_delta": share_b - share_a,
                "coverage_delta": cov_delta,
            }

        return report

    def flagged_drift(
        self, window_a: tuple[float, float], window_b: tuple[float, float],
        coverage_drop_threshold: float = -0.10, share_surge_threshold: float = 0.10,
    ) -> list[dict]:
        """Filters drift_report down to clusters worth a human looking at."""
        report = self.drift_report(window_a, window_b)
        flagged = []
        for cluster, stats in report.items():
            reasons = []
            # coverage_drop_threshold is negative (default -0.10), so this
            # catches drops of 10 percentage points or more.
            if stats["coverage_delta"] is not None and stats["coverage_delta"] <= coverage_drop_threshold:
                reasons.append(f"coverage dropped {stats['coverage_delta']:.1%}")
            # share_surge_threshold catches a cluster eating up 10+ more
            # percentage points of total traffic than it did before —
            # this is the "new feature launched and traffic spiked" case,
            # independent of whether coverage on it is good or bad yet.
            if stats["input_share_delta"] >= share_surge_threshold:
                reasons.append(f"input share up {stats['input_share_delta']:.1%}")
            if reasons:
                flagged.append({"cluster": cluster, "reasons": reasons, **stats})
        # Worst coverage_delta first (None treated as 0 / not-worse) so the
        # most urgent regressions surface at the top of the list.
        return sorted(flagged, key=lambda x: x.get("coverage_delta") or 0)
