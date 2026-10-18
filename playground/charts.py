"""
Matplotlib figures for the UI. Every chart sits next to a table with the
same numbers in the UI, so values are only direct-labelled where they
matter (the local run), not on every mark.

Colors: the reference palette's categorical slots in fixed order, a
neutral gray for context marks, and the highlighted series is always
slot 1 so "your model" reads the same in every chart.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # slots 1-3 (validated all-pairs)
CONTEXT = "#b9b8b1"
CONTEXT_DARK = "#7d7c76"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e0"


def _style(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _fig(w=8.0, h=4.0):
    fig, ax = plt.subplots(figsize=(w, h), dpi=110)
    fig.patch.set_facecolor("white")
    return fig, ax


def category_chart(rows: list[dict], columns: list[str], title: str):
    """Grouped bars, one group per category, up to three models (first = this run)."""
    rows = rows[:3]
    fig, ax = _fig()
    width = 0.8 / len(rows)
    for i, row in enumerate(rows):
        xs = [j + (i - (len(rows) - 1) / 2) * width for j in range(len(columns))]
        vals = [row.get(c) if row.get(c) is not None else 0 for c in columns]
        bars = ax.bar(xs, vals, width=width - 0.03, color=SERIES[i], label=row["name"])
        if i == 0:
            for bar, c in zip(bars, columns):
                if row.get(c) is not None:
                    ax.annotate(f"{row[c]:.1f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                ha="center", va="bottom", fontsize=8, color=INK, xytext=(0, 2),
                                textcoords="offset points")
    ax.axhline(25, color=CONTEXT_DARK, linewidth=1, linestyle=(0, (4, 3)))
    # label the chance line in the right margin, clear of every bar group
    ax.text(1.01, 25, "chance\n25%", transform=ax.get_yaxis_transform(), ha="left", va="center",
            fontsize=8, color=INK_2)
    ax.set_xticks(range(len(columns)), columns)
    ax.set_ylim(0, 100)
    ax.set_ylabel("accuracy (%)", color=INK_2)
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=24)
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncols=len(rows),
              borderaxespad=0.2)
    _style(ax)
    fig.tight_layout()
    return fig


def ranking_chart(rows: list[dict], title: str):
    """Horizontal ranked bars of overall accuracy; this run highlighted, the rest context."""
    rows = [r for r in rows if r.get("Average") is not None]
    fig, ax = _fig(8.0, max(3.0, 0.32 * len(rows) + 1.0))
    names = [f"{r['name']}  ·  {r['setting']}" if r["kind"] != "local" else r["name"] for r in rows]
    colors = [SERIES[0] if r["kind"] == "local" else CONTEXT_DARK if r["kind"] == "vendor" else CONTEXT
              for r in rows]
    ys = range(len(rows))[::-1]
    ax.barh(list(ys), [r["Average"] for r in rows], color=colors, height=0.7)
    for y, r in zip(ys, rows):
        if r["kind"] == "local":
            ax.annotate(f"{r['Average']:.1f}", (r["Average"], y), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK, fontweight="bold")
    ax.set_yticks(list(ys), names, fontsize=8)
    for tick, r in zip(ax.get_yticklabels(), rows):
        if r["kind"] == "local":
            tick.set_fontweight("bold")
            tick.set_color(INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel("MMLU average accuracy (%)", color=INK_2)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    _style(ax, grid_axis="x")
    fig.tight_layout()
    return fig


def line_chart(xs, series: dict[str, list[float]], title: str, xlabel: str, ylabel: str,
               context: dict[str, list[float]] | None = None, vline: tuple[float, str] | None = None,
               logx: bool = False, logy: bool = False):
    """Up to three highlighted series plus optional gray context series (e.g. a baseline)."""
    fig, ax = _fig()
    for name, ys in (context or {}).items():
        ax.plot(xs, ys, color=CONTEXT_DARK, linewidth=2, linestyle=(0, (4, 3)), label=name)
    for i, (name, ys) in enumerate(list(series.items())[:3]):
        ax.plot(xs, ys, color=SERIES[i], linewidth=2, label=name)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    if vline:
        ax.axvline(vline[0], color=INK_2, linewidth=1)
        ax.annotate(vline[1], (vline[0], 1), xycoords=("data", "axes fraction"), xytext=(4, -12),
                    textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    return fig


def bar_chart(labels: list[str], values: list[float], title: str, xlabel: str):
    """Single-series horizontal bars, sorted ascending so the worst sits on top."""
    pairs = sorted(zip(labels, values), key=lambda p: p[1])
    fig, ax = _fig(8.0, max(2.5, 0.35 * len(pairs) + 1.0))
    ax.barh(range(len(pairs))[::-1], [v for _, v in pairs], color=SERIES[0], height=0.65)
    ax.set_yticks(range(len(pairs))[::-1], [l for l, _ in pairs], fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    _style(ax, grid_axis="x")
    fig.tight_layout()
    return fig
