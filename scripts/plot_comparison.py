#!/usr/bin/env python3
"""Regenerate docs/comparison.png — accuracy per task type, three decision models.

Data sources are the frozen eval JSONs in evals/results/. Run:

    python scripts/plot_comparison.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
from matplotlib.lines import Line2D                                   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evals" / "results"
OUT = ROOT / "docs" / "comparison.png"

# task type -> readable row label
ROWS = [("blocked", "Blocked page (CAPTCHA / rate limit)"),
        ("flight", "Flight search (radio + 3 fields + submit)"),
        ("hotel", "Hotel filters (2 selects + checkbox)"),
        ("contact", "Contact form (3 fields + consent)"),
        ("wiki", "Autocomplete (type → pick option)"),
        ("multi", "Order form (many requirements + clock + note)")]

# model key -> (label, colour, marker, results file, filled)
MODELS = [
    ("base",  "MiniCPM5-2B (untrained)", "#c9ced6", "o", "base.json",          False),
    ("bonsai", "Bonsai-27B (zero-shot)",  "#f0a020", "o", "bonsai.json",        True),
    ("jeva",  "jeva (2B, ours)",          "#0b3d91", "o", "jeva_v7_site1.json", True),
]

NOTES = [
    ("Training a 2B on verified trajectories", "10% → 100% on the same suite"),
    ("Goals that need several manual steps", "0% → 100% (order form row)"),
    ("Subsets where the page pre-fills nothing", "100% (note / clock / contact, 20 each)"),
    ("Merged weights + Q4_K_M quantisation", "100% → 100%, 5.0 GB → 1.6 GB"),
    ("Latency per decision", "jeva 0.23 s · Bonsai-27B 1.75 s · naive HF 8.4 s"),
]


def load(name):
    p = RESULTS / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def rate(d, kind):
    bk = (d or {}).get("by_kind", {}).get(kind)
    if not bk or not bk[1]:
        return None
    return 100.0 * bk[0] / bk[1]


def main():
    data = {k: load(f) for k, _l, _c, _m, f, _f in MODELS}

    fig = plt.figure(figsize=(16.5, 9.0), dpi=150)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.35, 1.0], wspace=0.06,
                          left=0.30, right=0.985, top=0.795, bottom=0.105)
    ax = fig.add_subplot(gs[0, 0])
    axr = fig.add_subplot(gs[0, 1])
    axr.axis("off")

    ys = list(range(len(ROWS)))[::-1]
    for y, (kind, _label) in zip(ys, ROWS):
        vals = [(m, rate(data.get(m), kind)) for m, *_ in MODELS]
        present = [v for _m, v in vals if v is not None]
        if present and len(present) > 1:
            ax.plot([min(present), max(present)], [y, y], color="#dfe3ea", lw=2.2, zorder=1)
        for m, v in vals:
            if v is None:
                continue
            spec = next((s for s in MODELS if s[0] == m), None)
            if spec is None:
                continue
            _k, _l, colour, marker, _f, filled = spec
            ax.plot([v], [y], marker=marker, markersize=11,
                    markerfacecolor=colour if filled else "white",
                    markeredgecolor=colour, markeredgewidth=2.2, zorder=3)
            ax.annotate(f"{v:.0f}", (v, y), textcoords="offset points", xytext=(0, 11),
                        ha="center", fontsize=9, color=colour if filled else "#8b93a1",
                        fontweight="bold" if filled else "normal")

    ax.set_yticks(ys, [label for _k, label in ROWS], fontsize=10.5, color="#20242b")
    ax.set_xlim(-3, 108)
    ax.set_xticks([0, 20, 40, 60, 80, 100], ["0%", "20%", "40%", "60%", "80%", "100%"])
    ax.tick_params(axis="x", labelsize=10, colors="#5b6472")
    ax.set_ylim(-0.75, len(ROWS) - 0.25)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#e3e7ee")
    ax.grid(axis="x", color="#eef1f6", lw=1, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("Accuracy by task type", fontsize=12.5, color="#5b6472", pad=10, loc="left")

    handles = [Line2D([], [], marker=mk, linestyle="", markersize=10,
                      markerfacecolor=c if fl else "white", markeredgecolor=c, markeredgewidth=2.2,
                      label=lab)
               for _k, lab, c, mk, _f, fl in MODELS]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.05), ncol=3,
              frameon=False, fontsize=10.5, handletextpad=0.5, columnspacing=2.2)

    axr.set_title("What moved the number", fontsize=13.5, color="#111418", pad=26, loc="left")
    y = 0.94
    for head, val in NOTES:
        axr.text(0.0, y, head, fontsize=10.2, color="#20242b", transform=axr.transAxes, va="top")
        axr.text(0.0, y - 0.038, val, fontsize=9.4, color="#6b7280", transform=axr.transAxes, va="top")
        y -= 0.105
    axr.text(0.0, y - 0.01,
             "Held-out: site3 uses a different label vocabulary and\n"
             "layout that never appears in training.",
             fontsize=9.2, color="#8b93a1", transform=axr.transAxes, va="top")

    fig.suptitle("jeva: where a 2B decision model matches and beats a 27B, on browser tasks",
                 fontsize=17, color="#0b0e12", x=0.015, ha="left", y=0.970)
    fig.text(0.015, 0.920,
             "Task success rate on randomised suites, each task executed in a real Chrome and verified "
             "against the resulting page state (40–100 tasks per model).\n"
             "Every model received the same prompt and the same observations; only jeva was trained for "
             "this action space.",
             fontsize=10.2, color="#5b6472", ha="left", va="top")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor="white")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
