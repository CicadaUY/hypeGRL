"""Plot a two-stage chart schedule run: the trajectories, and the rate curves.

Left: every arm's whole trajectory, with the shared coarse phase prepended so the
handover is visible. All rates are drawn faintly and each chart's best is drawn solid,
because the spread between rates is usually larger than the gap between charts and a
plot showing only the winners hides that.

Right: the final stress against the learning rate, one line per chart. This is the
panel that carries the result. Two charts are only comparable at each one's own best
rate -- the angular step is lr/w_c(r), so a rate means a different step in each chart --
and this shows whether those optima are separated or whether the grid simply ran out
before finding one (a minimum at an end of the line is not a minimum).

Reads whatever the run has written; usable while a sweep is still going.

Usage:
    python two_stage_chart_schedule_plot.py caterpillar40-4 cuda
    python two_stage_chart_schedule_plot.py fabaceae_sub cuda
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"
COLOURS = {"tangent": "#2a78d6"}
CURVED = ["#c0392b", "#8e44ad", "#1e8449", "#d68910"]


def colour_of(chart, charts):
    if chart in COLOURS:
        return COLOURS[chart]
    curved = [c for c in charts if c not in COLOURS]
    return CURVED[curved.index(chart) % len(CURVED)]


def smooth(y, window=201):
    """A running mean as ``(step, value)``, so a noisy trace shows its trend.

    Returns the steps alongside the values because a valid-mode convolution is
    shorter than its input: pairing the result with a plain ``arange`` would
    silently shift every curve left by half a window.
    """
    if len(y) < window:
        return np.arange(len(y)) + 1, y
    smoothed = np.convolve(y, np.ones(window) / window, "valid")
    return np.arange(len(smoothed)) + window // 2 + 1, smoothed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tag", help="graph tag as it appears in the filename")
    ap.add_argument("device", nargs="?", default="cuda")
    args = ap.parse_args()

    stem = RESULTS / f"two_stage_chart_schedule_{args.tag}_{args.device}"
    curves = dict(np.load(stem.with_suffix(".npz")))
    meta = json.load(open(stem.with_suffix(".json")))
    coarse = np.asarray(curves.pop("coarse"), float)

    runs = {}                                   # chart -> {lr: history}
    for key, history in curves.items():
        chart, _, lr = key.partition("__lr")
        runs.setdefault(chart, {})[float(lr)] = np.asarray(history, float)
    charts = sorted(runs, key=lambda c: c != "tangent")
    best = {r["chart"]: r for r in meta["best"]}

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15.0, 6.0))

    ax.plot(*smooth(coarse), color="#555", lw=2.0,
            label=f"coarse: tangent, lr={meta['lr_coarse']:g}")
    for chart in charts:
        colour = colour_of(chart, charts)
        for lr, history in sorted(runs[chart].items()):
            is_best = chart in best and lr == best[chart]["lr"]
            ax.plot(*smooth(np.concatenate([coarse, history])), color=colour,
                    lw=2.4 if is_best else 0.8, alpha=1.0 if is_best else 0.35,
                    label=f"{chart}  lr={lr:g}" if is_best else None)
    ax.axvline(len(coarse), color="black", ls="--", lw=1.2, alpha=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("cumulative step")
    ax.set_ylabel("stress")
    ax.set_title("trajectories; dashed = handover\n"
                 "faint: every rate, solid: each chart's best", fontsize=11)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=9, loc="best")

    for chart in charts:
        colour = colour_of(chart, charts)
        rates = sorted(runs[chart])
        finals = [float(runs[chart][lr][-100:].mean()) for lr in rates]
        ax2.plot(rates, finals, "o-", color=colour, lw=2.0, ms=5, label=chart)
        if chart in best:
            ax2.plot(best[chart]["lr"], best[chart]["stress"], "*", color=colour,
                     ms=18, zorder=5)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("fine-phase learning rate")
    ax2.set_ylabel(f"stress after {meta['n_fine']} steps")
    ax2.set_title("rate curves; star = best\n"
                  "a best at an end of a line means the grid ran out", fontsize=11)
    ax2.grid(alpha=0.25, which="both")
    ax2.legend(fontsize=9, loc="best")

    fig.suptitle(f"{meta['graph']}  N={meta['n_nodes']}  device={meta['device']}  "
                 f"coarse {meta['n_coarse']} + fine {meta['n_fine']} steps",
                 fontsize=13)
    fig.tight_layout()
    out = stem.with_suffix(".png")
    fig.savefig(out, dpi=130, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
