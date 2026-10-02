"""The paper's two figures: stress along the iterations.

``single_curvature``
    fabaceae_sub (left) and falconiformes (right): one curvature for the whole budget,
    from the Hydra initialization.
``two_stage``
    caterpillar(40,4) (left) and marsupialia (right): stage 1 at c -> 0 up to the
    handover (vertical line), then stage 2 at each curvature with a fresh optimizer.

Every curve uses its own best learning rate (``manifest.best``), drawn as
``reference.downsample`` describes. Writes ``icassp2027_<figure>.pdf`` and ``.png`` to
``experiments/results/``.

Run, from the repository root:
    python -m experiments.icassp2027.figures               # from your own runs
    python -m experiments.icassp2027.figures --reference   # from reference/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import ticker  # noqa: E402

from experiments.icassp2027.manifest import FLAT, GRAPHS, LABEL, Run, best  # noqa: E402
from experiments.icassp2027.reference import Reference, Results  # noqa: E402
from experiments.icassp2027.two_stage_chart_schedule import RESULTS  # noqa: E402

# Four hues apart in both lightness and hue, and four line styles, so every curve
# is identifiable in grayscale print; the two ends of the c-axis are the broken ones.
STYLE = {FLAT: ("#52514e", (0, (5, 2))),
         "c=0.2": ("#2a78d6", "-"),
         "c=0.3": ("#eb6834", (0, (6, 2, 1.5, 2))),
         "c=1.0": ("#1baf7a", (0, (1.2, 1.8)))}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdcd6"
# fabaceae_sub spans under one decade, so a plain log axis would label a single tick
FIXED_YTICKS = {"fabaceae_sub": [2e6, 4e6, 1e7]}


def best_points(source: Results | Reference, runs: list[Run], start: int = 0):
    """(x in 10^3 iterations, stress) of the best rate among ``runs``, shifted to
    begin at iteration ``start``."""
    scores = {}
    for run in runs:
        scores.update({lr: (run, s) for lr, s in source.scores(run).items()})
    lr, _ = best({lr: s for lr, (_, s) in scores.items()})
    x, y = source.points(scores[lr][0], lr)
    return (x + start) / 1e3, y


def scientific(v: float, _) -> str:
    """Tick label ``2\\times10^6``, or ``10^7`` for a power of ten."""
    exponent = int(np.log10(v))
    mantissa = v / 10 ** exponent
    if mantissa == 1:
        return f"$10^{{{exponent}}}$"
    return f"${mantissa:g}\\times10^{{{exponent}}}$"


def style_axes(ax, name: str) -> None:
    ax.set_yscale("log")
    ax.margins(y=0.03)
    if name in FIXED_YTICKS:
        ax.yaxis.set_major_locator(ticker.FixedLocator(FIXED_YTICKS[name]))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(scientific))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlabel(r"iteration ($\times 10^3$)", fontsize=11.5, color=INK2, labelpad=2)
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(labelsize=10.5, colors=INK2, which="both", pad=2)


def finish(fig, axes, handles: dict, out: Path) -> None:
    axes[0].set_ylabel("stress", fontsize=11.5, color=INK2, labelpad=2)
    leg = fig.legend(list(handles.values()), [LABEL[c] for c in handles],
                     loc="upper center", ncol=len(handles), frameon=False,
                     fontsize=11.5, handlelength=3.2, columnspacing=1.6,
                     borderaxespad=0.1, bbox_to_anchor=(0.5, 1.02))
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout(rect=(0, 0, 1, 0.935), w_pad=1.2)
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {out}.pdf / .png")


def single_curvature(source: Results | Reference, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75))
    handles = {}
    for ax, name in zip(axes, ["fabaceae_sub", "falconiformes"]):
        for c, (n, rates) in GRAPHS[name].single.items():
            arms = [Run(name, "single", c, n, (lr,)) for lr in rates]
            col, ls = STYLE[c]
            (handles[c],) = ax.plot(*best_points(source, arms),
                                    color=col, ls=ls, lw=1.6)
        ax.set_xlim(0, n / 1e3)
        style_axes(ax, name)
    finish(fig, axes, handles, out_dir / "icassp2027_single_curvature")


def two_stage(source: Results | Reference, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75))
    handles = {}
    for ax, name in zip(axes, ["caterpillar", "marsupialia"]):
        g = GRAPHS[name]
        (n1, _), (n2, blocks) = g.stage1, g.stage2
        handover = Run(name, "stage1", FLAT, n1, (g.stage1_lr,))
        col, ls = STYLE[FLAT]
        ax.plot(*best_points(source, [handover]), color=col, ls=ls, lw=1.6)
        for c, rates in blocks.items():
            block = Run(name, "stage2", c, n2, tuple(rates))
            col, ls = STYLE[c]
            (handles[c],) = ax.plot(*best_points(source, [block], start=n1),
                                    color=col, ls=ls, lw=1.6)
        ax.axvline(n1 / 1e3, color=INK2, lw=0.9, ls=(0, (1, 2)), zorder=0)
        ax.set_xlim(0, (n1 + n2) / 1e3)
        style_axes(ax, name)
    finish(fig, axes, handles, out_dir / "icassp2027_two_stage")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", action="store_true",
                    help="read the committed reference/ instead of raw runs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--results", type=Path, default=RESULTS,
                    help="where the runs are read from")
    ap.add_argument("--out", type=Path, default=RESULTS, help="where the figures go")
    args = ap.parse_args()
    source = Reference() if args.reference else Results(args.device, args.results)
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    single_curvature(source, args.out)
    two_stage(source, args.out)


if __name__ == "__main__":
    main()
