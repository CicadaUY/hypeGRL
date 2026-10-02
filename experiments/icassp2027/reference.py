"""The paper's results in a compact, committed form, and one interface to read them.

The raw runs are about 350 MB of loss curves and are not tracked. ``reference/`` keeps
what the table and figures need:

``summary.csv``
    One row per learning rate of every run in the manifest: its final stress (mean
    over the last 10%), its initial stress, and the iteration at which it diverged.
``curves.npz``
    For each graph: the stage-1 handover, and the best rate's curve for every stage-2
    curvature and every single-curvature run, as the points the figures plot (raw for
    the first 250 iterations, then a 500-iteration running mean every 100 iterations).

``Results`` reads the raw runs and ``Reference`` the committed files; ``make_table``
and ``figures`` take either, so a reviewer without the GPU time gets the same table
and figures as a full rerun. Recreate the reference from raw runs with:
    python -m experiments.icassp2027.reference
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from experiments.icassp2027.manifest import GRAPHS, Run, best, final_stress, runs
from experiments.icassp2027.two_stage_chart_schedule import RESULTS

REFERENCE = Path(__file__).resolve().parent / "reference"
WINDOW = 500
FIELDS = ["graph", "kind", "curvature", "iterations", "lr", "final_stress",
          "initial_stress", "diverged_at"]


def downsample(curve: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(iteration, stress) as plotted: raw for the first WINDOW/2 iterations, then a
    centred running mean every 100 iterations. The opening drop is faster than the
    window, so smoothing it would detach the curve from the value it starts at."""
    head = np.unique(np.r_[0:10, 10:WINDOW // 2:5])
    smooth = np.convolve(curve, np.ones(WINDOW) / WINDOW, mode="valid")
    body = np.arange(len(smooth))[::100]
    return (np.concatenate([head, body + WINDOW // 2]),
            np.concatenate([curve[head], smooth[body]]))


def key(run: Run, lr: float) -> str:
    return f"{run.name}/{run.kind}/{run.curvature}/{run.iterations}/{lr:.10g}"


class Results:
    """The raw runs in ``results``, as written by the runner."""

    def __init__(self, device: str = "cuda", results: Path = RESULTS):
        self.device, self.results = device, results

    def scores(self, run: Run) -> dict[float, float]:
        return {lr: final_stress(c)
                for lr, c in run.curves(self.device, self.results).items()}

    def initial(self, run: Run, lr: float) -> float:
        return float(run.curves(self.device, self.results)[lr][0])

    def points(self, run: Run, lr: float) -> tuple[np.ndarray, np.ndarray]:
        return downsample(run.curves(self.device, self.results)[lr])


class Reference:
    """The committed ``summary.csv`` and ``curves.npz``."""

    def __init__(self, path: Path = REFERENCE):
        with open(path / "summary.csv") as f:
            self.rows = {(r["graph"], r["kind"], r["curvature"], int(r["iterations"]),
                          float(r["lr"])): r for r in csv.DictReader(f)}
        self.curves = np.load(path / "curves.npz")

    def _row(self, run: Run, lr: float):
        return self.rows.get((run.name, run.kind, run.curvature, run.iterations, lr))

    def scores(self, run: Run) -> dict[float, float]:
        return {lr: float(self._row(run, lr)["final_stress"])
                for lr in run.rates if self._row(run, lr) is not None}

    def initial(self, run: Run, lr: float) -> float:
        return float(self._row(run, lr)["initial_stress"])

    def points(self, run: Run, lr: float) -> tuple[np.ndarray, np.ndarray]:
        k = key(run, lr)
        if f"{k}/x" not in self.curves.files:
            raise KeyError(f"{k}: the reference keeps only each run's best rate")
        return self.curves[f"{k}/x"], self.curves[f"{k}/y"]


def plotted_runs(name: str) -> list[tuple[Run, ...]]:
    """The groups of runs whose best curve is kept: the handover, each stage-2 block,
    and each single-curvature grid."""
    g = GRAPHS[name]
    (n1, _), (n2, blocks) = g.stage1, g.stage2
    groups = [(Run(name, "stage1", "c=0.000001", n1, (g.stage1_lr,)),)]
    groups += [(Run(name, "stage2", c, n2, tuple(r)),) for c, r in blocks.items()]
    groups += [tuple(Run(name, "single", c, n, (lr,)) for lr in rates)
               for c, (n, rates) in g.single.items()]
    return groups


def export(source: Results, out: Path = REFERENCE) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for run in runs():
            for lr, c in run.curves(source.device, source.results).items():
                bad = ~np.isfinite(c)
                w.writerow(dict(graph=run.name, kind=run.kind, curvature=run.curvature,
                                iterations=run.iterations, lr=repr(lr),
                                final_stress=repr(final_stress(c)),
                                initial_stress=repr(float(c[0])),
                                diverged_at=int(np.argmax(bad)) if bad.any() else ""))
    arrays = {}
    for name in GRAPHS:
        for group in plotted_runs(name):
            scores = {}
            for run in group:
                scores.update({lr: (run, s) for lr, s in source.scores(run).items()})
            lr, _ = best({lr: s for lr, (_, s) in scores.items()})
            run = scores[lr][0]
            # float64: float32 would shift the plotted lines by a fraction of a pixel,
            # and the figures would no longer be reproduced exactly
            x, y = source.points(run, lr)
            arrays[f"{key(run, lr)}/x"] = x.astype(np.int32)
            arrays[f"{key(run, lr)}/y"] = y
    np.savez_compressed(out / "curves.npz", **arrays)
    print(f"wrote {out}/summary.csv and {out}/curves.npz ({len(arrays) // 2} curves)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--out", type=Path, default=REFERENCE)
    args = ap.parse_args()
    export(Results(args.device, args.results), args.out)


if __name__ == "__main__":
    main()
