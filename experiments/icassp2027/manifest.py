"""Every run behind the ICASSP 2027 paper, as data, and how to read its results.

Three kinds of run, all minimizing the Hydra+ stress from the Hydra initialization
with the optimization metric's curvature ``c`` as the only knob:

``stage1``
    The first stage of the two-stage schedule, at ``c = 1e-6`` (the ``c -> 0``
    endpoint), swept over a rate grid. The arm at ``stage1_lr`` is the handover.
``stage2``
    The second stage at one curvature, branching from that handover with a fresh
    optimizer, swept over a rate grid in one invocation of the runner.
``single``
    The single-curvature ablation: one curvature for the whole budget, from the
    initialization, one invocation per rate.

The rate grids are the ones the paper's numbers come from. Each was widened until
its best rate had a worse or divergent neighbour on both sides, or until every rate
gave the same stress, so the grids are irregular in places; they record what was run.
Two budgets differ from the rest and are reported in the paper: cichlidae's
single-curvature runs at ``c in {0.2, 0.3, 1}`` use 100k iterations, half its
two-stage budget, and its flat single-curvature run uses 200k at the best 100k rate.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiments.icassp2027.two_stage_chart_schedule import RESULTS, output_stem

FLAT = "c=0.000001"                       # stand-in for the unreachable c = 0
CURVATURES = (FLAT, "c=0.2", "c=0.3", "c=1.0")
LABEL = {FLAT: r"$c=10^{-6}$", "c=0.2": r"$c=0.2$", "c=0.3": r"$c=0.3$",
         "c=1.0": r"$c=1$"}


@dataclass(frozen=True)
class Graph:
    graph: str                                    # the runner's --graph
    stage1: tuple[int, list[float]]               # (iterations, rates)
    stage1_lr: float                              # the arm handed over
    stage2: tuple[int, dict[str, list[float]]]    # (iterations, {curvature: rates})
    single: dict[str, tuple[int, list[float]]]    # {curvature: (iterations, rates)}


GRAPHS = {
    "caterpillar": Graph(
        graph="caterpillar(40,4)",
        stage1=(100000, [0.0001, 0.0003, 0.001, 0.003]), stage1_lr=0.003,
        stage2=(100000, {
            "c=0.000001": [1e-05, 3e-05, 0.0001, 0.0003, 0.001, 0.003, 0.01],
            "c=0.2": [0.001, 0.003, 0.01, 0.03, 0.1],
            "c=0.3": [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
        }),
        single={
            "c=0.000001": (200000, [0.001, 0.003, 0.01, 0.03]),
            "c=0.2": (200000, [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]),
            "c=0.3": (200000, [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]),
            "c=1.0": (200000, [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]),
        },
    ),
    "marsupialia": Graph(
        graph="marsupialia",
        stage1=(200000, [0.001, 0.003, 0.01]), stage1_lr=0.003,
        stage2=(100000, {
            "c=0.000001": [1e-05, 0.0001, 0.001, 0.003],
            "c=0.2": [0.0003, 0.001, 0.003, 0.01, 0.03],
            "c=0.3": [0.001, 0.003, 0.01, 0.03, 0.1],
        }),
        single={
            "c=0.000001": (300000, [0.001, 0.003, 0.01, 0.03]),
            "c=0.2": (300000, [0.003, 0.01, 0.03, 0.1, 0.3]),
            "c=0.3": (300000, [0.003, 0.01, 0.03, 0.1, 0.3]),
            "c=1.0": (300000, [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]),
        },
    ),
    "fabaceae_sub": Graph(
        graph="fabaceae_sub",
        stage1=(200000, [0.003, 0.01, 0.03]), stage1_lr=0.01,
        stage2=(100000, {
            "c=0.000001": [3e-05, 0.0001, 0.0003, 0.001, 0.003],
            "c=0.2": [0.001, 0.003, 0.01, 0.03, 0.1],
            "c=0.3": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
        }),
        single={
            "c=0.000001": (300000, [0.003, 0.01, 0.03, 0.1]),
            "c=0.2": (300000, [0.01, 0.03, 0.1, 0.3]),
            "c=0.3": (300000, [0.01, 0.03, 0.1, 0.3]),
            "c=1.0": (300000, [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]),
        },
    ),
    "cichlidae": Graph(
        graph="cichlidae",
        stage1=(100000, [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]), stage1_lr=0.01,
        stage2=(100000, {
            "c=0.000001": [3e-05, 0.0001, 0.0003],
            "c=0.2": [0.003, 0.01, 0.03, 0.1, 0.2],
            "c=0.3": [0.03, 0.1, 0.2],
        }),
        single={
            "c=0.000001": (200000, [0.01]),
            "c=0.2": (100000, [0.03, 0.1, 0.3]),
            "c=0.3": (100000, [0.03, 0.1, 0.3]),
            "c=1.0": (100000, [0.01, 0.1, 1.0]),
        },
    ),
    "falconiformes": Graph(
        graph="falconiformes",
        stage1=(100000, [0.001, 0.003, 0.01, 0.03]), stage1_lr=0.001,
        stage2=(100000, {
            "c=0.000001": [3e-06, 1e-05, 3e-05, 0.0001, 0.0003, 0.001],
            "c=0.2": [0.0003, 0.001, 0.003, 0.01, 0.03],
            "c=0.3": [1e-05, 3e-05, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03],
        }),
        single={
            "c=0.000001": (200000, [3e-05, 0.0001, 0.0003, 0.001, 0.003, 0.01]),
            "c=0.2": (200000, [0.001, 0.003, 0.01, 0.03]),
            "c=0.3": (200000, [0.001, 0.003, 0.01, 0.03]),
            "c=1.0": (200000, [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]),
        },
    ),
    "powerlaw200": Graph(
        graph="experiments/data/hubs/powerlaw200.edgelist",
        stage1=(100000, [0.001, 0.003, 0.01, 0.03]), stage1_lr=0.03,
        stage2=(100000, {
            "c=0.000001": [1e-05, 3e-05, 0.0001, 0.0003, 0.001],
            "c=0.2": [0.0003, 0.001, 0.003, 0.01, 0.03],
            "c=0.3": [0.0003, 0.001, 0.003, 0.01, 0.03],
        }),
        single={
            "c=0.000001": (200000, [0.01, 0.03, 0.1, 0.3, 1.0]),
            "c=0.2": (200000, [0.03, 0.1, 0.3, 1.0]),
            "c=0.3": (200000, [0.03, 0.1, 0.3, 1.0]),
            "c=1.0": (200000, [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]),
        },
    ),
    "airports_europe": Graph(
        graph="experiments/data/hubs/airports_europe.edgelist",
        stage1=(100000, [0.001, 0.003, 0.01, 0.03]), stage1_lr=0.01,
        stage2=(100000, {
            "c=0.000001": [1e-05, 3e-05, 0.0001, 0.0003, 0.001],
            "c=0.2": [0.0003, 0.001, 0.003, 0.01, 0.03],
            "c=0.3": [0.0003, 0.001, 0.003, 0.01, 0.03],
        }),
        single={
            "c=0.000001": (200000, [0.003, 0.01, 0.03, 0.1]),
            "c=0.2": (200000, [0.01, 0.03, 0.1, 0.3]),
            "c=0.3": (200000, [0.01, 0.03, 0.1, 0.3]),
            "c=1.0": (200000, [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]),
        },
    ),
}


def slug(curvature: str) -> str:
    """``c=0.2`` -> ``c0.2``; the flat endpoint ``c=0.000001`` -> ``c1e-06``."""
    return f"c{float(curvature.split('=')[1]):g}"


@dataclass(frozen=True)
class Run:
    """One invocation of the runner."""
    name: str                     # key in GRAPHS
    kind: str                     # "stage1", "stage2" or "single"
    curvature: str                # the curvature this run optimizes under
    iterations: int
    rates: tuple[float, ...]      # one rate except for a stage-2 block

    @property
    def spec(self) -> Graph:
        return GRAPHS[self.name]

    @property
    def tag(self) -> str:
        if self.kind == "stage2":
            return f"stage2_{slug(self.curvature)}"
        return f"{self.kind}_lr{self.rates[0]:g}"

    @property
    def coarse_chart(self) -> str:
        """The runner names a file by the curvature of its first stage."""
        return FLAT if self.kind == "stage2" else self.curvature

    def stem(self, device: str, results: Path = RESULTS) -> Path:
        return results / output_stem(self.spec.graph, device, self.coarse_chart,
                                     self.tag).name

    def handover(self) -> Run:
        """The stage-1 arm a stage-2 block branches from."""
        n, _ = self.spec.stage1
        return Run(self.name, "stage1", FLAT, n, (self.spec.stage1_lr,))

    def argv(self, device: str, results: Path = RESULTS) -> list[str]:
        """Arguments for the runner, ``two_stage_chart_schedule``."""
        common = ["--graph", self.spec.graph, "--device", device, "--tag", self.tag]
        if self.kind == "stage2":
            source = f"{self.handover().stem(device, results)}_coords.npz"
            return common + ["--coarse-from", source, "--charts", self.curvature,
                             "--rates", ",".join(repr(r) for r in self.rates),
                             "--n-fine", str(self.iterations)]
        return common + ["--coarse-chart", self.curvature,
                         "--n-coarse", str(self.iterations),
                         "--lr-coarse", repr(self.rates[0]), "--n-fine", "0"]

    def curves(self, device: str, results: Path = RESULTS) -> dict[float, np.ndarray]:
        """{rate: loss per iteration} for every rate that has finished."""
        path = Path(f"{self.stem(device, results)}.npz")
        if not path.exists():
            return {}
        z = np.load(path)
        if self.kind != "stage2":
            return {self.rates[0]: z["coarse"]} if "coarse" in z.files else {}
        return {lr: z[f"{self.curvature}__lr{lr:.10g}"] for lr in self.rates
                if f"{self.curvature}__lr{lr:.10g}" in z.files}

    def done(self, device: str, results: Path = RESULTS) -> bool:
        return len(self.curves(device, results)) == len(self.rates)


def runs(names=None) -> list[Run]:
    """Every run, in an order that can be executed top to bottom: each graph's stage 1
    before the stage-2 blocks that branch from it."""
    out = []
    for name, g in GRAPHS.items():
        if names is not None and name not in names:
            continue
        n1, rates1 = g.stage1
        out += [Run(name, "stage1", FLAT, n1, (lr,)) for lr in rates1]
        n2, blocks = g.stage2
        out += [Run(name, "stage2", c, n2, tuple(r)) for c, r in blocks.items()]
        out += [Run(name, "single", c, n, (lr,))
                for c, (n, rates) in g.single.items() for lr in rates]
    return out


def final_stress(curve: np.ndarray) -> float:
    """Mean stress over the last 10% of a run, ``inf`` if any of it diverged.

    The last iterate alone fluctuates by several percent on the noisier runs."""
    tail = curve[-max(len(curve) // 10, 1):]
    return float(tail.mean()) if np.isfinite(tail).all() else float("inf")


def best(scores: dict[float, float]) -> tuple[float, float]:
    """(best rate, its final stress) among the rates that did not diverge.

    ``scores`` maps each rate to its ``final_stress``."""
    if not scores:
        raise ValueError("no finished runs to choose from; run run_experiments first")
    finite = {lr: s for lr, s in scores.items() if np.isfinite(s)}
    if not finite:
        raise ValueError("every rate diverged")
    lr = min(finite, key=finite.get)
    return lr, finite[lr]
