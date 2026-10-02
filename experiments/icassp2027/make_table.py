"""The paper's results table, recomputed from the runs in the manifest.

Every stress is the mean over the last 10% of a run (``manifest.final_stress``), at
the best learning rate of its grid, diverged rates excluded. Rows:

- graph statistics: nodes, edges, diameter, and the radial span of the Hydra
  initialization (the applicability screen);
- the stress of the Hydra initialization itself;
- single curvature: one curvature for the whole budget;
- two-stage: the handover, then the second stage at each curvature;
- curvature gain: best of c in {0.2, 0.3} against c -> 0 in the second stage;
- schedule gain: best two-stage run against the best single-curvature run.

A single-curvature entry whose budget differs from the two-stage one is marked with a
dagger. Prints a LaTeX ``tabular`` body (stress in units of 10^3) and a plain summary
with the selected learning rates.

Run, from the repository root (the graph statistics take a few minutes, most of it
cichlidae's all-pairs shortest paths):
    python -m experiments.icassp2027.make_table               # from your own runs
    python -m experiments.icassp2027.make_table --reference   # from reference/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx

from experiments.icassp2027.manifest import FLAT, GRAPHS, LABEL, Run, best
from experiments.icassp2027.reference import Reference, Results
from experiments.icassp2027.two_stage_chart_schedule import (
    RESULTS,
    load_graph,
    warm_start,
)

SHORT = {"caterpillar": "Cat.", "marsupialia": "Mar.", "fabaceae_sub": "Fab.",
         "cichlidae": "Cic.", "falconiformes": "Fal.", "powerlaw200": "HK",
         "airports_europe": "Eur."}
STAGE2 = (FLAT, "c=0.2", "c=0.3")


def fmt(stress: float) -> str:
    """Stress in units of 10^3: whole numbers from 100 up, three significant figures
    below, so no column needs an exponent."""
    v = stress / 1e3
    return f"{v:.0f}" if v >= 100 else f"{v:#.3g}"


def collect(name: str, source: Results | Reference) -> dict:
    g = GRAPHS[name]
    (n1, _), (n2, blocks) = g.stage1, g.stage2
    handover = Run(name, "stage1", FLAT, n1, (g.stage1_lr,))
    out = dict(init=source.initial(handover, g.stage1_lr),
               handover=source.scores(handover)[g.stage1_lr],
               single={}, stage2={}, short_budget=set())
    for c, (n, rates) in g.single.items():
        scores = {}
        for lr in rates:
            scores.update(source.scores(Run(name, "single", c, n, (lr,))))
        out["single"][c] = best(scores)
        if n != n1 + n2:
            out["short_budget"].add(c)
    for c, rates in blocks.items():
        out["stage2"][c] = best(source.scores(Run(name, "stage2", c, n2, tuple(rates))))
    s2 = {c: s for c, (_, s) in out["stage2"].items()}
    best_single = min(s for _, s in out["single"].values())
    best_two = min(s2.values())
    out["curvature_gain"] = 100 * (min(s2["c=0.2"], s2["c=0.3"]) - s2[FLAT]) / s2[FLAT]
    out["schedule_gain"] = 100 * (best_two - best_single) / best_single
    return out


def stats(name: str) -> dict:
    G = load_graph(GRAPHS[name].graph)
    r0, *_ = warm_start(G, 1.0)
    return dict(nodes=G.number_of_nodes(), edges=G.number_of_edges(),
                diameter=nx.diameter(G), span=float(r0.max() - r0.min()))


def latex(names: list[str], st: dict, res: dict) -> str:
    def row(label, cells):
        return f"{label} & " + " & ".join(cells) + r" \\"

    def block(key, curvatures):
        lines = []
        for c in curvatures:
            cells = []
            for n in names:
                s = res[n][key][c][1]
                cell = fmt(s)
                if s == min(v for _, v in res[n][key].values()):
                    cell = rf"\textbf{{{cell}}}"
                if key == "single" and c in res[n]["short_budget"]:
                    cell += r"$^\dagger$"
                cells.append(cell)
            lines.append(row(LABEL[c], cells))
        return lines

    def gain(v):
        return f"${v:+.1f}$" if round(v, 1) != 0 else "$0.0$"

    def heading(text):
        return rf"\multicolumn{{{len(names) + 1}}}{{@{{}}l}}{{\textit{{{text}}}}} \\"

    out = [row("", [SHORT[n] for n in names]), r"\midrule",
           row("$|V|$", [str(st[n]["nodes"]) for n in names]),
           row("$|E|$", [str(st[n]["edges"]) for n in names]),
           row("diameter", [str(st[n]["diameter"]) for n in names]),
           row("radial span $s$", [f"{st[n]['span']:.1f}" for n in names]),
           r"\midrule[\heavyrulewidth]",
           row(r"\textsc{Hydra} init.", [fmt(res[n]["init"]) for n in names]),
           r"\midrule", heading("Single curvature"),
           *block("single", list(GRAPHS[names[0]].single)),
           r"\midrule", heading("Two-stage"),
           row("handover", [fmt(res[n]["handover"]) for n in names]),
           *block("stage2", STAGE2),
           r"\midrule",
           row(r"curv.\ gain (\%)", [gain(res[n]["curvature_gain"]) for n in names]),
           row(r"sched.\ gain (\%)", [gain(res[n]["schedule_gain"]) for n in names])]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", action="store_true",
                    help="read the committed reference/ instead of raw runs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--results", type=Path, default=RESULTS)
    args = ap.parse_args()
    source = Reference() if args.reference else Results(args.device, args.results)

    names = list(GRAPHS)
    res = {n: collect(n, source) for n in names}
    st = {}
    for n in names:
        print(f"graph statistics: {n}", flush=True)
        st[n] = stats(n)

    def rates(chosen):
        return "  ".join(f"{c.split('=')[1]}:{lr:g}" for c, (lr, _) in chosen.items())

    print("\nselected learning rates (single curvature | second stage):")
    for n in names:
        print(f"  {n:16s} {rates(res[n]['single'])}  |  {rates(res[n]['stage2'])}")
    print("\n" + latex(names, st, res))


if __name__ == "__main__":
    main()
