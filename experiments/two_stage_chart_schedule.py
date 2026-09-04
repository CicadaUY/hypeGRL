"""Two-stage chart schedule: a coarse phase in the tangent chart, a fine one in a
curved polar chart.

THE CLAIM UNDER TEST. A chart does not change where the minima are — every chart here
computes the same curvature -1 distance — but it does decide how far a step moves a
point, and the two things an embedding needs are not the same at the start and at the
end. Early on the configuration is far from any minimum and wants large angular
rearrangement, which the tangent chart's flat warp allows and the exact metric's
sinh(r) suppresses. Late on what remains is small local correction, which a warped
metric handles at a sane rate. So run the coarse phase in one chart, hand the
coordinates to another, and ask whether the pair reaches a stress that neither reaches
alone.

WHY THE CONTROL IS A CHART SWAP AND NOT "NO SWAP". Handing over builds a new parameter
on a new manifold, which discards Adam's moment estimates; a fresh optimiser state
lowers the loss by itself, whatever the chart. So the second stage runs a TANGENT arm
from the same handover state with the same restart. It differs from the curved arms in
the chart alone, and both get the same step budget, so nothing wins on freshness or on
spend.

WHY EVERY ARM RE-SEARCHES THE RATE, OVER ONE SHARED GRID. The angular coordinate moves
by about lr/w_c(r) per step, and the warps differ by orders of magnitude at the radii a
deep graph reaches, so a rate carried across the handover is not the same step. Each arm
therefore gets its own rate. The grid must span the UNION of the arms' usable ranges,
not each arm's expected range: the winning curved rate here (0.005) and the winning
tangent rate (0.0001) are a factor of 50 apart, and a grid centred on either one alone
reports the other as far worse than it is.

REPRODUCIBILITY, AND THE LIMIT OF IT. The pipeline carries no randomness: the HYDRA warm
start is a closed-form eigendecomposition (a seed changes nothing -- verified) and the
stress loss is full-batch with no sampling, so re-running on one machine reproduces
every number bit for bit. It does NOT reproduce across devices, and not by a little:
CPU and CUDA differ by one unit in the last place at step 1, and 6000 steps of descent
amplify that to 5.9% on caterpillar(40,4). Nothing is wrong when that happens -- the
trajectory is simply chaotic, so a stress value is only meaningful to a few significant
figures, and comparisons must come from arms run on the same device. Record the device
with the result; this script writes it into the json.

Usage:
    python two_stage_chart_schedule.py                     # caterpillar(40,4)
    python two_stage_chart_schedule.py --graph "caterpillar(20,9)"
    python two_stage_chart_schedule.py --graph fabaceae_sub --n-fine 50000
    python two_stage_chart_schedule.py --curvatures 0.1,0.3,0.5
    python two_stage_chart_schedule.py --device cpu        # comparable to CPU runs
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from sklearn.metrics import average_precision_score

from hypegrl.embedders.hydra_plus import HydraPlusEmbedder, _stress_loss_from_dist
from hypegrl.inference.riemannian_optimizer import riemannian_optimize
from hypegrl.representations import (
    CurvedPolarRepresentation,
    TangentRepresentation,
)

RESULTS = Path(__file__).resolve().parent / "results"
PHYLOGENY = Path(__file__).resolve().parent / "data" / "phylogeny"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Spans both arms' usable ranges; see the module docstring on why one shared grid.
RATES = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.3]
N_COARSE, LR_COARSE, N_FINE = 6000, 0.03, 24000
CURVATURES = [0.3]


def caterpillar(spine: int, leaves: int) -> nx.Graph:
    """A path of ``spine`` nodes carrying ``leaves`` each.

    Depth is how a tree reaches a large radial span at unit curvature, but a balanced
    tree grows exponentially in depth, so a large diameter costs an unusable node
    count. A caterpillar buys diameter linearly: (40, 4) is 200 nodes with diameter 41
    and a radial span near 20, which is where the charts visibly disagree.
    """
    G = nx.path_graph(spine)
    nxt = spine
    for i in range(spine):
        for _ in range(leaves):
            G.add_edge(i, nxt)
            nxt += 1
    return G


def read_edgelist(path) -> nx.Graph:
    """An undirected edge list, relabelled to ``0..N-1`` in sorted-label order.

    The relabelling is not cosmetic. Node and edge order fix the row order of every
    matrix downstream, so a loader that emits them in an arbitrary order (a set
    iteration, say) makes two runs of the same experiment incomparable while looking
    identical. Sorting makes the graph a function of the file alone.
    """
    G = nx.read_edgelist(path)
    order = {name: i for i, name in enumerate(sorted(G.nodes()))}
    H = nx.Graph()
    H.add_nodes_from(range(len(order)))
    H.add_edges_from(sorted(tuple(sorted((order[a], order[b]))) for a, b in G.edges()))
    return H


def load_graph(name: str) -> nx.Graph:
    """A synthetic spec, a vendored phylogeny name, or a path to an edge list.

    ``caterpillar(spine,leaves)`` and ``balanced_tree(branch,depth)`` are generated;
    a bare name such as ``fabaceae_sub`` is one of the vendored trees in
    ``experiments/data/phylogeny/``; anything with a suffix is read as a path. A real
    graph therefore enters the experiment the same way a generated one does.
    """
    if "(" not in name:
        path = Path(name)
        return read_edgelist(path if path.suffix else PHYLOGENY / f"{name}.edgelist")
    head, _, rest = name.partition("(")
    a, b = (int(x) for x in rest.rstrip(")").split(","))
    if head == "caterpillar":
        return caterpillar(a, b)
    if head == "balanced_tree":
        return nx.balanced_tree(a, b)
    raise ValueError(f"unsupported graph {name!r}")


def warm_start(G: nx.Graph, curvature: float):
    """HYDRA's closed-form spectral solution, read out in polar coordinates."""
    emb = HydraPlusEmbedder(dim=2, curvature=curvature, n_steps=0).fit(G)
    X = emb.embeddings()
    norm = np.linalg.norm(X, axis=1).clip(0.0, 1.0 - 1e-15)
    theta = np.arctan2(X[:, 1], X[:, 0])
    return (2.0 * np.arctanh(norm),
            np.stack([np.cos(theta), np.sin(theta)], axis=-1),
            float(emb._curvature_fitted),
            emb.nodes())


def build(chart: str, r, v, device: str = DEFAULT_DEVICE):
    """``"tangent"``, or ``"c=<chart_curvature>"`` for the curved polar chart."""
    r = torch.as_tensor(r)
    v = torch.as_tensor(v)
    if chart == "tangent":
        return TangentRepresentation.from_polar(r, v, device=device)
    return CurvedPolarRepresentation.from_polar(
        r, v, device=device, chart_curvature=float(chart.split("=")[1]))


def refine(chart, r, v, target, mask, lr, n_steps, device: str = DEFAULT_DEVICE):
    """One optimisation; returns ``(final stress, representation, loss history)``.

    The final stress averages the last 100 steps rather than reading the last one,
    which is noisy at a large rate. A diverged run returns ``inf`` so it sorts last
    instead of raising.
    """
    rep = build(chart, r, v, device)
    history = np.asarray(riemannian_optimize(
        representation=rep, s_A=target,
        loss_fn=lambda rep_, s_A_: _stress_loss_from_dist(rep_.dist(), s_A_, mask),
        lr=lr, n_steps=n_steps, log_every=0, device=device,
    )["loss_history"], dtype=float)
    if not np.isfinite(history[-100:]).all():
        return float("inf"), rep, history
    return float(history[-100:].mean()), rep, history


def edge_average_precision(rep, A) -> float:
    """AP of edge existence ranked by proximity — the task behind the stress."""
    with torch.no_grad():
        d = rep.dist().cpu().numpy()
    if not np.isfinite(d).all():
        return float("nan")
    iu = np.triu_indices_from(A, k=1)
    return float(average_precision_score(A[iu], -d[iu]))


def sweep(chart, r, v, target, mask, A, rates, n_steps, curves, device):
    """Run one chart at every rate; return the row for its best."""
    best = None
    for lr in rates:
        t0 = time.perf_counter()
        stress, rep, history = refine(chart, r, v, target, mask, lr, n_steps, device)
        curves[f"{chart}__lr{lr:.10g}"] = history
        row = dict(chart=chart, lr=lr, stress=stress,
                   average_precision=edge_average_precision(rep, A))
        shown = f"{stress:.1f}" if np.isfinite(stress) else "diverged"
        print(f"    {chart:>9}  lr={lr:<8g} {shown:>12}"
              f"   ({time.perf_counter() - t0:.0f}s)", flush=True)
        if best is None or stress < best["stress"]:
            best = row
    best["on_grid_edge"] = best["lr"] in (rates[0], rates[-1])
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", default="caterpillar(40,4)")
    ap.add_argument("--curvature", type=float, default=1.0,
                    help="curvature of the embedding space, for HYDRA's target scaling")
    ap.add_argument("--curvatures", default=",".join(str(c) for c in CURVATURES),
                    help="chart curvatures for the fine phase (comma-separated)")
    ap.add_argument("--rates", default=",".join(str(x) for x in RATES))
    ap.add_argument("--n-coarse", type=int, default=N_COARSE)
    ap.add_argument("--lr-coarse", type=float, default=LR_COARSE)
    ap.add_argument("--n-fine", type=int, default=N_FINE)
    ap.add_argument("--device", default=DEFAULT_DEVICE,
                    help="results are comparable only within one device; see the "
                         "module docstring")
    args = ap.parse_args()
    device = args.device

    rates = [float(x) for x in args.rates.split(",")]
    charts = ["tangent"] + [f"c={c}" for c in args.curvatures.split(",")]

    G = load_graph(args.graph)
    n = G.number_of_nodes()
    D = np.array(nx.floyd_warshall_numpy(G), dtype=np.float64)
    mask = torch.as_tensor(np.triu(np.ones((n, n), dtype=bool), k=1)).to(device)
    r0, v0, k, nodes = warm_start(G, args.curvature)
    A = nx.to_numpy_array(G, nodelist=nodes, weight=None)
    target = D * np.sqrt(k)

    print(f"=== {args.graph}  N={n}  diameter={nx.diameter(G)}  k={k:g}  "
          f"device={device} ===")
    print(f"warm start: r in [{r0.min():.2f}, {r0.max():.2f}]  "
          f"span {r0.max() - r0.min():.2f}", flush=True)

    curves = {}
    coarse_stress, coarse_rep, coarse_history = refine(
        "tangent", r0, v0, target, mask, args.lr_coarse, args.n_coarse, device)
    curves["coarse"] = coarse_history
    r1, v1 = coarse_rep.to_polar()
    print(f"\ncoarse: tangent, lr={args.lr_coarse:g}, {args.n_coarse} steps  "
          f"{coarse_history[0]:.0f} -> {coarse_stress:.0f}   "
          f"r in [{float(r1.min()):.2f}, {float(r1.max()):.2f}]", flush=True)

    print(f"\nfine: {args.n_fine} steps from that state, every chart over the same "
          f"{len(rates)} rates", flush=True)
    rows = [sweep(c, r1, v1, target, mask, A, rates, args.n_fine, curves, device)
            for c in charts]

    RESULTS.mkdir(exist_ok=True)
    # The device is part of the filename because it is part of the result: the same
    # sweep on CPU and on CUDA gives different stresses (see the module docstring), so
    # one stem for both would silently overwrite one run's numbers with the other's.
    tag = Path(args.graph).stem.replace("(", "").replace(")", "").replace(",", "-")
    stem = RESULTS / f"two_stage_chart_schedule_{tag}_{device}"
    json.dump(dict(graph=args.graph, n_nodes=n, curvature=k, rates=rates,
                   device=device, n_coarse=args.n_coarse, lr_coarse=args.lr_coarse,
                   coarse_stress=coarse_stress, n_fine=args.n_fine, best=rows),
              open(stem.with_suffix(".json"), "w"), indent=1)
    np.savez(stem.with_suffix(".npz"), **curves)

    control = next(r for r in rows if r["chart"] == "tangent")
    print(f"\n{'chart':>9}{'best lr':>10}{'stress':>12}{'AP':>8}"
          f"{'vs control':>12}")
    for row in sorted(rows, key=lambda r: r["stress"]):
        rel = row["stress"] / control["stress"]
        print(f"{row['chart']:>9}{row['lr']:>10g}{row['stress']:>12.1f}"
              f"{row['average_precision']:>8.4f}{rel:>11.2f}x"
              f"{'   [grid edge]' if row['on_grid_edge'] else ''}")
    print(f"\nwrote {stem}.json")


if __name__ == "__main__":
    main()
