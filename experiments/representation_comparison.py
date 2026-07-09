# -*- coding: utf-8 -*-
"""
Representation comparison: does the polar chart's better conditioning yield
measurable gains over ball / hyperboloid *beyond* the outright failure cases?

Three tiers (Stage D of the representation refactor):

1. **Distance conditioning** — pairwise-distance relative error vs a 50-digit
   mpmath oracle, swept over radius. Pure chart conditioning, no optimisation.
2. **Downstream fidelity** — reconstruction AUC of the decoded distances
   (edges should be closer than non-edges), per chart, on a moderate-radius
   graph (karate) and a large-radius one (OpenFlights, from the cached init).
3. **Optimisation robustness** — refine the Fermi-Dirac NLL in each chart over
   an lr sweep; report final loss and whether it stayed finite.

Run:
    python experiments/representation_comparison.py
"""
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp
import networkx as nx
import numpy as np
import torch

REPO = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, REPO)

from hypegrl.embedders._dmercator_init import dmercator_init  # noqa: E402
from hypegrl.embedders.dmercator import _fermi_dirac_nll  # noqa: E402
from hypegrl.inference.riemannian_optimizer import riemannian_optimize  # noqa: E402
from hypegrl.representations import (  # noqa: E402
    BallRepresentation,
    HyperboloidRepresentation,
    PolarRepresentation,
)

torch.set_default_dtype(torch.float64)
mp.mp.dps = 50
REPS = {
    "polar": PolarRepresentation,
    "ball": BallRepresentation,
    "hyperboloid": HyperboloidRepresentation,
}
RESULTS = Path(REPO) / "experiments" / "results"


def _oracle_distances(r, V):
    """Exact (N, N) hyperbolic distances at 50 digits, from polar coords."""
    n = len(r)
    D = np.zeros((n, n))
    for i in range(n):
        ri = mp.mpf(float(r[i]))
        for j in range(i + 1, n):
            rj = mp.mpf(float(r[j]))
            cos_ang = sum(mp.mpf(float(a)) * mp.mpf(float(b))
                          for a, b in zip(V[i], V[j]))
            val = mp.cosh(ri) * mp.cosh(rj) - mp.sinh(ri) * mp.sinh(rj) * cos_ang
            D[i, j] = D[j, i] = float(mp.acosh(val if val > 1 else mp.mpf(1)))
    return D


def _auc(dist, A):
    """Reconstruction AUC: edges should be *closer* (smaller distance).

    Mann-Whitney form on the similarity ``-dist``: AUC = P(sim_edge > sim_non).
    """
    iu = np.triu_indices_from(A, k=1)
    sim = -dist[iu]                 # higher similarity ⇒ more edge-like
    y = A[iu] > 0
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(sim, kind="mergesort")
    ranks = np.empty(len(sim))
    ranks[order] = np.arange(1, len(sim) + 1)      # 1-indexed ascending
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


# --------------------------------------------------------------------------- #
# Tier 1 — distance conditioning vs radius
# --------------------------------------------------------------------------- #
def tier1_conditioning():
    print("\n== Tier 1: pairwise-distance relative error vs mpmath oracle ==")
    print(f"  {'radius':>7} | {'polar':>11} {'ball':>11} {'hyperboloid':>11}")
    print("  " + "-" * 46)
    radii = [2, 5, 10, 12, 15, 18, 20, 25, 30, 40]
    curves = {name: [] for name in REPS}
    rng = np.random.default_rng(0)
    for rlevel in radii:
        n = 24
        r = rng.uniform(rlevel - 0.5, rlevel + 0.5, size=n)
        V = rng.standard_normal((n, 3))
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        ref = _oracle_distances(r, V)
        iu = np.triu_indices(n, k=1)
        row = {}
        for name, cls in REPS.items():
            D = cls.from_polar(r, V).dist().detach().numpy()
            rel = np.abs(D[iu] - ref[iu]) / np.maximum(1.0, ref[iu])
            row[name] = np.median(rel)
            curves[name].append(np.median(rel))
        print(f"  {rlevel:>7} | {row['polar']:>11.2e} {row['ball']:>11.2e} "
              f"{row['hyperboloid']:>11.2e}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in REPS:
        ax.semilogy(radii, np.maximum(curves[name], 1e-17), marker="o", label=name)
    ax.axvline(12.2, ls=":", c="gray", lw=1)
    ax.axvline(18, ls=":", c="gray", lw=1)
    ax.set_xlabel("hyperbolic radius r")
    ax.set_ylabel("median relative distance error")
    ax.set_title("Distance conditioning by chart (vs 50-digit oracle)")
    ax.legend()
    fig.tight_layout()
    out = RESULTS / "representation_conditioning.png"
    fig.savefig(out, dpi=130)
    print(f"  wrote {out}")


# --------------------------------------------------------------------------- #
# Tier 2 — reconstruction AUC per chart
# --------------------------------------------------------------------------- #
def _dmercator_native(G):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = dmercator_init(G, D=1, beta=None, random_state=0, d1_init="le")
    return res["r"], res["V"], res["nodes"]


def tier2_auc():
    print("\n== Tier 2: reconstruction AUC of decoded distances, per chart ==")
    # karate: moderate radius
    G = nx.karate_club_graph()
    r, V, nodes = _dmercator_native(G)
    A = nx.to_numpy_array(G, nodelist=nodes, weight=None)
    print(f"  {'karate':<20} r∈[{r.min():.1f},{r.max():.1f}]")
    for name, cls in REPS.items():
        D = cls.from_polar(r, V).dist().detach().numpy()
        print(f"    {name:<12} AUC = {_auc(D, A):.4f}")

    # OpenFlights: large radius, from the cached init (no 13-min re-run)
    npz = RESULTS / "embeddings" / "openflights_dmercator_native.npz"
    if npz.exists():
        d = np.load(npz, allow_pickle=True)
        r, V, nodes = d["r"], d["v"], d["nodes"]
        try:
            from experiments.datasets import openflights_graph
            G = openflights_graph()
            A = nx.to_numpy_array(G, nodelist=list(nodes), weight=None)
            print(f"  OpenFlights (N={len(r)})     r∈[{r.min():.1f},{r.max():.1f}]")
            for name, cls in REPS.items():
                D = cls.from_polar(r, V).dist().detach().numpy()
                print(f"    {name:<12} AUC = {_auc(D, A):.4f}")
        except Exception as e:  # noqa: BLE001
            print(f"  (OpenFlights AUC skipped: {e})")
    else:
        print("  (OpenFlights cache not found — run the D-Mercator fit first)")


# --------------------------------------------------------------------------- #
# Tier 3 — optimisation robustness: lr sweep on karate
# --------------------------------------------------------------------------- #
def tier3_lr_sweep():
    print("\n== Tier 3: FD-NLL refinement (karate) — final loss and change "
          "'final (Δ=final−init)'; Δ<0 means the loss decreased ==")
    G = nx.karate_club_graph()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = dmercator_init(G, D=1, beta=None, random_state=0, d1_init="le")
    r, V, nodes = res["r"], res["V"], res["nodes"]
    A = nx.to_numpy_array(G, nodelist=nodes, weight=None)
    half_beta, R_hat = res["beta"] / 2.0, res["R_hat"]
    lrs = [0.01, 0.03, 0.1, 0.3, 1.0]
    print(f"  {'lr':>6} | " + " ".join(f"{n:>18}" for n in REPS))
    print("  " + "-" * 66)
    for lr in lrs:
        cells = []
        for name, cls in REPS.items():
            rep = cls.from_polar(r, V)

            def loss_fn(rep_, A_t):
                return _fermi_dirac_nll(rep_.dist(), A_t, half_beta, R_hat)

            hist = riemannian_optimize(
                representation=rep, s_A=A, loss_fn=loss_fn,
                lr=lr, n_steps=100, log_every=0)["loss_history"]
            init, final = hist[0], hist[-1]
            cells.append("NaN" if not np.isfinite(final)
                         else f"{final:.1f} ({final - init:+.1f})")
        print(f"  {lr:>6} | " + " ".join(f"{c:>18}" for c in cells))


if __name__ == "__main__":
    tier1_conditioning()
    tier2_auc()
    tier3_lr_sweep()
