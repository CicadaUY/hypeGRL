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

# Builders from polar coords. The hyperboloid uses a large max_norm so the
# comparison exposes its *genuine* distance behaviour (exact until r≈18, then
# Minkowski cancellation) rather than the StableLorentz radius clamp — at the
# default 1e3 that clamp caps r≤7.6 and would dominate any graph with larger
# radii (making the loss/AUC an artifact of the clamp, not the chart).
REPS = {
    "polar": lambda r, V: PolarRepresentation.from_polar(r, V),
    "ball": lambda r, V: BallRepresentation.from_polar(r, V),
    "hyperboloid": lambda r, V: HyperboloidRepresentation.from_polar(
        r, V, max_norm=1e18),
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


def _auc_str(rep, A):
    """AUC of a representation's decoded distances, or FAIL if non-finite."""
    D = rep.dist().detach().numpy()
    if not np.isfinite(D).all():
        n_bad = int((~np.isfinite(D)).sum())
        return f"AUC = FAIL ({n_bad} non-finite distances past r≈18)"
    return f"AUC = {_auc(D, A):.4f}"


# --------------------------------------------------------------------------- #
# Tier 1 — distance conditioning vs radius
# --------------------------------------------------------------------------- #
def tier1_conditioning():
    """Two failure modes, each with the metric that actually distinguishes them.

    - **Accuracy on well-separated pairs** (random directions): exposes the
      ball's *radial* saturation, which corrupts every large-r pair regardless
      of angle. Hyperboloid and polar stay exact here.
    - **Usability on near-coincident pairs** (same direction, Δr=1): exposes the
      hyperboloid's *angular* Minkowski cancellation, which returns NaN/negative
      distances past r≈18 — fatal for a loss. Ball stays finite (wrong but
      usable); polar stays finite too. (Accuracy is the wrong metric here: at
      large r even the 50-digit oracle can't resolve a same-direction pair,
      because the float64 direction is only unit to ~ε and ε·sinh²r swamps the
      O(1) inner product — the Stage-A fundamental limit, chart-independent.)
    """
    print("\n== Tier 1: distance conditioning — two failure modes ==")
    radii = [2, 5, 10, 12, 15, 18, 20, 25, 30, 40]

    # (a) accuracy, well-separated pairs — median relative error vs oracle
    acc = {name: [] for name in REPS}
    rng = np.random.default_rng(0)
    print("\n  (a) accuracy on well-separated pairs — median rel. distance error")
    print(f"  {'radius':>7} | {'polar':>11} {'ball':>11} {'hyperboloid':>11}")
    print("  " + "-" * 46)
    for rlevel in radii:
        r = rng.uniform(rlevel - 0.5, rlevel + 0.5, size=24)
        V = rng.standard_normal((24, 3))
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        ref = _oracle_distances(r, V)
        iu = np.triu_indices(24, k=1)
        for name, build in REPS.items():
            D = build(r, V).dist().detach().numpy()
            rel = np.abs(D[iu] - ref[iu]) / np.maximum(1.0, ref[iu])
            acc[name].append(np.median(np.nan_to_num(rel, nan=10.0, posinf=10.0)))
        print(f"  {rlevel:>7} | {acc['polar'][-1]:>11.2e} {acc['ball'][-1]:>11.2e} "
              f"{acc['hyperboloid'][-1]:>11.2e}")

    # (b) usability, near-coincident pairs — fraction of invalid (NaN/neg) distances
    inv = {name: [] for name in REPS}
    rng = np.random.default_rng(1)
    print("\n  (b) usability on near-coincident pairs — fraction NaN/negative")
    print(f"  {'radius':>7} | {'polar':>11} {'ball':>11} {'hyperboloid':>11}")
    print("  " + "-" * 46)
    for rlevel in radii:
        dirs = rng.standard_normal((12, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        V = np.repeat(dirs, 2, axis=0)
        r = np.empty(24)
        r[0::2], r[1::2] = rlevel, rlevel + 1.0
        pi, pj = np.arange(0, 24, 2), np.arange(1, 24, 2)      # within-pair
        for name, build in REPS.items():
            pd = build(r, V).dist().detach().numpy()[pi, pj]
            inv[name].append(float(np.mean(~np.isfinite(pd) | (pd < 0))))
        print(f"  {rlevel:>7} | {inv['polar'][-1]:>11.2f} {inv['ball'][-1]:>11.2f} "
              f"{inv['hyperboloid'][-1]:>11.2f}")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.5))
    for name in REPS:
        ax0.semilogy(radii, np.maximum(acc[name], 1e-17), marker="o", label=name)
        ax1.plot(radii, inv[name], marker="o", label=name)
    for ax in (ax0, ax1):
        ax.axvline(12.2, ls=":", c="gray", lw=1)
        ax.axvline(18, ls=":", c="gray", lw=1)
        ax.set_xlabel("hyperbolic radius r")
    ax0.set_ylabel("median relative distance error")
    ax0.set_title("(a) accuracy, well-separated pairs\n(ball saturates radially)")
    ax1.set_ylabel("fraction of NaN/negative distances")
    ax1.set_title("(b) usability, near-coincident pairs\n(hyperboloid cancels → NaN)")
    ax1.legend()
    fig.tight_layout()
    out = RESULTS / "representation_conditioning.png"
    fig.savefig(out, dpi=130)
    print(f"\n  wrote {out}")


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
    for name, build in REPS.items():
        print(f"    {name:<12} {_auc_str(build(r, V), A)}")

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
            for name, build in REPS.items():
                print(f"    {name:<12} {_auc_str(build(r, V), A)}")
        except Exception as e:  # noqa: BLE001
            print(f"  (OpenFlights AUC skipped: {e})")
    else:
        print("  (OpenFlights cache not found — run the D-Mercator fit first)")


# --------------------------------------------------------------------------- #
# Tier 3 — optimisation robustness: lr sweep on karate
# --------------------------------------------------------------------------- #
def tier3_lr_sweep():
    print("\n== Tier 3: FD-NLL refinement (karate) — 'final (Δloss, r=final max "
          "radius)' ==")
    print("   init: loss≈76, max radius≈16.1. Δ<0 = loss fell; r≫16 = the radii "
          "ran away (hyperboloid's e^r-scale coords are radially unstable and "
          "blow out to the max_norm ceiling, r≈42).")
    G = nx.karate_club_graph()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = dmercator_init(G, D=1, beta=None, random_state=0, d1_init="le")
    r, V, nodes = res["r"], res["V"], res["nodes"]
    A = nx.to_numpy_array(G, nodelist=nodes, weight=None)
    half_beta, R_hat = res["beta"] / 2.0, res["R_hat"]
    lrs = [0.01, 0.03, 0.1, 0.3, 1.0]
    print(f"  {'lr':>6} | " + " ".join(f"{n:>20}" for n in REPS))
    print("  " + "-" * 72)
    for lr in lrs:
        cells = []
        for name, build in REPS.items():
            rep = build(r, V)

            def loss_fn(rep_, A_t):
                return _fermi_dirac_nll(rep_.dist(), A_t, half_beta, R_hat)

            hist = riemannian_optimize(
                representation=rep, s_A=A, loss_fn=loss_fn,
                lr=lr, n_steps=100, log_every=0)["loss_history"]
            init, final = hist[0], hist[-1]
            rmax = float(rep.to_polar()[0].max())      # rep optimised in place
            cells.append("NaN" if not np.isfinite(final)
                         else f"{final:.0f} (Δ{final - init:+.0f}, r{rmax:.0f})")
        print(f"  {lr:>6} | " + " ".join(f"{c:>20}" for c in cells))


if __name__ == "__main__":
    tier1_conditioning()
    tier2_auc()
    tier3_lr_sweep()
