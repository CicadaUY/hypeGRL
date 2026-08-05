# -*- coding: utf-8 -*-
"""
Recovering a random hyperbolic geometry from a Fermi-Dirac-sampled graph.

Ground truth -> graph -> re-embed -> compare, entirely through the public
library API:

1. Sample N points uniformly at random (by area) on the Poincaré disk —
   the "ground-truth" embedding, wrapped as a
   :class:`~hypegrl.embedders.precomputed.PrecomputedEmbedder` so the rest
   of the library can treat it like any fitted embedder.
2. Sample a graph from that embedding under the Fermi-Dirac connection
   model (:class:`~hypegrl.generation.fermi_dirac.FermiDiracGenerator`).
3. Re-embed the sampled graph from scratch with a generative method
   (HyperMap or D-Mercator) — it sees only the graph, never the
   ground-truth coordinates.
4. Compare the recovered embedding against the ground truth:
   - Pairwise hyperbolic distance correlation (Pearson + Spearman). This is
     the principled comparison: the Poincaré disk's isometry group
     (rotations + Möbius translations) can move the recovered points
     anywhere without changing a single pairwise distance, so a
     generative fit is only *identifiable up to isometry* — raw-coordinate
     or Procrustes comparisons would penalize a perfectly recovered
     geometry that simply landed in a different orientation. Pairwise
     distance is the isometry-invariant quantity a fit can actually be
     expected to recover.
   - Radius correlation (Pearson): the Fermi-Dirac / E-PSO model ties
     radius to degree, so both embeddings should agree on *which* nodes
     are central even though their raw coordinates differ.
   - Reconstruction AUC: how well the recovered embedding's own decoder
     explains the sampled graph's edges — a sanity check that the fit
     converged, independent of the ground truth.
   - A side-by-side Poincaré disk plot, coloured by degree, since a visual
     check often catches what a single summary statistic hides (e.g. a
     globally rotated but locally faithful layout).

Run:
    python experiments/fermi_dirac_recovery_demo.py
    python experiments/fermi_dirac_recovery_demo.py --method dmercator --n 150
    python experiments/fermi_dirac_recovery_demo.py --sampling euclidean  # contrast
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

REPO = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, REPO)

from hypegrl.embedders.precomputed import PrecomputedEmbedder  # noqa: E402
from hypegrl.generation.fermi_dirac import FermiDiracGenerator  # noqa: E402
from hypegrl.manifolds.poincare import POINCARE_BALL  # noqa: E402

RESULTS = Path(REPO) / "experiments" / "results"


# ---------------------------------------------------------------------------
# Ground-truth geometry
# ---------------------------------------------------------------------------

def sample_uniform_disk(
    n: int, max_radius: float, rng: np.random.Generator
) -> np.ndarray:
    """
    ``(n, 2)`` points sampled uniformly by *Euclidean area* on the Poincaré
    disk, out to Euclidean radius ``max_radius`` (``< 1``).

    Uniform by Euclidean area means the Euclidean radius ``rho`` has density
    proportional to ``rho``, so ``rho = max_radius * sqrt(U)``, ``U ~
    Uniform(0, 1)``. This is uniform in the *chart*, not in hyperbolic
    measure — see :func:`sample_uniform_hyperbolic_disk` and the warning
    below.

    .. warning::
       This is **not** uniform with respect to hyperbolic area, and that
       matters more than it looks. The hyperbolic area element is
       ``dA = sinh(rho) dρ dθ``, which grows exponentially with ``rho`` —
       so a fixed Euclidean cap like ``max_radius=0.9`` (hyperbolic radius
       ``2 artanh(0.9) ≈ 2.94``) restricts every point to a *tiny* sliver
       near the hyperbolic origin, giving almost no radius spread and
       hence almost no degree heterogeneity once you sample a graph from
       it. HyperMap and D-Mercator are fit under an implicit
       popularity-similarity generative model that *assumes* a roughly
       hyperbolic-uniform (or quasi-uniform) radial density with real
       heterogeneity; feeding them a graph generated from this narrow,
       near-degree-homogeneous regime is a genuine model mismatch, not
       just a harder instance of the same problem — and shows up as
       noticeably weaker recovery than :func:`sample_uniform_hyperbolic_disk`
       at the same ``n``. Kept here mainly as that contrast case; prefer
       :func:`sample_uniform_hyperbolic_disk` for anything meant to look
       like a "typical" hyperbolic random graph.
    """
    theta = rng.uniform(0.0, 2 * np.pi, n)
    rho = max_radius * np.sqrt(rng.uniform(0.0, 1.0, n))
    return np.stack([rho * np.cos(theta), rho * np.sin(theta)], axis=1)


def sample_uniform_hyperbolic_disk(
    n: int, R: float, rng: np.random.Generator, alpha: float = 1.0
) -> np.ndarray:
    r"""
    ``(n, 2)`` points sampled uniformly with respect to the **hyperbolic**
    area measure, out to hyperbolic radius ``R``, returned as Poincaré-ball
    coordinates.

    On :math:`\mathbb{H}^2` (curvature :math:`-1`) the area element in polar
    coordinates is :math:`dA = \sinh(\rho)\, d\rho\, d\theta` — it grows
    exponentially in :math:`\rho`, unlike the Euclidean :math:`\rho\, d\rho\,
    d\theta` used by :func:`sample_uniform_disk`. So true-uniform sampling
    needs radial density :math:`\propto \sinh(\rho)` on :math:`[0, R]`, not
    :math:`\propto \rho`. Its CDF inverts in closed form:

    .. math::

        F(\rho) = \frac{\cosh(\rho) - 1}{\cosh(R) - 1}
        \quad\Longrightarrow\quad
        \rho = \operatorname{arccosh}\!\big(1 + U\,(\cosh(R) - 1)\big),
        \quad U \sim \mathrm{Uniform}(0, 1).

    ``alpha`` generalises this to density :math:`\propto \sinh(\alpha
    \rho)`, the same control knob used in the hyperbolic random graph
    literature (Krioukov et al., 2010) to tune how sharply points — and so
    node degrees — concentrate: ``alpha=1`` is the exact uniform measure
    above; ``alpha>1`` pushes more mass toward the boundary ``R``, producing
    a heavier-tailed (more strongly power-law) degree distribution when a
    graph is later sampled from these points; ``alpha<1`` flattens it back
    toward more homogeneous degrees.

    Because hyperbolic area grows exponentially, samples concentrate close
    to the boundary radius ``R`` even at ``alpha=1`` — expect most Euclidean
    norms near 1 and only a few points near the origin. That's not a bug:
    those few central points are exactly the high-degree hubs a
    hierarchical embedding is supposed to recover.

    Parameters
    ----------
    n:
        Number of points.
    R:
        Hyperbolic radius cutoff (unbounded in principle; ``R`` around 5-6
        already gives a clearly heterogeneous degree distribution for a
        graph of a few hundred nodes — see ``run()``'s ``--sampling
        hyperbolic`` default).
    rng:
        NumPy random generator.
    alpha:
        Radial concentration parameter, ``alpha > 0`` (default ``1.0``,
        exact hyperbolic-uniform).

    Returns
    -------
    ``(n, 2)`` Poincaré-ball coordinates, via ``rho -> tanh(rho / 2)``.
    """
    theta = rng.uniform(0.0, 2 * np.pi, n)
    u = rng.uniform(0.0, 1.0, n)
    rho = np.arccosh(1.0 + u * (np.cosh(alpha * R) - 1.0)) / alpha
    euclidean_radius = np.tanh(rho / 2.0)
    return np.stack(
        [euclidean_radius * np.cos(theta), euclidean_radius * np.sin(theta)], axis=1
    )


def hyperbolic_radius(X: np.ndarray) -> np.ndarray:
    """Exact H^2 radius ``2 * artanh(||x||)`` of Poincaré-ball points ``X``."""
    norm = np.linalg.norm(X, axis=1)
    return 2.0 * np.arctanh(np.clip(norm, 0.0, 1.0 - 1e-12))


def pairwise_ball_distance(X: np.ndarray) -> np.ndarray:
    """``(N, N)`` exact Poincaré-ball distance matrix for ball coords ``X``."""
    X_t = torch.as_tensor(X, dtype=torch.float64)
    return POINCARE_BALL.dist(X_t.unsqueeze(1), X_t.unsqueeze(0)).numpy()


# ---------------------------------------------------------------------------
# Re-embedding
# ---------------------------------------------------------------------------

def build_embedder(method: str, d: int = 2):
    if method == "hypermap":
        from hypegrl.embedders.hypermap import HyperMapEmbedder
        # verbose_init=False: the greedy init otherwise prints one line per
        # node placed, which floods stdout for anything but a toy graph.
        return HyperMapEmbedder(d=d, n_steps=300, log_every=0, verbose_init=False)
    if method == "dmercator":
        # D-Mercator's bundled init calls np.trapz, removed from NumPy>=2.0
        # in favour of np.trapezoid. Pre-existing library/NumPy-version
        # mismatch, unrelated to this script's logic — shim it here rather
        # than letting the run fail on an unrelated AttributeError.
        if not hasattr(np, "trapz"):
            np.trapz = np.trapezoid
        from hypegrl.embedders.dmercator import DMercatorEmbedder
        return DMercatorEmbedder(d=d, n_steps=300)
    raise ValueError(f"Unknown method {method!r}; choose 'hypermap' or 'dmercator'.")


def align_to_reference_order(
    values: np.ndarray, embedder_nodes: list, reference_nodes: list
) -> np.ndarray:
    """
    Reindex an array from ``embedder_nodes`` row order into
    ``reference_nodes`` row order (both are permutations of the same node
    labels — needed because e.g. HyperMap/D-Mercator reorder nodes by
    degree). Handles a per-node ``(N,)`` array or a symmetric ``(N, N)``
    matrix.
    """
    position = {node: i for i, node in enumerate(embedder_nodes)}
    order = [position[node] for node in reference_nodes]
    if values.ndim == 1:
        return values[order]
    return values[np.ix_(order, order)]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def disk_display_coords(
    radius: np.ndarray, angle: np.ndarray, max_display: float = 0.97
) -> np.ndarray:
    """
    ``(N, 2)`` Cartesian points on the disk for *display only*: ``angle`` is
    kept exact, but ``radius`` is min-max rescaled into ``[0, max_display]``.

    Two reasons this is necessary, not cosmetic:

    1. **Ball saturation.** ``tanh(r/2) -> 1`` as the exact hyperbolic radius
       ``r`` grows, so raw Poincaré-ball coordinates visually collapse onto
       the boundary once ``r ≳ 12`` — exactly the regime HyperMap/D-Mercator
       place well-connected hubs in on graphs of even moderate size.
    2. **The absolute radius scale is not comparable across panels even
       without saturation.** The recovered embedding's radius is set by the
       fitting method's own generative assumptions (e.g. HyperMap's
       ``m, L, γ, T``), not by the arbitrary ``max_radius`` used to sample
       the ground truth — a Fermi-Dirac fit is identifiable only up to
       isometry *and* this radius normalisation, never in absolute units.
       ``radius_pearson`` in the printed results already accounts for this
       (it correlates the two radius vectors, scale and all); this plot
       just makes the same relative structure visible.

    So only the *rank* of radius (and the angle) is meaningful to compare
    panel to panel — which is exactly what this rescaling preserves.
    """
    radius = np.asarray(radius, dtype=np.float64)
    span = radius.max() - radius.min()
    radius_norm = (radius - radius.min()) / span if span > 0 else np.zeros_like(radius)
    radius_display = radius_norm * max_display
    return np.stack(
        [radius_display * np.cos(angle), radius_display * np.sin(angle)], axis=1
    )


def _scatter_disk(ax, X: np.ndarray, colour_values: np.ndarray, title: str) -> None:
    ax.add_artist(plt.Circle((0, 0), 1.0, fill=False, color="black", linewidth=1))
    sc = ax.scatter(
        X[:, 0], X[:, 1], c=colour_values, cmap="viridis",
        s=25, edgecolors="k", linewidths=0.3,
    )
    plt.colorbar(sc, ax=ax, label="degree", shrink=0.8)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.set_title(title)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run(
    n: int,
    method: str,
    sampling: str,
    max_radius: float,
    R: float,
    alpha: float,
    r: Optional[float],
    t: Optional[float],
    seed: int,
    out_dir: Path,
) -> dict:
    rng = np.random.default_rng(seed)

    # 1. Ground-truth embedding.
    if sampling == "hyperbolic":
        X_true = sample_uniform_hyperbolic_disk(n, R, rng, alpha=alpha)
        r = 0.9 * R if r is None else r      # a bit inside the sampling cutoff
        t = 0.15 if t is None else t
    elif sampling == "euclidean":
        X_true = sample_uniform_disk(n, max_radius, rng)
        r = 0.6 if r is None else r
        t = 0.15 if t is None else t
    else:
        raise ValueError(
            f"Unknown sampling {sampling!r}; choose 'hyperbolic' or 'euclidean'."
        )

    true_radius_stats = hyperbolic_radius(X_true)
    print(
        f"Ground-truth hyperbolic radius: min={true_radius_stats.min():.2f} "
        f"mean={true_radius_stats.mean():.2f} max={true_radius_stats.max():.2f} "
        f"std={true_radius_stats.std():.2f}  (sampling={sampling!r}, r={r}, t={t})"
    )

    true_nodes = list(range(n))
    ground_truth = PrecomputedEmbedder(X_true, nodes=true_nodes)

    # 2. Sample a graph under the Fermi-Dirac connection model.
    generator = FermiDiracGenerator(ground_truth, r=r, t=t, random_state=seed)
    G = generator.sample(n_graphs=1)[0]
    mean_degree = 2 * G.number_of_edges() / G.number_of_nodes()
    print(
        f"Sampled graph: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges, mean degree {mean_degree:.2f}"
    )

    # 3. Re-embed from the graph alone — X_true is never passed in.
    embedder = build_embedder(method).fit(G)
    recovered_nodes = embedder.nodes()

    # 4a. Pairwise-distance correlation (the isometry-invariant comparison).
    D_true = pairwise_ball_distance(X_true)
    rep = embedder.embeddings_representation()
    D_recovered_native = (
        rep.dist().detach().cpu().numpy() if rep is not None
        else pairwise_ball_distance(embedder.embeddings())
    )
    D_recovered = align_to_reference_order(
        D_recovered_native, recovered_nodes, true_nodes
    )

    iu = np.triu_indices(n, k=1)
    d_true, d_rec = D_true[iu], D_recovered[iu]
    dist_pearson, _ = pearsonr(d_true, d_rec)
    dist_spearman, _ = spearmanr(d_true, d_rec)

    # 4b. Radius correlation.
    radius_true = hyperbolic_radius(X_true)
    radius_recovered_native = (
        rep.to_polar()[0].detach().cpu().numpy() if rep is not None
        else hyperbolic_radius(embedder.embeddings())
    )
    radius_recovered = align_to_reference_order(
        radius_recovered_native, recovered_nodes, true_nodes
    )
    radius_pearson, _ = pearsonr(radius_true, radius_recovered)

    degree = np.array([G.degree(node) for node in true_nodes], dtype=np.float64)
    degree_radius_true, _ = pearsonr(degree, radius_true)
    degree_radius_recovered, _ = pearsonr(degree, radius_recovered)
    # 4c. Reconstruction AUC: does the recovered embedding explain G at all?
    A_recovered_order = np.array([
        [1.0 if G.has_edge(u, v) else 0.0 for v in recovered_nodes]
        for u in recovered_nodes
    ])
    P_recovered = embedder.decode(rep if rep is not None else embedder.embeddings())
    auc = roc_auc_score(A_recovered_order[iu], P_recovered[iu])

    results = {
        "n": n,
        "sampling": sampling,
        "edges": G.number_of_edges(),
        "mean_degree": mean_degree,
        "max_degree": int(degree.max()),
        "degree_std": float(degree.std()),
        "method": method,
        "distance_pearson": dist_pearson,
        "distance_spearman": dist_spearman,
        "radius_pearson": radius_pearson,
        "degree_radius_true_pearson": degree_radius_true,
        "degree_radius_recovered_pearson": degree_radius_recovered,
        "reconstruction_auc": auc,
    }
    print("\nResults")
    print("-------")
    for key, value in results.items():
        print(f"{key:32s}: {value}")

    # 5. Side-by-side plot, coloured by (true) degree. Radius is rescaled
    # per panel for display — see disk_display_coords for why the absolute
    # scale is not comparable across panels in the first place.
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))

    angle_true = np.arctan2(X_true[:, 1], X_true[:, 0])
    _scatter_disk(
        axes[0], disk_display_coords(radius_true, angle_true), degree,
        "Ground truth (uniform on disk)",
    )

    X_recovered_native = embedder.embeddings()  # native (recovered_nodes) order
    angle_recovered_native = np.arctan2(
        X_recovered_native[:, 1], X_recovered_native[:, 0]
    )
    degree_recovered_order = np.array([G.degree(node) for node in recovered_nodes])
    _scatter_disk(
        axes[1],
        disk_display_coords(radius_recovered_native, angle_recovered_native),
        degree_recovered_order,
        f"Recovered ({method}) — radius rescaled for display, see docstring",
    )
    fig.suptitle(
        f"N={n}, |E|={G.number_of_edges()}  |  "
        f"distance corr (Pearson/Spearman) = "
        f"{dist_pearson:.3f}/{dist_spearman:.3f}  |  "
        f"radius corr = {radius_pearson:.3f}  |  reconstruction AUC = {auc:.3f}"
    )
    fig.tight_layout()
    out_path = out_dir / f"fermi_dirac_recovery_{method}_{sampling}.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved figure to {out_path}")

    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--n", type=int, default=200, help="Number of nodes.")
    p.add_argument(
        "--method", choices=["hypermap", "dmercator"], default="hypermap",
        help="Generative re-embedding method (default: hypermap).",
    )
    p.add_argument(
        "--sampling", choices=["hyperbolic", "euclidean"], default="hyperbolic",
        help=(
            "How the ground-truth points are sampled. 'hyperbolic' "
            "(default, recommended) is uniform w.r.t. hyperbolic area "
            "(sample_uniform_hyperbolic_disk) and gives realistic degree "
            "heterogeneity; 'euclidean' is uniform w.r.t. Euclidean area "
            "(sample_uniform_disk) and is kept only as a weak-recovery "
            "contrast case — see that function's docstring."
        ),
    )
    p.add_argument(
        "--R", type=float, default=6.0,
        help="Hyperbolic radius cutoff for --sampling hyperbolic.",
    )
    p.add_argument(
        "--alpha", type=float, default=1.0,
        help=(
            "Radial concentration for --sampling hyperbolic (1.0 = exact "
            "hyperbolic-uniform; >1 concentrates more mass near R, giving "
            "heavier-tailed degrees)."
        ),
    )
    p.add_argument(
        "--max-radius", type=float, default=0.9,
        help="Euclidean radius cutoff for --sampling euclidean (<1).",
    )
    p.add_argument(
        "--r", type=float, default=None,
        help=(
            "Fermi-Dirac connection radius. Defaults depend on --sampling "
            "(0.9*R for hyperbolic, 0.6 for euclidean) unless given."
        ),
    )
    p.add_argument(
        "--t", type=float, default=None,
        help="Fermi-Dirac temperature. Defaults to 0.15 unless given.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=RESULTS)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.n, args.method, args.sampling, args.max_radius, args.R, args.alpha,
        args.r, args.t, args.seed, args.out_dir,
    )
