"""Hyperbolic-vs-Euclidean geometry diagnostics for a network.

Four per-graph descriptors, each targeting one of the structural signals
discussed for when hyperbolic embeddings out-perform Euclidean ones:

- ``mean_hyperbolicity`` (reused from :mod:`experiments.graph_stats`) —
  sampled Gromov four-point ``delta``: small relative to the diameter means
  the shortest-path metric is close to a tree metric.
- :func:`degree_and_clustering` — mean degree and average clustering
  coefficient, the two quantities Seshadhri et al. (PNAS 2020) prove cannot
  *both* be favourable (low degree, high clustering) in a low-dimensional
  Euclidean embedding.
- :func:`powerlaw_exponent` — the Clauset-Shalizi-Newman maximum-likelihood
  power-law exponent ``gamma`` of the degree sequence, a measure of degree
  heterogeneity. The S^1/H^2 hyperbolic random graph model produces
  ``gamma`` roughly in ``(2, 3)``; homogeneous (Poisson-like) degree
  sequences have no radial structure for hyperbolic space to exploit.
- :func:`ollivier_ricci_curvature` — sampled mean Ollivier-Ricci curvature
  (Wasserstein-1 optimal transport between lazy-random-walk neighbourhoods).
  Negative mean curvature indicates a bottleneck/bridge-dominated,
  tree-like connectivity pattern; positive indicates dense, grid/clique-like,
  Euclidean-friendly connectivity.

:func:`diagnose` bundles all four into one report plus a heuristic
hyperbolic-vs-Euclidean recommendation. None of these tools call into the
``hypegrl`` embedders — they are pre-embedding, graph-only descriptors, so
(matching :mod:`experiments.graph_stats`'s reasoning) they live here with
the experiments rather than in the installed library.
"""
from typing import Optional

import networkx as nx
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from graph_stats import mean_hyperbolicity

# ---------------------------------------------------------------------------
# Degree / clustering (the PNAS axis)
# ---------------------------------------------------------------------------


def degree_and_clustering(G: nx.Graph) -> dict:
    """
    Basic degree and clustering descriptors.

    ``clustering`` is the Watts-Strogatz average local clustering
    coefficient (``nx.average_clustering``) — the quantity in the
    sparse-vs-clustered incompatibility Seshadhri et al. prove for
    low-dimensional embeddings, not the (also common) global transitivity,
    which is reported alongside it for reference.
    """
    degrees = np.array([d for _, d in G.degree()], dtype=np.float64)
    return {
        "n": G.number_of_nodes(),
        "m": G.number_of_edges(),
        "mean_degree": float(degrees.mean()),
        "degree_std": float(degrees.std()),
        "max_degree": int(degrees.max()),
        "clustering": float(nx.average_clustering(G)),
        "transitivity": float(nx.transitivity(G)),
    }


# ---------------------------------------------------------------------------
# Degree heterogeneity: power-law exponent (Clauset-Shalizi-Newman)
# ---------------------------------------------------------------------------


def _ks_distance(tail: np.ndarray, k_min: float, gamma: float) -> float:
    """KS distance between the empirical and fitted (continuous-power-law
    approximation) CCDFs of ``tail``, both evaluated at the sorted tail."""
    sorted_tail = np.sort(tail)
    n = sorted_tail.size
    empirical_ccdf = 1.0 - np.arange(n) / n
    theoretical_ccdf = (sorted_tail / k_min) ** (-(gamma - 1.0))
    return float(np.max(np.abs(empirical_ccdf - theoretical_ccdf)))


def _loglik_ratio_vs_exponential(tail: np.ndarray, k_min: float, gamma: float) -> float:
    """
    Per-tail-point log-likelihood advantage of the fitted power law over the
    best-fitting exponential on the same tail (both continuous
    approximations, both anchored at the same ``k_min`` cutoff), i.e. a
    lightweight stand-in for the significance test Clauset, Shalizi & Newman
    (2009) recommend (full version: bootstrap p-value against several
    alternative distributions) — cheap enough for no extra dependency,
    covering the alternative that most often produces a false "power law"
    verdict on real network data: a Poisson/exponential-tailed degree
    sequence (e.g. Erdos-Renyi) that a loose KS cutoff alone doesn't reject.

    Positive means the power law explains the tail better per point;
    negative (the common case for genuinely non-power-law data) means the
    exponential does. This is a *relative* comparison between two specific
    fits, not a p-value — see :func:`powerlaw_exponent`'s docstring for how
    it's used and its own known limits (still no comparison against
    log-normal or other heavier-tailed alternatives, and no significance
    threshold with a formal error rate).
    """
    n = tail.size
    spread = np.sum(tail - k_min)
    if spread <= 0:
        # Degenerate (zero-variance) tail, e.g. a regular graph: the
        # exponential MLE is undefined (rate -> infinity), and a point-mass
        # tail is definitionally not power-law-heterogeneous either, so
        # report a decisive "exponential wins" rather than raising.
        return float("-inf")
    lam = n / spread
    ll_exp = n * np.log(lam) - lam * spread
    ll_pl = (
        n * np.log(gamma - 1.0) - n * np.log(k_min - 0.5)
        - gamma * np.sum(np.log(tail / (k_min - 0.5)))
    )
    return float((ll_pl - ll_exp) / n)


def powerlaw_exponent(
    G: nx.Graph,
    k_min: Optional[float] = None,
    min_tail: int = 20,
) -> dict:
    """
    Fit a power-law tail ``P(k) ~ k^-gamma`` to the degree sequence.

    Uses the standard Clauset-Shalizi-Newman recipe: for a candidate cutoff
    ``k_min``, the continuous-approximation MLE of the tail ``{k_i >=
    k_min}`` is ``gamma = 1 + n / sum(log(k_i / (k_min - 0.5)))``; ``k_min``
    itself is chosen (when not given) by scanning the observed degree values
    up to the 90th percentile and picking whichever minimises the
    Kolmogorov-Smirnov distance between the fitted and empirical CCDFs — the
    part of the CSN method that keeps the fit honest about *how much* of the
    degree sequence is actually well described by a power law, rather than
    reporting a ``gamma`` regardless of fit quality.

    This is a compact reimplementation of that method (no ``powerlaw``
    dependency); it does not include CSN's full bootstrap significance test.
    In its place, :func:`_loglik_ratio_vs_exponential` gives a much cheaper
    (if narrower) check against the single alternative that most often
    produces a false "power law" verdict here: a Poisson/exponential-tailed
    degree sequence (e.g. Erdos-Renyi) can pass a loose KS cutoff with an
    in-range ``gamma`` despite having no real heterogeneity — see
    ``loglik_ratio_vs_exponential``'s docstring and :func:`diagnose`'s use
    of it. Treat a returned ``gamma`` as a description of tail shape, not a
    hypothesis-test verdict on its own; check ``loglik_ratio_vs_exponential``
    and ``ks`` alongside it.

    Parameters
    ----------
    G:
        The graph.
    k_min:
        Fix the cutoff instead of scanning for it.
    min_tail:
        Minimum number of points required in a candidate tail for its
        ``gamma`` to be considered (candidates with fewer are skipped —
        the MLE is unstable on very small tails).

    Returns
    -------
    dict with ``gamma``, ``k_min``, ``n_tail``, ``ks`` (the KS distance at
    the selected ``k_min``, lower is a better power-law fit), ``tail_fraction``
    (``n_tail / n``, how much of the graph the fit is actually describing),
    and ``loglik_ratio_vs_exponential`` (per-tail-point log-likelihood
    advantage of the power law over the best-fitting exponential on the same
    tail; positive favours power-law, and in practice genuinely
    heterogeneous degree sequences land close to 0 while narrow/Poisson-like
    ones are clearly negative — see that function's docstring). ``gamma`` is
    ``nan`` if no candidate had enough points.
    """
    degrees = np.sort(np.array([d for _, d in G.degree() if d > 0], dtype=np.float64))
    if degrees.size < min_tail:
        return {
            "gamma": float("nan"), "k_min": float("nan"), "n_tail": 0,
            "ks": float("nan"), "tail_fraction": float("nan"),
            "loglik_ratio_vs_exponential": float("nan"),
        }

    if k_min is not None:
        candidates = np.array([float(k_min)])
    else:
        candidates = np.unique(degrees)
        candidates = candidates[candidates <= np.percentile(degrees, 90)]
        if candidates.size == 0:
            candidates = degrees[:1]

    best = None
    for kmin in candidates:
        tail = degrees[degrees >= kmin]
        n = tail.size
        if n < min_tail:
            continue
        gamma = 1.0 + n / np.sum(np.log(tail / (kmin - 0.5)))
        ks = _ks_distance(tail, kmin, gamma)
        if best is None or ks < best[0]:
            best = (ks, kmin, gamma, n)

    if best is None:
        # Fall back to the whole degree sequence rather than reporting nan.
        kmin = float(degrees[0])
        tail = degrees
        n = tail.size
        gamma = 1.0 + n / np.sum(np.log(tail / (kmin - 0.5)))
        lr = _loglik_ratio_vs_exponential(tail, kmin, gamma)
        return {"gamma": float(gamma), "k_min": kmin, "n_tail": int(n), "ks": float("nan"),
                "tail_fraction": 1.0, "loglik_ratio_vs_exponential": lr}

    ks, kmin, gamma, n = best
    tail = degrees[degrees >= kmin]
    lr = _loglik_ratio_vs_exponential(tail, kmin, gamma)
    return {
        "gamma": float(gamma),
        "k_min": float(kmin),
        "n_tail": int(n),
        "ks": float(ks),
        "tail_fraction": float(n / degrees.size),
        "loglik_ratio_vs_exponential": lr,
    }


# ---------------------------------------------------------------------------
# Ollivier-Ricci curvature (sampled)
# ---------------------------------------------------------------------------


def _lazy_walk_measure(
    G: nx.Graph, node, alpha: float, max_neighbors: int, rng: np.random.Generator
) -> tuple:
    """
    Support and mass of the lazy-random-walk measure at ``node``: mass
    ``alpha`` on ``node`` itself, ``(1 - alpha)`` spread uniformly over its
    neighbours. If ``node`` has more than ``max_neighbors`` neighbours, a
    random subset of that size stands in for the full neighbourhood (an
    approximation that keeps the optimal-transport LP's variable count
    bounded on high-degree hubs — see :func:`ollivier_ricci_curvature`).
    """
    neighbors = list(G.neighbors(node))
    if len(neighbors) > max_neighbors:
        idx = rng.choice(len(neighbors), size=max_neighbors, replace=False)
        neighbors = [neighbors[i] for i in idx]
    if not neighbors:
        return [node], np.array([1.0])
    support = [node] + neighbors
    mass = np.concatenate(
        [[alpha], np.full(len(neighbors), (1.0 - alpha) / len(neighbors))]
    )
    return support, mass


def _wasserstein1(mass_a: np.ndarray, mass_b: np.ndarray, cost: np.ndarray) -> float:
    """Exact W1 between two discrete measures via the transportation LP."""
    n_a, n_b = cost.shape
    A_eq = lil_matrix((n_a + n_b, n_a * n_b))
    for i in range(n_a):
        A_eq[i, i * n_b:(i + 1) * n_b] = 1.0
    for j in range(n_b):
        A_eq[n_a + j, j::n_b] = 1.0
    b_eq = np.concatenate([mass_a, mass_b])
    result = linprog(
        cost.ravel(), A_eq=A_eq.tocsr(), b_eq=b_eq, bounds=(0, None), method="highs"
    )
    return float(result.fun) if result.success else float("nan")


def ollivier_ricci_curvature(
    G: nx.Graph,
    n_edges: int = 500,
    alpha: float = 0.5,
    max_neighbors: int = 25,
    distance_cutoff: int = 3,
    seed: Optional[int] = None,
) -> dict:
    """
    Sampled mean Ollivier-Ricci curvature over edges of ``G``.

    For an edge ``(u, v)``, curvature is ``1 - W1(m_u, m_v)``, where
    ``m_u``/``m_v`` are the lazy-random-walk-at-``u``/``v`` measures (mass
    ``alpha`` on the node itself, ``(1-alpha)`` uniform over its neighbours)
    and ``W1`` is their Wasserstein-1 distance under the graph's hop metric,
    solved exactly via the transportation LP (``scipy.optimize.linprog``).
    Negative values mark bridge/bottleneck edges (geodesics through the
    edge diverge — tree-like); positive values mark edges inside dense,
    grid/clique-like neighbourhoods.

    Two approximations make this tractable on real networks with high-degree
    hubs (airport/citation hubs can have degree in the hundreds, which would
    otherwise blow up the LP's variable count as ``O(deg(u) * deg(v))``):

    - **Edges are sampled**, not exhaustively enumerated — ``n_edges``
      drawn uniformly at random (all edges, if the graph has fewer).
    - **Neighbourhoods are capped** at ``max_neighbors``: nodes with more
      neighbours than that get a random subsample standing in for the full
      lazy-walk measure (see :func:`_lazy_walk_measure`).

    Both are documented deviations from an exact, exhaustive Ollivier-Ricci
    computation, not a different definition — for graphs with max degree at
    or below ``max_neighbors`` this recovers the exact value.

    Parameters
    ----------
    G:
        The graph.
    n_edges:
        Number of edges to sample (without replacement).
    alpha:
        Lazy-walk idleness, ``alpha in [0, 1)``. ``0.5`` is a common default
        in the Ricci-curvature-on-graphs literature.
    max_neighbors:
        Cap on neighbourhood size fed to the transport LP (see above).
    distance_cutoff:
        BFS depth used for the local shortest-path lookups between the two
        endpoints' neighbourhoods; pairs farther apart than this are given
        distance ``distance_cutoff + 1``. ``3`` covers every pair reachable
        through the edge itself (max true distance for a lazy-walk-1
        neighbourhood pair is 3), so this is exact, not an approximation,
        for the ``max_neighbors``-capped neighbourhoods used here.
    seed:
        Seed for edge/neighbour sampling.

    Returns
    -------
    dict with ``mean``, ``std``, ``n_sampled`` (edges actually evaluated —
    can be less than ``n_edges`` if some LPs failed to solve), and
    ``fraction_negative``.
    """
    edges = list(G.edges())
    if not edges:
        return {"mean": float("nan"), "std": float("nan"), "n_sampled": 0,
                "fraction_negative": float("nan")}

    rng = np.random.default_rng(seed)
    if len(edges) > n_edges:
        idx = rng.choice(len(edges), size=n_edges, replace=False)
        edges = [edges[i] for i in idx]

    curvatures = []
    for u, v in edges:
        supp_u, mass_u = _lazy_walk_measure(G, u, alpha, max_neighbors, rng)
        supp_v, mass_v = _lazy_walk_measure(G, v, alpha, max_neighbors, rng)

        cost = np.empty((len(supp_u), len(supp_v)))
        for i, a in enumerate(supp_u):
            lengths = nx.single_source_shortest_path_length(G, a, cutoff=distance_cutoff)
            for j, b in enumerate(supp_v):
                cost[i, j] = lengths.get(b, distance_cutoff + 1)

        w1 = _wasserstein1(mass_u, mass_v, cost)
        if not np.isnan(w1):
            curvatures.append(1.0 - w1)  # d(u, v) = 1: (u, v) is an edge

    if not curvatures:
        return {"mean": float("nan"), "std": float("nan"), "n_sampled": 0,
                "fraction_negative": float("nan")}

    curvatures = np.array(curvatures)
    return {
        "mean": float(curvatures.mean()),
        "std": float(curvatures.std()),
        "n_sampled": int(curvatures.size),
        "fraction_negative": float((curvatures < 0).mean()),
    }


# ---------------------------------------------------------------------------
# Bundled report + heuristic recommendation
# ---------------------------------------------------------------------------


def diagnose(
    G: nx.Graph,
    n_hyperbolicity_samples: int = 20_000,
    n_ricci_edges: int = 500,
    seed: Optional[int] = None,
) -> dict:
    """
    Run all four diagnostics on ``G`` and add a heuristic recommendation.

    ``G`` must be connected (required by
    :func:`~experiments.graph_stats.mean_hyperbolicity` and by the
    shortest-path distances the other descriptors use); pass the largest
    connected component if it isn't.

    The recommendation is a simple, transparent point score, not a fitted
    classifier — there is no ground-truth labelled dataset of "networks
    where hyperbolic actually won" to calibrate thresholds against, so this
    thresholds each signal at values that are qualitatively well-established
    in the literature (see the module docstring) and counts how many fire.
    Treat the returned label as a starting hypothesis to check against an
    actual embedding + reconstruction/link-prediction comparison (e.g. via
    :mod:`hypegrl.evaluation`), not as a substitute for one.

    Returns
    -------
    dict with keys ``degree_clustering``, ``powerlaw``, ``hyperbolicity``
    (the raw ``delta_mean`` and its diameter-normalised value), ``ricci``,
    and ``recommendation`` (``{"score", "max_score", "signals", "verdict"}``).
    """
    if not nx.is_connected(G):
        raise ValueError(
            "diagnose() requires a connected graph (shortest-path distances "
            "are undefined across components); pass the largest connected "
            "component, e.g. G.subgraph(max(nx.connected_components(G), key=len))."
        )

    dc = degree_and_clustering(G)
    pl = powerlaw_exponent(G)
    delta = mean_hyperbolicity(G, n_samples=n_hyperbolicity_samples, seed=seed)
    diam = nx.diameter(G) if G.number_of_nodes() < 5000 else None
    delta_norm = delta / diam if diam else float("nan")
    ricci = ollivier_ricci_curvature(G, n_edges=n_ricci_edges, seed=seed)

    signals = {
        # Sparse (mean degree well below n) yet clustered well above the
        # random-graph expectation p = mean_degree / n -- the PNAS axis.
        "sparse_and_clustered": dc["mean_degree"] < 0.1 * dc["n"]
        and dc["clustering"] > 5 * (dc["mean_degree"] / dc["n"]),
        # Heterogeneous, roughly scale-free tail. Gated on the fitted power
        # law beating the exponential alternative on its own tail (see
        # powerlaw_exponent / _loglik_ratio_vs_exponential): -0.05 nats/point
        # is a loose bar (0 would mean an exact tie), chosen from this
        # module's own synthetic tests -- it separates a genuinely
        # heterogeneous BA(300, m=3) fit (-0.02) from every non-heterogeneous
        # case tried (tree, grid, ER, WS all land at -0.27 to -0.66) with
        # comfortable margin, while still tolerating that even a textbook
        # scale-free generator doesn't overwhelmingly beat the exponential
        # alternative at moderate N -- a known finite-size limitation of
        # power-law significance testing in general, not particular to this
        # implementation.
        "heterogeneous_degree": (not np.isnan(pl["gamma"])) and 1.5 < pl["gamma"] < 3.5
        and pl["loglik_ratio_vs_exponential"] > -0.05,
        # Small hyperbolicity relative to how spread out the graph is.
        "tree_like_metric": (not np.isnan(delta_norm)) and delta_norm < 0.2,
        # Negative mean curvature: bottleneck/tree-dominated connectivity.
        # Caveats (both demonstrated by this module's own synthetic tests,
        # and confirmed correct -- not implementation bugs -- against
        # closed-form Wasserstein-1 calculations for star/path/triangle
        # graphs): leaf-heavy trees can still average non-negative (edges
        # incident to a degree-1 leaf have curvature >= 0 even in a tree --
        # only internal branching edges are strongly negative); and sparse
        # Erdos-Renyi graphs also average negative despite having no real
        # hierarchy, since "no local triangles to shortcut around" is a
        # signature of sparsity as much as of hierarchy. Informative in
        # combination with the others, not standalone.
        "negative_curvature": (not np.isnan(ricci["mean"])) and ricci["mean"] < 0,
    }
    score = sum(signals.values())
    verdict = (
        "hyperbolic" if score >= 3 else
        "euclidean" if score <= 1 else
        "ambiguous"
    )

    return {
        "degree_clustering": dc,
        "powerlaw": pl,
        "hyperbolicity": {
            "delta_mean": delta, "diameter": diam, "delta_normalized": delta_norm,
        },
        "ricci": ricci,
        "recommendation": {
            "score": score, "max_score": len(signals), "signals": signals,
            "verdict": verdict,
        },
    }


def format_report(name: str, report: dict) -> str:
    """One-line-per-descriptor human-readable summary of a :func:`diagnose` report."""
    dc, pl, hyp, ricci, rec = (
        report["degree_clustering"], report["powerlaw"], report["hyperbolicity"],
        report["ricci"], report["recommendation"],
    )
    active_signals = ", ".join(k for k, v in rec["signals"].items() if v) or "none"
    lines = [
        f"=== {name} (N={dc['n']}, E={dc['m']}) ===",
        f"  mean degree        : {dc['mean_degree']:.2f} (std {dc['degree_std']:.2f}, "
        f"max {dc['max_degree']})",
        f"  clustering          : {dc['clustering']:.3f}  "
        f"(transitivity {dc['transitivity']:.3f})",
        f"  power-law gamma     : {pl['gamma']:.2f}  (k_min={pl['k_min']:.0f}, "
        f"tail={pl['tail_fraction']:.0%}, "
        f"LR_vs_exp={pl['loglik_ratio_vs_exponential']:.3f})",
        f"  Gromov delta_mean   : {hyp['delta_mean']:.3f}  "
        f"(diam={hyp['diameter']}, normalized={hyp['delta_normalized']:.3f})",
        f"  Ollivier-Ricci mean : {ricci['mean']:.3f}  "
        f"(n={ricci['n_sampled']}, frac<0={ricci['fraction_negative']:.0%})",
        f"  --> {rec['verdict'].upper()}  "
        f"(score {rec['score']}/{rec['max_score']}: {active_signals})",
        "",
    ]
    return "\n".join(lines)
