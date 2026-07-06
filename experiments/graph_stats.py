"""Graph descriptors reported alongside the benchmark tables.

Currently the mean Gromov hyperbolicity ``delta_mean`` — a scalar that is
small (relative to the diameter) for graphs with pronounced hyperbolic /
tree-like structure. It is a per-graph statistic, not a metric of an
embedding, so it lives with the experiments rather than in the library.
"""
from typing import Optional

import networkx as nx
import numpy as np


def _distance_matrix(G: nx.Graph) -> np.ndarray:
    """Dense ``(N, N)`` unweighted shortest-path (hop) distance matrix.

    Uses hop counts (ignoring edge weights) so the derived diameter matches
    the integer diameters reported for these graphs. Unreachable pairs are
    ``inf``; the caller should pass a connected graph.
    """
    nodes = list(G.nodes())
    index = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    D = np.full((N, N), np.inf)
    for src, lengths in nx.all_pairs_shortest_path_length(G):
        i = index[src]
        for dst, d in lengths.items():
            D[i, index[dst]] = d
    return D


def _sample_quadruples(N: int, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """``(n_samples, 4)`` array of distinct node indices per row.

    Rejection sampling of uniform 4-tuples with all entries distinct.
    """
    quads = np.empty((n_samples, 4), dtype=int)
    filled = 0
    while filled < n_samples:
        need = n_samples - filled
        cand = rng.integers(0, N, size=(2 * need + 8, 4))
        a, b, c, d = cand.T
        distinct = (
            (a != b) & (a != c) & (a != d) & (b != c) & (b != d) & (c != d)
        )
        good = cand[distinct]
        take = min(need, len(good))
        quads[filled : filled + take] = good[:take]
        filled += take
    return quads


def mean_hyperbolicity(
    G: nx.Graph,
    n_samples: int = 50_000,
    seed: Optional[int] = None,
) -> float:
    """Monte-Carlo estimate of the mean Gromov (four-point) hyperbolicity.

    For four nodes with pairwise hop distances, form the three sums of
    opposite-pair distances, sort them ``S1 >= S2 >= S3``; the four-point
    hyperbolicity is ``(S1 - S2) / 2``. ``delta_mean`` is this quantity
    averaged over node quadruples — the "on average" hyperbolicity of Gilbert
    & Yim (arXiv:2412.05746). A tree gives ``0``; the value is estimated by
    averaging over ``n_samples`` random distinct quadruples (drawn uniformly).

    The sampling estimator and the use of unweighted (hop) distances are this
    implementation's choices — no reference implementation from the original
    experiments was available to match.

    Parameters
    ----------
    G:
        A connected graph (disconnected graphs yield ``inf``/``nan``).
    n_samples:
        Number of random quadruples to average over. If the graph has fewer
        than ``n_samples`` distinct quadruples, quadruples repeat — the mean is
        still unbiased.
    seed:
        Seed for the quadruple sampling.

    Returns
    -------
    float
        The estimated ``delta_mean``.
    """
    N = G.number_of_nodes()
    if N < 4:
        raise ValueError(f"need at least 4 nodes for hyperbolicity, got {N}")

    D = _distance_matrix(G)
    rng = np.random.default_rng(seed)
    quads = _sample_quadruples(N, n_samples, rng)
    a, b, c, d = quads.T

    s1 = D[a, b] + D[c, d]
    s2 = D[a, c] + D[b, d]
    s3 = D[a, d] + D[b, c]
    sums = np.sort(np.stack([s1, s2, s3], axis=1), axis=1)  # ascending
    delta = (sums[:, 2] - sums[:, 1]) / 2.0
    return float(delta.mean())
