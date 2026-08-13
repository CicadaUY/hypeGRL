"""Graph descriptors reported alongside the benchmark tables.

Currently the mean Gromov hyperbolicity ``delta_mean`` — a scalar that is
small (relative to the diameter) for graphs with pronounced hyperbolic /
tree-like structure. It is a per-graph statistic, not a metric of an
embedding, so it lives with the experiments rather than in the library.
"""
from typing import Optional

import networkx as nx
import numpy as np
from scipy.sparse.csgraph import shortest_path


def _distance_matrix(
    G: nx.Graph,
    nodes: Optional[list] = None,
    memory_limit: int = 512 * 1024**2,
) -> np.ndarray:
    """Dense unweighted shortest-path (hop) distance matrix, restricted to
    ``nodes`` (default: every node in ``G``, giving the full ``(N, N)``
    matrix).

    Uses hop counts (ignoring edge weights) so the derived diameter matches
    the integer diameters reported for these graphs. ``G`` is assumed
    connected (the caller should pass a connected graph), so unreachable
    pairs don't arise.

    Computed via batched calls to
    :func:`scipy.sparse.csgraph.shortest_path` (an unweighted BFS per batch
    of source rows) rather than a single all-pairs pass over the whole
    graph: only BFS trees rooted at ``nodes`` are computed (irrelevant when
    ``nodes`` is every node, but the point when it's a small subset — see
    ``mean_hyperbolicity``'s ``max_candidate_nodes``), and each batch's
    ``(batch_size, N)`` row block is immediately sliced down to just the
    ``len(nodes)`` target columns before the next batch runs, so peak memory
    is ``batch_size × N`` rather than ``len(nodes) × N`` or ``N × N``.
    ``batch_size`` is sized from ``memory_limit`` (bytes; default 512 MiB,
    with 20% reserved for scipy/numpy overhead) via the same batching
    pattern prototyped in ``experiments/diameter_computation.py``.
    """
    all_nodes = list(G.nodes())
    index = {n: i for i, n in enumerate(all_nodes)}
    N = len(all_nodes)

    if nodes is None:
        nodes = all_nodes
    node_indices = np.array([index[n] for n in nodes], dtype=int)
    P = len(node_indices)

    A = nx.to_scipy_sparse_array(G, nodelist=all_nodes, format="csr", dtype=np.float64)

    usable_memory = int(0.8 * memory_limit)
    bytes_per_distance = np.dtype(np.float64).itemsize
    batch_size = max(1, min(P, usable_memory // (N * bytes_per_distance)))

    D = np.empty((P, P))
    for start in range(0, P, batch_size):
        stop = min(start + batch_size, P)
        batch = shortest_path(
            A, directed=False, unweighted=True, indices=node_indices[start:stop],
        )
        D[start:stop] = batch[:, node_indices]
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
    max_candidate_nodes: int = 5000,
    memory_limit: int = 512 * 1024**2,
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

    A full ``(N, N)`` distance matrix is only tractable up to a few thousand
    nodes (169,343 nodes would need 214 GiB of float64). Above
    ``max_candidate_nodes``, quadruples are instead drawn from a uniform
    random pool of ``max_candidate_nodes`` nodes rather than from all ``N``
    (bounding both the distance-matrix memory, to
    ``max_candidate_nodes**2 * 8`` bytes, and the number of BFS sources
    needed to build it). The pool being uniformly random keeps a single
    quadruple's marginal distribution unbiased over all ``C(N, 4)``
    quadruples, but since every quadruple in one call is drawn from the same
    fixed pool, they are no longer independent draws from the whole graph —
    the estimate is, more precisely, the mean hyperbolicity of that random
    node sample, with correspondingly higher variance than sampling
    quadruples independently from all ``N`` nodes would give. This is our
    own scalability heuristic (not from any reference), the same category of
    approximation ``ollivier_ricci_curvature``'s ``n_edges`` already makes
    for Ricci curvature. Below ``max_candidate_nodes``, the pool is every
    node and this caveat doesn't apply.

    Parameters
    ----------
    G:
        A connected graph (disconnected graphs yield ``inf``/``nan``).
    n_samples:
        Number of random quadruples to average over. If the candidate pool
        has fewer than ``n_samples`` distinct quadruples, quadruples
        repeat — the mean is still unbiased.
    seed:
        Seed for the pool and quadruple sampling.
    max_candidate_nodes:
        Size of the random node pool quadruples are drawn from when
        ``G`` has more nodes than this (see above). Ignored (pool = every
        node) when ``G.number_of_nodes() <= max_candidate_nodes``.
    memory_limit:
        Byte budget for ``_distance_matrix``'s batched computation of the
        pool's distance matrix (default 512 MiB).

    Returns
    -------
    float
        The estimated ``delta_mean``.
    """
    N = G.number_of_nodes()
    if N < 4:
        raise ValueError(f"need at least 4 nodes for hyperbolicity, got {N}")

    rng = np.random.default_rng(seed)
    all_nodes = list(G.nodes())
    if N <= max_candidate_nodes:
        pool_nodes = all_nodes
    else:
        pool_idx = rng.choice(N, size=max_candidate_nodes, replace=False)
        pool_nodes = [all_nodes[i] for i in pool_idx]

    D = _distance_matrix(G, nodes=pool_nodes, memory_limit=memory_limit)
    P = len(pool_nodes)
    quads = _sample_quadruples(P, n_samples, rng)
    a, b, c, d = quads.T

    s1 = D[a, b] + D[c, d]
    s2 = D[a, c] + D[b, d]
    s3 = D[a, d] + D[b, c]
    sums = np.sort(np.stack([s1, s2, s3], axis=1), axis=1)  # ascending
    delta = (sums[:, 2] - sums[:, 1]) / 2.0
    return float(delta.mean())
