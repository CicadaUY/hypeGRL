"""Link-prediction evaluation: the edge-removal split.

Implements the edge-removal protocol used to evaluate embeddings on link
prediction: a fraction of the observed edges is hidden, embeddings are
learned on the remaining graph, and the hidden edges are ranked against
true non-links by the decoder. This module provides the split; the ranking
metrics (F1 at a cutoff, decile lift) live alongside it.
"""
from dataclasses import dataclass
from itertools import combinations
from typing import Hashable, Optional

import networkx as nx
import numpy as np

Pair = tuple[Hashable, Hashable]


@dataclass(frozen=True)
class LinkPredictionSplit:
    """The three disjoint node-pair sets produced by an edge-removal split.

    Pairs are unordered ``(u, v)`` tuples over ``G``'s node labels (never
    self-pairs). The three sets partition all node pairs: the observed edges
    split into ``omega_E`` and ``omega_R``, and every non-edge is an
    ``omega_N`` entry.

    Attributes
    ----------
    omega_E:
        Retained edges. The observed graph handed to the embedder is
        ``(V, omega_E)`` — all nodes, only these edges.
    omega_R:
        Removed edges, held out as the positive examples to recover.
    omega_N:
        True non-links: node pairs never connected in the original graph.
    """

    omega_E: list[Pair]
    omega_R: list[Pair]
    omega_N: list[Pair]

    @property
    def candidates(self) -> list[Pair]:
        """Ranking candidates ``omega_R + omega_N`` (positives then negatives)."""
        return self.omega_R + self.omega_N


def link_prediction_split(
    G: nx.Graph,
    q: float = 0.9,
    seed: Optional[int] = None,
) -> LinkPredictionSplit:
    """Randomly hide edges of ``G`` for a link-prediction experiment.

    Each edge is independently *retained* with probability ``q`` (and removed
    with probability ``1 - q``), partitioning node pairs into the retained
    edges ``omega_E``, the removed edges ``omega_R`` (held-out positives), and
    the true non-links ``omega_N`` (pairs never connected in ``G``).

    The split is over node pairs only; it does not build the training graph or
    carry edge weights (see :func:`training_graph`). Isolated nodes may result
    from removal — all original nodes are still represented in
    ``omega_N``/``omega_R`` endpoints, but the caller is responsible for any
    connectivity requirement of the chosen embedder.

    Parameters
    ----------
    G:
        Input graph. Treated as undirected and unweighted for the purpose of
        the split (weights, if any, are recovered by :func:`training_graph`).
    q:
        Probability of retaining each edge. ``q = 0.9`` hides ~10% of edges.
    seed:
        Seed for the retention draws; ``None`` is nondeterministic.

    Returns
    -------
    LinkPredictionSplit
    """
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")

    rng = np.random.default_rng(seed)
    omega_E: list[Pair] = []
    omega_R: list[Pair] = []
    for u, v in G.edges():
        if rng.random() < q:
            omega_E.append((u, v))
        else:
            omega_R.append((u, v))

    omega_N = [
        (u, v) for u, v in combinations(G.nodes(), 2) if not G.has_edge(u, v)
    ]
    return LinkPredictionSplit(omega_E=omega_E, omega_R=omega_R, omega_N=omega_N)


def candidate_scores(
    split: LinkPredictionSplit,
    score_matrix: np.ndarray,
    nodes: Optional[list] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read a decoder matrix into candidate-level ranking arrays.

    Pulls one score per candidate pair out of an ``(N, N)`` decoder output
    (a distance or probability matrix) and marks which candidates are held-out
    positives, ready for :mod:`hypegrl.evaluation.ranking`.

    Parameters
    ----------
    split:
        The split whose ``candidates`` (``omega_R`` then ``omega_N``) are scored.
    score_matrix:
        ``(N, N)`` decoder output; ``score_matrix[i, j]`` scores the pair of
        nodes at rows ``i`` and ``j``.
    nodes:
        Node label for each row of ``score_matrix`` (use ``embedder.nodes()``).
        ``None`` assumes the labels index the matrix directly (integer labels
        ``0..N-1``).

    Returns
    -------
    (scores, is_positive):
        Arrays over ``split.candidates``; ``is_positive`` is ``True`` for the
        ``omega_R`` prefix.
    """
    candidates = split.candidates
    if nodes is None:
        scores = np.array([score_matrix[u, v] for u, v in candidates], dtype=float)
    else:
        pos = {node: i for i, node in enumerate(nodes)}
        scores = np.array(
            [score_matrix[pos[u], pos[v]] for u, v in candidates], dtype=float
        )
    is_positive = np.zeros(len(candidates), dtype=bool)
    is_positive[: len(split.omega_R)] = True
    return scores, is_positive


def training_graph(G: nx.Graph, split: LinkPredictionSplit) -> nx.Graph:
    """Build the observed graph ``(V, omega_E)`` for an edge-removal split.

    Keeps every node of ``G`` (so embedding rows stay aligned with ``G``'s
    nodes) but only the retained edges, copying their edge data — in
    particular ``weight`` — from ``G`` so weighted methods see the original
    weights.

    Parameters
    ----------
    G:
        The original graph the split was drawn from.
    split:
        A :class:`LinkPredictionSplit` produced from ``G``.

    Returns
    -------
    nx.Graph
        A new graph on ``G``'s nodes with edges ``split.omega_E``.
    """
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v in split.omega_E:
        H.add_edge(u, v, **G.get_edge_data(u, v, default={}))
    return H
