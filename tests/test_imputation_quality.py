"""Tests for the Lee & Verleysen quality criteria of the imputation quality screen."""
from itertools import combinations

import networkx as nx
import numpy as np

from experiments.imputation.imputation_gap_screen_v2 import hide, make_graph
from experiments.imputation.imputation_quality_screen import (
    auc_lnk,
    coranking,
    load,
    q_nx,
    q_scalars,
    r_nx,
    rank_matrix,
)


def _euclidean(X):
    return np.linalg.norm(X[:, None] - X[None], axis=-1)


def test_quality_matches_official_implementation():
    """Values produced by the official embedding_quality_score.py on the same data."""
    rng = np.random.default_rng(0)
    D_ref = _euclidean(rng.standard_normal((15, 3)))
    D_emb = _euclidean(rng.standard_normal((15, 2)))
    C = coranking(rank_matrix(D_ref), rank_matrix(D_emb))
    q_local, q_global, k_max = q_scalars(q_nx(C))
    assert k_max == 12
    assert np.isclose(q_local, 0.46800889993197686, atol=1e-12)
    assert np.isclose(q_global, 0.9692307692307692, atol=1e-12)


def test_identical_distances_preserve_every_neighbourhood():
    D = _euclidean(np.random.default_rng(1).standard_normal((20, 2)))
    assert np.allclose(q_nx(coranking(rank_matrix(D), rank_matrix(D))), 1.0)


def test_r_nx_and_auc_are_one_for_a_perfect_embedding():
    D = _euclidean(np.random.default_rng(3).standard_normal((20, 2)))
    rnx = r_nx(q_nx(coranking(rank_matrix(D), rank_matrix(D))))
    assert len(rnx) == 20 - 2
    assert np.allclose(rnx, 1.0)
    assert np.isclose(auc_lnk(rnx), 1.0)


def test_r_nx_and_auc_near_zero_for_a_random_embedding():
    rng = np.random.default_rng(4)
    D_ref = _euclidean(rng.standard_normal((300, 2)))
    D_rand = _euclidean(rng.random((300, 2)))
    rnx = r_nx(q_nx(coranking(rank_matrix(D_ref), rank_matrix(D_rand))))
    assert abs(auc_lnk(rnx)) < 0.05


def test_r_nx_matches_its_definition():
    """Eq. (44): R_NX(K) = ((N−1)·Q_NX(K) − K)/(N−1−K), checked at every K."""
    qnx = np.random.default_rng(5).random(9)  # N − 1 = 9
    expected = [(9 * qnx[K - 1] - K) / (9 - K) for K in range(1, 9)]
    assert np.allclose(r_nx(qnx), expected)
    w = 1 / np.arange(1, 9)
    assert np.isclose(auc_lnk(r_nx(qnx)), (np.array(expected) * w).sum() / w.sum())


def test_rank_matrix_self_zero_and_ties_by_index():
    D = np.array([[0.0, 1.0, 1.0],
                  [1.0, 0.0, 2.0],
                  [1.0, 2.0, 0.0]])
    R = rank_matrix(D)
    assert (np.diag(R) == 0).all()
    assert R[0, 1] == 1 and R[0, 2] == 2  # tie at distance 1: lower index ranks first
    assert sorted(R[1]) == [0, 1, 2]


def test_coranking_counts_every_ordered_pair_once():
    D = _euclidean(np.random.default_rng(2).standard_normal((12, 2)))
    C = coranking(rank_matrix(D), rank_matrix(D.T + 0.0))
    assert C.sum() == 12 * 11


def test_hide_keep_connected_leaves_the_zeroed_graph_connected():
    """GSC N=100, beta=1.5, 50% hidden, seed 1 splits into 6 components by default."""
    G = make_graph("gsc", 100, 1.5)
    for keep, expect_connected in ((False, False), (True, True)):
        G0 = G.copy()
        G0.remove_edges_from(hide(G, 0.5, 1, keep_connected=keep))
        assert nx.is_connected(G0) == expect_connected


def test_hide_default_is_unchanged():
    """keep_connected=False must reproduce the published screens' hidden pairs."""
    G = make_graph("gsc", 100, 1.5)
    rng = np.random.default_rng(0)
    edges = list(G.edges())
    non_edges = [(u, v) for u, v in combinations(G, 2) if not G.has_edge(u, v)]
    n_con, n_dis = int(len(edges) * 0.2), int(len(non_edges) * 0.2 / 5)
    expected = ([non_edges[i] for i in rng.choice(len(non_edges), n_dis, replace=False)]
                + [edges[i] for i in rng.choice(len(edges), n_con, replace=False)])
    assert hide(G, 0.2, 0) == [tuple(map(int, p)) for p in expected]


def test_myeloid_node_order_equals_labels():
    """Hidden pairs index matrices by label, so node order must equal label order."""
    G, refs = load("myeloid", None, None)
    assert list(G.nodes()) == list(range(G.number_of_nodes()))
    assert refs["features"].shape == (G.number_of_nodes(),) * 2
    assert np.isfinite(refs["features"]).all()
