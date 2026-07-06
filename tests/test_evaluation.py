"""Tests for the evaluation utilities."""
import networkx as nx
import numpy as np
import pytest

from hypegrl.evaluation import (
    LinkPredictionSplit,
    candidate_scores,
    f1_at_k,
    hyperbolic_knn_classification,
    lift_curve,
    link_prediction_split,
    pairwise_distance_matrix,
    precision_recall_f1_at_k,
    training_graph,
)


@pytest.fixture
def karate():
    return nx.karate_club_graph()


def test_split_partitions_all_node_pairs(karate):
    """omega_E, omega_R, omega_N are disjoint and cover every node pair."""
    split = link_prediction_split(karate, q=0.9, seed=0)
    N = karate.number_of_nodes()
    total_pairs = N * (N - 1) // 2

    def canon(pairs):
        return {frozenset(p) for p in pairs}

    e, r, n = canon(split.omega_E), canon(split.omega_R), canon(split.omega_N)
    # No self-pairs, and the three sets are pairwise disjoint.
    assert e & r == set() and e & n == set() and r & n == set()
    # Together they cover exactly all unordered node pairs.
    assert len(e) + len(r) + len(n) == total_pairs


def test_split_conserves_edges_and_nonlinks(karate):
    """Retained + removed = original edges; non-links are the true non-edges."""
    split = link_prediction_split(karate, q=0.9, seed=0)
    assert len(split.omega_E) + len(split.omega_R) == karate.number_of_edges()
    for u, v in split.omega_R:
        assert karate.has_edge(u, v)
    for u, v in split.omega_N:
        assert not karate.has_edge(u, v)
    # candidates are exactly positives followed by negatives.
    assert split.candidates == split.omega_R + split.omega_N


def test_split_is_deterministic_under_seed(karate):
    a = link_prediction_split(karate, q=0.9, seed=42)
    b = link_prediction_split(karate, q=0.9, seed=42)
    assert (a.omega_E, a.omega_R) == (b.omega_E, b.omega_R)


def test_q_one_removes_nothing_q_zero_removes_all(karate):
    keep_all = link_prediction_split(karate, q=1.0, seed=0)
    assert keep_all.omega_R == [] and len(keep_all.omega_E) == karate.number_of_edges()
    drop_all = link_prediction_split(karate, q=0.0, seed=0)
    assert drop_all.omega_E == [] and len(drop_all.omega_R) == karate.number_of_edges()


def test_invalid_q_raises(karate):
    with pytest.raises(ValueError):
        link_prediction_split(karate, q=1.5)


def test_training_graph_keeps_all_nodes_and_weights():
    G = nx.Graph()
    G.add_nodes_from(["a", "b", "c", "d"])
    G.add_edge("a", "b", weight=2.5)
    G.add_edge("b", "c", weight=1.0)
    split = LinkPredictionSplit(
        omega_E=[("a", "b")], omega_R=[("b", "c")], omega_N=[]
    )
    H = training_graph(G, split)
    # All nodes preserved (row alignment), only retained edges present.
    assert set(H.nodes()) == set(G.nodes())
    assert H.number_of_edges() == 1 and H.has_edge("a", "b")
    # Weight copied from the original graph.
    assert H["a"]["b"]["weight"] == 2.5


# ----------------------------------------------------------------------
# Ranking metrics
# ----------------------------------------------------------------------

# 10 candidates, 2 positives sitting at the two highest scores.
_SCORES = np.array([0.9, 0.8, 0.1, 0.2, 0.3, 0.05, 0.7, 0.4, 0.6, 0.15])
_POS_TOP = np.zeros(10, dtype=bool)
_POS_TOP[[0, 1]] = True


def test_f1_perfect_when_positives_rank_first():
    m = precision_recall_f1_at_k(_SCORES, _POS_TOP, higher_is_link=True)
    # k defaults to n_positives (=2); both predicted, all metrics = 1.
    assert m["k"] == 2 and m["tp"] == 2
    assert m["precision"] == m["recall"] == m["f1"] == 1.0


def test_f1_half_when_one_positive_is_low_ranked():
    is_pos = np.zeros(10, dtype=bool)
    is_pos[[0, 5]] = True  # 0.9 (top) and 0.05 (bottom)
    m = precision_recall_f1_at_k(_SCORES, is_pos, higher_is_link=True)
    # Predict top 2 -> catches only the high-scored positive.
    assert m["tp"] == 1
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5


def test_direction_flag_flips_ranking():
    # As distances (smaller = link), the two smallest are 0.05 and 0.1.
    is_pos = np.zeros(10, dtype=bool)
    is_pos[[5, 2]] = True  # 0.05 and 0.1
    assert f1_at_k(_SCORES, is_pos, higher_is_link=False) == 1.0
    # Under the probability convention those same two rank last -> miss both.
    assert f1_at_k(_SCORES, is_pos, higher_is_link=True) == 0.0


def test_lift_curve_first_bin_capture():
    curve = lift_curve(_SCORES, _POS_TOP, n_bins=2, higher_is_link=True)
    assert curve.bin_counts == [5, 5]
    # Both positives are top-ranked -> both fall in the first bin.
    assert curve.captured_in_first_bin == (2, 2)
    # First-bin lift is (2/5) / (2/10) = 2.0; second bin has none.
    assert curve.lift[0] == pytest.approx(2.0)
    assert curve.lift[1] == pytest.approx(0.0)


def test_candidate_scores_maps_pairs_through_node_order():
    # Split over labelled nodes; positives are the omega_R prefix.
    split = LinkPredictionSplit(
        omega_E=[], omega_R=[("a", "c")], omega_N=[("a", "b")]
    )
    # Rows are ordered [b, a, c]; encode each entry as 10*row + col for checking.
    nodes = ["b", "a", "c"]
    M = np.array([[10 * i + j for j in range(3)] for i in range(3)], dtype=float)
    scores, is_pos = candidate_scores(split, M, nodes=nodes)
    # ("a","c") -> rows 1,2 -> M[1,2]=12 ; ("a","b") -> rows 1,0 -> M[1,0]=10.
    assert list(scores) == [12.0, 10.0]
    assert list(is_pos) == [True, False]


# ----------------------------------------------------------------------
# Distance-based KNN classification
# ----------------------------------------------------------------------


def _two_poincare_clusters(seed=0, per_class=40):
    """Two well-separated clusters in the Poincaré disk, opposite directions."""
    rng = np.random.default_rng(seed)
    a = np.array([0.5, 0.0]) + 0.03 * rng.standard_normal((per_class, 2))
    b = np.array([-0.5, 0.0]) + 0.03 * rng.standard_normal((per_class, 2))
    X = np.vstack([a, b])
    y = np.array([0] * per_class + [1] * per_class)
    return X, y


def test_pairwise_distance_matrix_is_symmetric_zero_diag():
    X, _ = _two_poincare_clusters()
    D = pairwise_distance_matrix(X)
    assert D.shape == (len(X), len(X))
    assert np.allclose(D, D.T)
    assert np.allclose(np.diag(D), 0.0, atol=1e-9)


def test_knn_separates_well_separated_clusters():
    X, y = _two_poincare_clusters()
    res = hyperbolic_knn_classification(X, y, k=5, seed=0)
    assert res["accuracy"] == 1.0 and res["f1"] == 1.0
    assert res["n_classes"] == 2
    assert res["n_test"] == int(round(0.2 * len(y)))


def test_knn_is_deterministic_under_seed():
    X, y = _two_poincare_clusters()
    a = hyperbolic_knn_classification(X, y, k=3, seed=7)
    b = hyperbolic_knn_classification(X, y, k=3, seed=7)
    assert a == b
