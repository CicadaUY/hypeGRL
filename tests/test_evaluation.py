"""Tests for the evaluation utilities."""
import networkx as nx
import pytest

from hypegrl.evaluation import (
    LinkPredictionSplit,
    link_prediction_split,
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
