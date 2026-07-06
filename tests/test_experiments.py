"""Tests for the experiments/ reproduction helpers (not part of the library)."""
import networkx as nx
import pytest

from experiments.graph_stats import mean_hyperbolicity


def test_tree_is_zero_hyperbolic():
    # Trees are 0-hyperbolic: every quadruple has delta = 0.
    G = nx.balanced_tree(2, 4)
    assert mean_hyperbolicity(G, n_samples=2000, seed=0) == 0.0


def test_four_cycle_has_delta_one():
    # C4: opposite pairs at distance 2, adjacent at 1 -> sums {4, 2, 2},
    # delta = (4 - 2) / 2 = 1. Its only quadruple, so delta_mean = 1.
    G = nx.cycle_graph(4)
    assert mean_hyperbolicity(G, n_samples=500, seed=0) == pytest.approx(1.0)


def test_requires_four_nodes():
    with pytest.raises(ValueError):
        mean_hyperbolicity(nx.path_graph(3))


def test_runs_and_is_bounded_on_karate():
    G = nx.karate_club_graph()
    delta = mean_hyperbolicity(G, n_samples=5000, seed=0)
    # Small, dense, low-diameter graph: non-negative and well below the diameter.
    assert 0.0 <= delta < nx.diameter(G)
