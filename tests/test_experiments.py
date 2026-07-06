"""Tests for the experiments/ reproduction helpers (not part of the library)."""
import networkx as nx
import pytest

from experiments.datasets import balanced_tree_graph, single_cell_graph
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


# ----------------------------------------------------------------------
# Dataset loaders
# ----------------------------------------------------------------------


def test_balanced_tree_default_shape():
    G = balanced_tree_graph(2, 4)
    assert G.number_of_nodes() == 31 and G.number_of_edges() == 30


# Paper Table I: (nodes, edges, diameter) for the single-cell k-NN graphs.
@pytest.mark.parametrize(
    "name, n, m, diam",
    [
        ("ToggleSwitch", 200, 1896, 16),
        ("Olsson", 382, 4214, 8),
        ("MyeloidProgenitors", 640, 5649, 38),
    ],
)
def test_single_cell_graph_matches_paper(name, n, m, diam):
    G = single_cell_graph(name)
    assert G.number_of_nodes() == n
    assert G.number_of_edges() == m
    assert nx.is_connected(G)
    assert nx.diameter(G) == diam
    # Edges carry the k-NN distance as weight; all nodes have a cell-type label.
    u, v, data = next(iter(G.edges(data=True)))
    assert data["weight"] > 0
    assert all("label" in d for _, d in G.nodes(data=True))
