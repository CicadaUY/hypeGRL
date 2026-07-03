"""Tests for Poincaré disk visualization (``hypegrl.visualization.disk``)."""
import matplotlib
import networkx as nx
import numpy as np
import pytest

matplotlib.use("Agg")  # headless; no display needed
import matplotlib.pyplot as plt  # noqa: E402

from hypegrl.visualization.disk import plot_poincare_graph  # noqa: E402


def _edge_segments(ax):
    """The set of drawn edges as frozensets of their two rounded endpoints."""
    segs = set()
    for line in ax.lines:
        xd, yd = line.get_xdata(), line.get_ydata()
        if len(xd) == 2:  # an edge (the disk boundary is a patch, not a line)
            segs.add(frozenset({
                (round(float(xd[0]), 3), round(float(yd[0]), 3)),
                (round(float(xd[1]), 3), round(float(yd[1]), 3)),
            }))
    return segs


def test_nodes_arg_aligns_reordered_embedding():
    """`nodes` maps each X row to its node so edges connect the right points.

    Path 0-1-2 with an embedding whose rows are in a scrambled order. With the
    correct `nodes` mapping the edges connect the intended points; without it the
    default (row == G.nodes() order) would connect a *different* pair — so this
    genuinely exercises the alignment.
    """
    G = nx.Graph([(0, 1), (1, 2)])
    A, B, C = (0.5, 0.0), (0.0, 0.5), (-0.5, 0.0)
    X = np.array([A, B, C])           # rows: 0->A, 1->B, 2->C
    nodes = [2, 0, 1]                 # ...but row0 is node2, row1 node0, row2 node1
    # => node0=B, node1=C, node2=A ; edges 0-1,1-2 connect B-C and C-A

    fig = plot_poincare_graph(G, X, nodes=nodes,
                              show_node_labels=False, show_weights=False)
    segs = _edge_segments(fig.axes[0])
    plt.close(fig)

    assert segs == {frozenset({B, C}), frozenset({C, A})}
    # sanity: this differs from the (wrong) default alignment A-B, B-C
    assert segs != {frozenset({A, B}), frozenset({B, C})}


def test_default_uses_graph_node_order():
    """Without `nodes`, row i is the i-th node of G.nodes() (backward compatible)."""
    G = nx.Graph([(0, 1), (1, 2)])
    A, B, C = (0.5, 0.0), (0.0, 0.5), (-0.5, 0.0)
    X = np.array([A, B, C])
    fig = plot_poincare_graph(G, X, show_node_labels=False, show_weights=False)
    segs = _edge_segments(fig.axes[0])
    plt.close(fig)
    assert segs == {frozenset({A, B}), frozenset({B, C})}


def test_nodes_length_mismatch_raises():
    G = nx.Graph([(0, 1)])
    X = np.zeros((2, 2))
    with pytest.raises(ValueError, match="len\\(nodes\\)"):
        plot_poincare_graph(G, X, nodes=[0, 1, 2])


def test_node_missing_from_nodes_raises():
    """A graph node absent from `nodes` is a clear error, not a wrong-point plot."""
    G = nx.Graph([(0, 5)])           # node 5 has no row
    X = np.zeros((2, 2))
    with pytest.raises(KeyError, match="not in `nodes`"):
        plot_poincare_graph(G, X, nodes=[0, 1],
                            show_node_labels=False, show_weights=False)
