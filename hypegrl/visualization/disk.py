"""
Poincaré disk visualization.

Plots graphs with nodes placed at their Poincaré disk embeddings.
Known and unknown edges are rendered differently, with optional
weight annotations showing estimated and true values side by side.
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_poincare_graph(
    G: nx.Graph,
    X: np.ndarray,
    unknown_edges: Optional[list[tuple[int, int]]] = None,
    a_omega_estimated: Optional[np.ndarray] = None,
    show_weights: bool = True,
    true_weights: Optional[dict[tuple[int, int], float]] = None,
    ax: Optional[plt.Axes] = None,
    node_size: int = 80,
    known_edge_color: str = "#444444",
    unknown_edge_color: str = "#e05c3a",
    disk_color: str = "#f7f7f7",
    show_node_labels: bool = True,
    title: Optional[str] = None,
    figsize: tuple[int, int] = (7, 7),
    nodes: Optional[Sequence] = None,
) -> plt.Figure:
    """
    Plot a graph in the Poincaré disk with nodes at their embeddings.

    Known edges are drawn as solid lines; unknown (imputed) edges are
    drawn as dashed lines in a contrasting colour, regardless of whether
    they exist in ``G``. This means non-existent candidate edges in
    ``unknown_edges`` are rendered too.

    .. warning::
       ``X`` row ``i`` must correspond to a specific node of ``G``. By default
       that node is the ``i``-th of ``G.nodes()`` — but some embedders reorder
       nodes and break this silently, notably
       :class:`~hypegrl.embedders.hypermap.HyperMapEmbedder`, whose
       ``embeddings()`` rows follow degree-descending order, *not* ``G.nodes()``
       order. Passing ``plot_poincare_graph(G, emb.embeddings())`` directly then
       pairs every node with the wrong point and the plot looks like noise. Pass
       the embedding's node order — which every embedder reports via
       :meth:`~hypegrl.embedders.base.HyperbolicEmbedder.nodes` — through the
       ``nodes`` argument so the alignment is handled for you::

           plot_poincare_graph(G, emb.embeddings(), nodes=emb.nodes())

       Embedders that keep ``G.nodes()`` order (e.g. ``PoincareMapsEmbedder``,
       which builds its matrices with ``nx.to_numpy_array(G)``) can omit ``nodes``.

    Parameters
    ----------
    G:
        Original NetworkX graph. Determines which edges are *known*.
    X:
        ``(N, 2)`` array of Poincaré disk embeddings. Row ``i`` corresponds to
        ``nodes[i]`` if ``nodes`` is given, else to the ``i``-th node of
        ``G.nodes()`` (see the warning above).
    unknown_edges:
        List of ``(m, n)`` tuples whose weights were treated as unknown.
        Drawn as dashed edges whether or not they exist in ``G``.
    a_omega_estimated:
        ``(|Omega|,)`` array of imputed weights for ``unknown_edges``.
        Required if ``show_weights=True`` and ``unknown_edges`` is given.
    show_weights:
        If ``True``, annotate each unknown edge with its estimated weight
        and, if available, the true weight.
    true_weights:
        Dict ``{(m, n): true_weight}`` for unknown edges. When provided
        alongside ``show_weights=True``, annotations show both
        ``est=...`` and ``true=...``. Edge ordering is normalised
        internally so ``(m, n)`` and ``(n, m)`` are treated as the same.
    ax:
        Existing ``Axes`` to draw on. If ``None``, a new figure is created.
    node_size:
        Scatter marker size for nodes.
    known_edge_color:
        Colour for known edges.
    unknown_edge_color:
        Colour for unknown (imputed) edges and their annotations.
    disk_color:
        Background fill colour of the Poincaré disk.
    show_node_labels:
        If ``True``, render each node's label as white text inside it (the
        ``nodes`` label when given, otherwise the row index). Disable for large
        graphs.
    title:
        Plot title. Defaults to ``"Graph embeddings in the Poincaré disk"``.
    figsize:
        Figure size in inches, used only when ``ax`` is ``None``.
    nodes:
        Node label for each row of ``X`` (typically ``emb.nodes()``): ``nodes[i]``
        is the node whose embedding is ``X[i]``. Use this when the embedding rows
        are not in ``G.nodes()`` order (see the warning above). When ``None``
        (default), row ``i`` is assumed to be the ``i``-th node of ``G.nodes()``.

    Returns
    -------
    ``matplotlib.figure.Figure``

    Raises
    ------
    AssertionError
        If ``X`` does not have exactly 2 columns.

    Examples
    --------
    >>> import networkx as nx, numpy as np
    >>> from hypegrl.visualization.disk import plot_poincare_graph
    >>> G = nx.path_graph(4)
    >>> X = np.random.randn(4, 2) * 0.3
    >>> fig = plot_poincare_graph(G, X)
    """
    assert X.shape[1] == 2, (
        "plot_poincare_graph requires 2D embeddings; "
        f"got shape {X.shape}."
    )

    if nodes is not None and len(nodes) != X.shape[0]:
        raise ValueError(
            f"len(nodes)={len(nodes)} must equal the number of embedding rows "
            f"X.shape[0]={X.shape[0]}."
        )
    row_of = None if nodes is None else {node: i for i, node in enumerate(nodes)}

    def _pos(node):
        # (x, y) of a node. Without ``nodes``, rows are in G.nodes() order, so the
        # node label doubles as the row index; with ``nodes``, look the row up.
        if row_of is None:
            return X[node]
        try:
            return X[row_of[node]]
        except KeyError:
            raise KeyError(
                f"node {node!r} appears in G but not in `nodes`; `nodes` must "
                "give the node label for every row of X."
            ) from None

    unknown_edges     = unknown_edges or []
    a_omega_estimated = (
        a_omega_estimated if a_omega_estimated is not None
        else np.zeros(len(unknown_edges))
    )

    unknown_set = {(min(m, n), max(m, n)) for m, n in unknown_edges}
    estimated_map = {
        (min(m, n), max(m, n)): float(w)
        for (m, n), w in zip(unknown_edges, a_omega_estimated)
    }
    true_map: dict[tuple[int, int], float] = {}
    if true_weights is not None:
        true_map = {
            (min(m, n), max(m, n)): float(v)
            for (m, n), v in true_weights.items()
        }

    # ── Canvas ────────────────────────────────────────────────────────────
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Disk background and boundary
    ax.add_patch(plt.Circle((0, 0), 1.0, color=disk_color, zorder=0))
    ax.add_patch(plt.Circle(
        (0, 0), 1.0, color="#bbbbbb", fill=False, linewidth=1.5, zorder=1,
    ))

    # ── Edge drawing helpers ──────────────────────────────────────────────
    def _draw_edge(
        m: int, n: int,
        color: str, lw: float, ls: str, zorder: int,
    ) -> None:
        x0, y0 = _pos(m)
        x1, y1 = _pos(n)
        ax.plot(
            [x0, x1], [y0, y1],
            color=color, linewidth=lw, linestyle=ls,
            zorder=zorder, solid_capstyle="round",
        )

    def _annotate_unknown(m: int, n: int) -> None:
        if not show_weights:
            return
        x0, y0 = _pos(m)
        x1, y1 = _pos(n)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        key = (min(m, n), max(m, n))
        est = estimated_map.get(key, float("nan"))
        label = (
            f"est={est:.2f}\ntrue={true_map[key]:.2f}"
            if key in true_map
            else f"est={est:.2f}"
        )
        ax.text(
            mx, my, label,
            fontsize=6.5, color=unknown_edge_color,
            ha="center", va="center",
            bbox=dict(
                boxstyle="round,pad=0.2", fc="white",
                ec=unknown_edge_color, alpha=0.85, linewidth=0.8,
            ),
            zorder=6,
        )

    # ── Known edges (exclude those also in unknown_set) ───────────────────
    for m, n in G.edges():
        if (min(m, n), max(m, n)) not in unknown_set:
            _draw_edge(m, n, known_edge_color, lw=1.0, ls="-", zorder=2)

    # ── Unknown edges (drawn regardless of existence in G) ────────────────
    for m, n in unknown_edges:
        _draw_edge(m, n, unknown_edge_color, lw=1.8, ls="--", zorder=3)
        _annotate_unknown(m, n)

    # ── Nodes ─────────────────────────────────────────────────────────────
    ax.scatter(
        X[:, 0], X[:, 1],
        s=node_size, c="#3a7ebf", edgecolors="white",
        linewidths=0.8, zorder=5,
    )
    if show_node_labels:
        for i in range(len(X)):
            ax.text(
                X[i, 0], X[i, 1], str(i if nodes is None else nodes[i]),
                fontsize=5.5, ha="center", va="center",
                color="white", fontweight="bold", zorder=6,
            )

    # ── Legend ────────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(color=known_edge_color,   label="Known edge"),
        mpatches.Patch(color=unknown_edge_color, label="Unknown edge (imputed)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)

    # ── Axes cosmetics ────────────────────────────────────────────────────
    margin = 0.08
    ax.set_xlim(-1 - margin, 1 + margin)
    ax.set_ylim(-1 - margin, 1 + margin)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        title or "Graph embeddings in the Poincaré disk",
        fontsize=11, pad=10,
    )

    fig.tight_layout()
    return fig
