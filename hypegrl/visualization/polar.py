"""
Polar (native) hyperbolic visualization.

Plots a graph in the *native representation* of the hyperbolic disk: each node
sits at its angular coordinate ``θ`` and its hyperbolic radial coordinate ``r``
on a polar disk, with popular (central, small-radius) nodes near the origin and
peripheral nodes near the rim.

This is a property of the **representation**, not of any particular embedder: it
works for any 2D Poincaré-ball embedding, whatever produced it (D-Mercator,
Poincaré maps, HyperMap, …). It is, in particular, the polar layout used by the
Mercator / D-Mercator papers and the official D-Mercator C++ tool, which makes it
handy for eyeballing such embeddings against that reference.

Companion to :func:`hypegrl.visualization.disk.plot_poincare_graph`, which draws
the same kind of embedding on the (Cartesian) Poincaré disk.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_polar(
    G: nx.Graph,
    X: np.ndarray,
    nodes: Optional[Sequence] = None,
    unknown_edges: Optional[list[tuple[int, int]]] = None,
    node_color: Union[str, Sequence] = "#3a7ebf",
    ax: Optional[plt.Axes] = None,
    node_size: int = 80,
    known_edge_color: str = "#444444",
    unknown_edge_color: str = "#e05c3a",
    show_node_labels: bool = True,
    show_radial_grid: bool = True,
    title: Optional[str] = None,
    figsize: tuple[int, int] = (7, 7),
) -> plt.Figure:
    """
    Plot a graph in the polar (native) hyperbolic representation.

    Each node is placed at angle ``θ_i = atan2(y_i, x_i)`` and hyperbolic radius
    ``r_i = 2·arctanh(‖x_i‖)``, derived from its Poincaré-ball embedding ``x_i``.
    Known edges are drawn as solid polar segments; ``unknown_edges`` (if given)
    are drawn dashed in a contrasting colour, whether or not they exist in ``G``.

    The polar layout is 2-dimensional, so this expects 2-column embeddings. It is
    agnostic to which embedder produced ``X`` — D-Mercator (``d=2``), Poincaré
    maps, HyperMap, etc.

    Node-row mapping
    ----------------
    ``X[i]`` is assumed to correspond to node ``nodes[i]``. Every embedder reports
    the row order of its ``embeddings()`` via ``embedder.nodes()`` — pass that as
    ``nodes`` (essential for reordering embedders like ``HyperMapEmbedder`` and
    ``DMercatorEmbedder``). If ``nodes`` is ``None``, the rows are assumed to be
    indexed by integer node labels ``0..N-1`` (matching ``G``).

    Parameters
    ----------
    G:
        Original NetworkX graph. Determines which edges are *known*.
    X:
        ``(N, 2)`` array of Poincaré-ball embeddings.
    nodes:
        Sequence mapping row index ``i`` of ``X`` to a node label in ``G``.
        Defaults to ``range(N)``. Typically ``embedder.nodes()``.
    unknown_edges:
        List of ``(m, n)`` node-label tuples to render as dashed edges, drawn for
        visual reference whether or not they exist in ``G`` (no weight
        annotations).
    node_color:
        A single colour, or a per-node sequence of colours/values (length ``N``,
        ordered like ``X``). Useful for colouring by community or by degree.
    ax:
        Existing **polar** ``Axes`` to draw on. If ``None``, a new polar figure
        is created. A non-polar ``Axes`` will raise.
    node_size:
        Scatter marker size for nodes.
    known_edge_color:
        Colour for known edges.
    unknown_edge_color:
        Colour for unknown (reference) edges.
    show_node_labels:
        If ``True``, render node labels next to each node. Disable for large graphs.
    show_radial_grid:
        If ``True``, keep the polar radial gridlines/ticks (the hyperbolic radius
        is meaningful — for D-Mercator it encodes the hidden degree). If ``False``,
        hide them for a cleaner look closer to the C++ examples.
    title:
        Plot title. Defaults to ``"Polar (native) hyperbolic representation"``.
    figsize:
        Figure size in inches, used only when ``ax`` is ``None``.

    Returns
    -------
    ``matplotlib.figure.Figure``

    Raises
    ------
    AssertionError
        If ``X`` does not have exactly 2 columns.

    Examples
    --------
    >>> import networkx as nx
    >>> from hypegrl.embedders.dmercator import DMercatorEmbedder
    >>> from hypegrl.visualization import plot_polar
    >>> G = nx.karate_club_graph()
    >>> emb = DMercatorEmbedder(d=2, random_state=0).fit(G)
    >>> fig = plot_polar(G, emb.embeddings(), nodes=emb.nodes())
    """
    assert X.shape[1] == 2, (
        "plot_polar draws the polar (native) representation, which is 2D; "
        f"got embeddings of shape {X.shape}."
    )

    unknown_edges = unknown_edges or []

    # ── Polar coordinates from the Poincaré-ball embedding ────────────────────
    theta = np.arctan2(X[:, 1], X[:, 0])
    norms = np.linalg.norm(X, axis=1)
    # Guard the boundary: arctanh(1) = ∞. Points should be inside the ball, but
    # numerical drift can land exactly on (or just past) the rim.
    norms = np.clip(norms, 0.0, 1.0 - 1e-12)
    r = 2.0 * np.arctanh(norms)

    # ── Row index for each node label ─────────────────────────────────────────
    if nodes is None:
        node2row = {i: i for i in range(X.shape[0])}
    else:
        node2row = {node: i for i, node in enumerate(nodes)}

    unknown_set = {(min(m, n), max(m, n)) for m, n in unknown_edges}

    # ── Canvas ────────────────────────────────────────────────────────────────
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
    else:
        if ax.name != "polar":
            raise ValueError("plot_polar requires a polar Axes.")
        fig = ax.get_figure()

    # ── Edge drawing helper ────────────────────────────────────────────────────
    def _draw_edge(m, n, color: str, lw: float, ls: str, zorder: int) -> None:
        if m not in node2row or n not in node2row:
            return
        i, j = node2row[m], node2row[n]
        ax.plot(
            [theta[i], theta[j]], [r[i], r[j]],
            color=color, linewidth=lw, linestyle=ls,
            zorder=zorder, solid_capstyle="round", alpha=0.6,
        )

    # ── Known edges (exclude those also flagged unknown) ───────────────────────
    for m, n in G.edges():
        if (min(m, n), max(m, n)) not in unknown_set:
            _draw_edge(m, n, known_edge_color, lw=1.0, ls="-", zorder=2)

    # ── Unknown edges (drawn regardless of existence in G) ─────────────────────
    for m, n in unknown_edges:
        _draw_edge(m, n, unknown_edge_color, lw=1.8, ls="--", zorder=3)

    # ── Nodes ──────────────────────────────────────────────────────────────────
    ax.scatter(
        theta, r,
        s=node_size, c=node_color, edgecolors="white",
        linewidths=0.8, zorder=5,
    )
    if show_node_labels:
        labels = list(nodes) if nodes is not None else list(range(X.shape[0]))
        for i, label in enumerate(labels):
            ax.text(
                theta[i], r[i], str(label),
                fontsize=5.5, ha="center", va="center",
                color="black", zorder=6,
            )

    # ── Legend (only show the unknown handle when relevant) ────────────────────
    handles = [mpatches.Patch(color=known_edge_color, label="Known edge")]
    if unknown_edges:
        handles.append(
            mpatches.Patch(color=unknown_edge_color, label="Unknown edge")
        )
    ax.legend(
        handles=handles, loc="upper right",
        bbox_to_anchor=(1.1, 1.1), fontsize=8, framealpha=0.9,
    )

    # ── Axes cosmetics ──────────────────────────────────────────────────────────
    ax.set_rlabel_position(90)
    if not show_radial_grid:
        ax.set_rticks([])
        ax.set_xticklabels([])
    ax.set_title(
        title or "Polar (native) hyperbolic representation",
        fontsize=11, pad=18,
    )

    fig.tight_layout()
    return fig
