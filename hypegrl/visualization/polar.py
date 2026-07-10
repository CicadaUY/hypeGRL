"""
Polar (native) hyperbolic visualization.

Plots a graph in the *native representation* of the hyperbolic disk: each node
sits at its angular coordinate ``θ`` and its hyperbolic radial coordinate ``r``
on a polar disk, with popular (central, small-radius) nodes near the origin and
peripheral nodes near the rim.

This is a property of the **representation**, not of any particular embedder: it
takes the exact polar ``(r, v)`` readout (``Representation.to_polar()``) of any 2D
hyperbolic embedding, whatever produced it (D-Mercator, Poincaré maps, HyperMap,
…). It is, in particular, the polar layout used by the Mercator / D-Mercator
papers and the official D-Mercator C++ tool, which makes it handy for eyeballing
such embeddings against that reference.

Companion to :func:`hypegrl.visualization.disk.plot_poincare_graph`, which draws
the same kind of embedding on the (Cartesian) Poincaré disk.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def _as_np(a) -> np.ndarray:
    """Coerce a numpy array or a (detached) torch tensor to a numpy array."""
    return a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)


def plot_polar(
    G: nx.Graph,
    polar_coords: tuple,
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

    Each node is placed at its **hyperbolic radius** ``r_i`` (used directly — no
    ball round-trip, so large radii are not saturated) and angle
    ``θ_i = atan2(v_{i,1}, v_{i,0})`` from its unit direction ``v_i``. Known edges
    are drawn as solid polar segments; ``unknown_edges`` (if given) are drawn
    dashed in a contrasting colour, whether or not they exist in ``G``.

    The polar layout is 2-dimensional, so ``v`` must have 2 columns. It is
    agnostic to which embedder produced the coordinates — D-Mercator (``d=2``),
    Poincaré maps, HyperMap, etc.

    Node-row mapping
    ----------------
    Row ``i`` of ``polar_coords`` is assumed to correspond to node ``nodes[i]``.
    Every embedder reports the row order of its ``embeddings()`` /
    ``embeddings_representation()`` via ``embedder.nodes()`` — pass that as
    ``nodes`` (essential for reordering embedders like ``HyperMapEmbedder`` and
    ``DMercatorEmbedder``). If ``nodes`` is ``None``, the rows are assumed to be
    indexed by integer node labels ``0..N-1`` (matching ``G``).

    Parameters
    ----------
    G:
        Original NetworkX graph. Determines which edges are *known*.
    polar_coords:
        The exact polar coordinates ``(r, v)`` as returned by
        ``Representation.to_polar()`` — e.g.
        ``emb.embeddings_representation().to_polar()``. ``r`` is the ``(N,)``
        hyperbolic radius (used directly, so large radii are *not* saturated) and
        ``v`` the ``(N, 2)`` unit direction. Numpy arrays or detached torch
        tensors are both accepted.
    nodes:
        Sequence mapping row index ``i`` of ``polar_coords`` to a node label in
        ``G``. Defaults to ``range(N)``. Typically ``embedder.nodes()``.
    unknown_edges:
        List of ``(m, n)`` node-label tuples to render as dashed edges, drawn for
        visual reference whether or not they exist in ``G`` (no weight
        annotations).
    node_color:
        A single colour, or a per-node sequence of colours/values (length ``N``,
        ordered like the rows of ``polar_coords``). Useful for colouring by
        community or by degree.
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
        If ``v`` does not have exactly 2 columns (the polar plot is 2D).

    Examples
    --------
    >>> import networkx as nx
    >>> from hypegrl.embedders.dmercator import DMercatorEmbedder
    >>> from hypegrl.visualization import plot_polar
    >>> G = nx.karate_club_graph()
    >>> emb = DMercatorEmbedder(d=2, random_state=0).fit(G)
    >>> fig = plot_polar(
    ...     G, emb.embeddings_representation().to_polar(), nodes=emb.nodes())
    """
    # ── Polar coordinates (r, θ) straight from the (r, v) readout ─────────────
    # ``polar_coords`` is the exact ``(r, v)`` from ``Representation.to_polar()``:
    # the true hyperbolic radius ``r`` is used directly, so there is no ball
    # round-trip and hence no rim saturation collapsing large-radius nodes.
    r, v = polar_coords
    r = _as_np(r)
    v = _as_np(v)
    assert v.shape[1] == 2, (
        "plot_polar draws the 2D polar representation; the direction ``v`` must "
        f"have 2 columns, got shape {v.shape}."
    )
    theta = np.arctan2(v[:, 1], v[:, 0])
    n_rows = r.shape[0]

    unknown_edges = unknown_edges or []

    # ── Row index for each node label ─────────────────────────────────────────
    if nodes is None:
        node2row = {i: i for i in range(n_rows)}
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
        labels = list(nodes) if nodes is not None else list(range(n_rows))
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
