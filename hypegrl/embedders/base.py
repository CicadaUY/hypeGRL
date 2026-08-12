"""Abstract base class for all embedding methods."""
from abc import ABC, abstractmethod
from typing import Optional
import networkx as nx
import numpy as np

class HyperbolicEmbedder(ABC):
    """
    Common interface for graph embedding methods.

    All methods support:

    - Full-graph embedding via :meth:`fit`.
    - Partially observed graphs via the ``unknown_edges`` argument.
    - Incremental edge updates via :meth:`update` (if :meth:`supports_update`).
    - Incremental node updates via :meth:`update` (if :meth:`supports_node_update`).
    - Graph generation via a paired
      :class:`~hypegrl.generation.base.GraphGenerator`
      (if :meth:`is_generative`).
    """

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(
        self,
        G: nx.Graph,
        unknown_edges: Optional[list[tuple[int, int]]] = None,
        X_init: Optional[np.ndarray] = None,
    ) -> "HyperbolicEmbedder":
        """
        Compute embeddings for all nodes in ``G``. Acts as the encode function,
        but using the fitted parameters.

        Parameters
        ----------
        G:
            Input graph. Edge weights are used when present.
        unknown_edges:
            Edges whose weights (or existence) are unknown. These are
            treated as free variables and jointly optimized with the
            embeddings. Most gradient-based methods route through the joint
            optimizer. Two kinds of method fall back to zero-imputation with
            a warning instead: closed-form ones (Hydra), which have no
            optimizer to carry the free variables, and gradient-based ones
            whose structural target is a combinatorial function of the
            adjacency (Hydra+, whose ``s(A)`` is the shortest-path matrix),
            through which no gradient can reach the unknown weights.
        X_init:
            Initial embeddings for gradient-based methods. Either an
            ``(N, d)`` coordinate array in the method's native input chart
            (Poincaré ball for most, hyperboloid for Lorentz Embeddings) or a
            fitted :class:`~hypegrl.representations.Representation` (any chart —
            re-charted into the method's optimisation chart via ``from_polar``,
            so an exact large-radius warm start survives without ball
            saturation). If ``None``, each method uses its own default
            initialisation (e.g. random for Poincaré Maps, greedy for HyperMap,
            HYDRA spectral step for HYDRA+). Ignored by non-gradient methods.

        Returns
        -------
        self
        """

    @abstractmethod
    def embeddings(self) -> np.ndarray:
        """
        Return the current embeddings as an ``(N, d)`` **Poincaré-ball**
        coordinate array. Must be called after :meth:`fit`.

        **Legacy interop only — prefer :meth:`embeddings_representation`.**
        This method predates the ``Representation`` layer and survives because
        the disk plotters, :mod:`hypegrl.generation` and outside code accept a
        plain coordinate array. It is not the library's canonical geometry, and
        the ball is not a neutral container for it:

        - the ball radius ``tanh(r/2)`` crowds exponentially against ``1``
          (``1 − 4e-9`` at ``r = 20``), so nodes at large radius are
          indistinguishable at the rim and the array is faithful only to
          ``r ≈ 28``;
        - any distance recomputed from these coordinates through ``geoopt`` is
          capped at ``16.811243``, with an exactly zero gradient past it (see
          :class:`~hypegrl.representations.ball.BallRepresentation`). If you
          need distances from a bare coordinate array, re-chart it first —
          ``PolarRepresentation.from_ball(X).dist()``.

        :meth:`embeddings_representation` returns the fitted
        :class:`~hypegrl.representations.Representation` — the exact geometry in
        the chart it was optimised in, readable in any other chart via its
        ``to_polar`` / ``to_ball`` / ``to_hyperboloid`` methods, and carrying a
        ``dist()`` that is exact at every radius. Use it for anything
        quantitative; use this for plotting and interop.
        """

    def embeddings_representation(self):
        """
        Return the fitted :class:`~hypegrl.representations.Representation` — the
        chart the embedding was optimised in, holding the *exact* geometry.

        Unlike :meth:`embeddings` (Poincaré-ball coordinates, whose radial
        resolution crowds against the rim), it preserves the full radius and
        can be read out in any chart via its ``to_polar`` / ``to_ball`` /
        ``to_hyperboloid`` methods. Rows follow :meth:`nodes` order.

        Returns ``None`` before :meth:`fit`, and for non-gradient methods that do
        not build a representation. Concrete gradient embedders record it in
        ``self._rep`` during :meth:`fit`.
        """
        return getattr(self, "_rep", None)

    def nodes(self) -> Optional[list]:
        """
        Node labels in the row order of :meth:`embeddings` — ``nodes()[i]`` is the
        node whose embedding is row ``i``. Returns ``None`` before :meth:`fit`.

        This is the single accessor for row↔node alignment across all embedders.
        Methods that reorder nodes report that order here (e.g. HyperMap sorts by
        degree); methods that keep the input order return ``list(G.nodes())``. Pass
        it to a plotter's ``nodes=`` argument to align a plot with the embedding::

            plot_polar(G, emb.embeddings_representation().to_polar(), nodes=emb.nodes())

        Concrete embedders record the order in ``self._nodes`` during :meth:`fit`.
        """
        return getattr(self, "_nodes", None)

    @abstractmethod
    def structural_similarity(self, G: nx.Graph) -> np.ndarray:
        """
        Compute ``s(A)``: the structural (dis)similarity matrix used by
        this method (e.g. adjacency matrix for ASE, forest matrix for
        Poincare Maps).

        Parameters
        ----------
        G:
            Input graph.

        Returns
        -------
        (N, N) NumPy array.
        """

    @abstractmethod
    def decode(self, X) -> np.ndarray:
        """
        Compute ``Dec(X)``: the (dis)similarity matrix induced by an embedding.

        Parameters
        ----------
        X:
            Either an ``(N, d)`` coordinate array (Poincaré ball, e.g. the output
            of :meth:`embeddings`) or a fitted
            :class:`~hypegrl.representations.Representation` (as returned by
            :meth:`embeddings_representation`). Passing the representation decodes
            the *exact* geometry — for distance-based decoders it uses
            ``rep.dist()`` directly, avoiding the ``16.811243`` ceiling a
            distance recomputed from ball coordinates hits.

        Returns
        -------
        (N, N) NumPy array.
        """

    # ------------------------------------------------------------------
    # Streaming / re-inference
    # ------------------------------------------------------------------

    def update(
        self,
        added_edges: Optional[list[tuple[int, int]]] = None,
        removed_edges: Optional[list[tuple[int, int]]] = None,
        revealed_edges: Optional[dict[tuple[int, int], float]] = None,
        added_nodes: Optional[list[int]] = None,
        removed_nodes: Optional[list[int]] = None,
        node_edges: Optional[dict[int, list[tuple[int, int]]]] = None,
    ) -> "HyperbolicEmbedder":
        """
        Incrementally update embeddings after graph changes.

        Edge and node updates may be combined in a single call.

        Parameters
        ----------
        added_edges:
            New edges to add to the graph. Each is initially treated as
            an unknown edge and added to the joint optimization.
        removed_edges:
            Edges to remove from the graph. Raises an error if removal
            disconnects the graph, since this changes the character of
            the structural similarity matrix.
        revealed_edges:
            Edges that were previously unknown and whose true weights are
            now known. Dict mapping ``(i, j)`` to the revealed weight.
            These are removed from the unknown set and fixed in the graph.
        added_nodes:
            New node IDs to add. Each new node is embedded via
            out-of-sample extension by default (holding existing
            embeddings fixed and optimizing only the new node's
            position). Falls back to full refit if
            :meth:`supports_node_update` returns ``False``.
        removed_nodes:
            Node IDs to remove. Their embeddings are dropped and the
            graph is updated accordingly. Raises an error if removal
            disconnects the graph.
        node_edges:
            Edges connecting new nodes (from ``added_nodes``) to the
            existing graph or to each other. Dict mapping node ID to a
            list of ``(i, j)`` tuples. Edges not listed here are treated
            as unknown and added to ``Omega``.

        Returns
        -------
        self

        Raises
        ------
        NotImplementedError
            If neither :meth:`supports_update` nor
            :meth:`supports_node_update` applies to the requested change.
        ValueError
            If a removal would disconnect the graph.
        """
        # Check which kinds of updates are being requested
        has_edge_update = any(
            x is not None for x in [added_edges, removed_edges, revealed_edges]
        )
        has_node_update = any(
            x is not None for x in [added_nodes, removed_nodes]
        )

        if has_edge_update and not self.supports_update():
            raise NotImplementedError(
                f"{type(self).__name__} does not support incremental edge "
                "updates. Call fit() again on the updated graph."
            )
        if has_node_update and not self.supports_node_update():
            raise NotImplementedError(
                f"{type(self).__name__} does not support incremental node "
                "updates. Call fit() again on the updated graph."
            )
        raise NotImplementedError(
            f"{type(self).__name__} has not implemented update()."
        )

    # ------------------------------------------------------------------
    # Capability flags
    # ------------------------------------------------------------------

    def supports_update(self) -> bool:
        """
        Return ``True`` if this method supports incremental edge updates
        via :meth:`update` (add/remove/reveal edges).
        """
        return False

    def supports_node_update(self) -> bool:
        """
        Return ``True`` if this method supports incremental node
        addition/deletion via :meth:`update`.

        Methods that implement out-of-sample extension add a node by
        optimizing its position alone, holding the rest fixed. The others
        (Hydra, Hydra+, D-Mercator) require a full refit — for Hydra because
        it is closed-form, and for the other two because their warm-start
        pipelines (the spectral step, the κ/β inference) are whole-graph
        computations with no single-node counterpart.
        """
        return False

    def is_gradient_based(self) -> bool:
        """Return ``True`` if this method uses gradient-based optimization."""
        return False

    def is_generative(self) -> bool:
        """
        Return ``True`` if this method has a natural generative direction
        (i.e. a paired :class:`~hypegrl.generation.base.GraphGenerator`
        is available).
        """
        return False

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
