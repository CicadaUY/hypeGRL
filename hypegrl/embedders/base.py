"""Abstract base class for all embedding methods."""
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import networkx as nx
import torch

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
            embeddings. Gradient-based methods route through the joint
            optimizer; non-gradient methods (e.g. Hydra) fall back to
            zero-imputation with a warning.
        X_init:
            ``(N, d)`` initial embeddings for gradient-based methods.
            If ``None``, each method uses its own default initialisation
            (e.g. random for Poincaré Maps, greedy for HyperMap, HYDRA
            spectral step for HYDRA+). Ignored by non-gradient methods.

        Returns
        -------
        self
        """

    @abstractmethod
    def embeddings(self) -> np.ndarray:
        """
        Return the current embeddings as an ``(N, d)`` NumPy array.

        Must be called after :meth:`fit`.
        """

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
    def decode(self, X: np.ndarray) -> np.ndarray:
        """
        Compute ``Dec(X)``: the (dis)similarity matrix induced by embeddings.

        Parameters
        ----------
        X:
            ``(N, d)`` embedding matrix.

        Returns
        -------
        (N, N) NumPy array.
        """
    
    # @abstractmethod
    # def encode(self, G: nx.Graph) -> np.ndarray:
    #     """
    #     Compute ``Enc(A)``: the mapping from the adjacency matrix to the embeddings.

    #     Parameters
    #     ----------
    #     G:
    #         Input graph.

    #     Returns
    #     -------
    #     (N, d) NumPy array.
    
    #     """
    
    @abstractmethod
    def distance(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Computes the distance between structural_similarity(A) and decode(X). 
        Will be used as the loss to be minimized.

        Parameters
        ----------
        X : torch.Tensor
            ``(N,d)`` tensor containing the embeddings.
        A : torch.Tensor
            ``(N,N)`` tensor of the full adjacency matrix.

        Returns
        -------
        Loss evaluated at X and A.

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

        Gradient-based methods support out-of-sample extension for node
        addition by default. Non-gradient methods (Hydra+, d-Mercator)
        require a full refit.
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
