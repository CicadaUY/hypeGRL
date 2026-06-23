"""
Poincare Maps embedder.

Implements the encoder-decoder framework of Klimovskaia et al. (2020)
with the extension to unknown edges from the joint optimisation framework.

Model
-----
::

    Structural similarity : Forest matrix  Q = (I + L)^(-1)
    Decoder               : Soft-min of hyperbolic distances
                             A_hat_ij = softmax_j(-d_H(x_i,x_j)/gamma)
    Loss                  : Symmetric KL divergence
                             sum_i SymKL(A_hat_i, Q_i)

References
----------
Klimovskaia et al., *Poincaré Maps for Analyzing Complex Hierarchies
in Single-Cell Data*, Nature Communications, 2020.
"""

from __future__ import annotations

import warnings
from typing import Optional

import geoopt
import networkx as nx
import numpy as np
import torch

from hypegrl.embedders.base import HyperbolicEmbedder
from hypegrl.manifolds.poincare import POINCARE_BALL
from hypegrl.inference.joint_optimizer import joint_optimize


# ---------------------------------------------------------------------------
# Forest matrix
# ---------------------------------------------------------------------------

def forest_matrix(A: torch.Tensor) -> torch.Tensor:
    """
    Compute the forest matrix ``Q = (I + L)^{-1}`` where ``L = D - A``.

    Parameters
    ----------
    A:
        ``(N, N)`` adjacency matrix.

    Returns
    -------
    ``(N, N)`` forest matrix tensor.

    Notes
    -----
    Uses dense matrix inversion via ``torch.linalg.inv``. For large graphs
    consider sparse solvers; see ``hypegrl.streaming.woodbury`` for
    incremental updates that avoid full recomputation.
    """
    D = torch.diag(A.sum(dim=1))
    L = D - A
    I = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    return torch.linalg.inv(I + L)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

def soft_decoder(X: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """
    Compute the soft-min decoder matrix.

    .. math::

        \\hat{A}_{ij} = \\frac{e^{-d_H(x_i, x_j)/\\gamma}}
                              {\\sum_k e^{-d_H(x_i, x_k)/\\gamma}}

    Parameters
    ----------
    X:
        ``(N, d)`` embedding matrix on the Poincare disk.
    gamma:
        Temperature parameter controlling the sharpness of the soft-min.

    Returns
    -------
    ``(N, N)`` row-stochastic matrix.
    """
    D = POINCARE_BALL.dist(X.unsqueeze(1), X.unsqueeze(0))
    logits = -D / gamma
    return torch.softmax(logits, dim=1)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def symkl_loss_fn(
    X: torch.Tensor,
    A: torch.Tensor,
    gamma: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Symmetric KL divergence loss for Poincare Maps.

        \\mathcal{L} = \\sum_i \\mathrm{SymKL}(\\hat{A}_i,\\, Q_i)

    Parameters
    ----------
    X:
        ``(N, d)`` embeddings on the Poincare disk.
    A:
        ``(N, N)`` adjacency matrix (may contain imputed unknown entries).
    gamma:
        Decoder temperature.
    eps:
        Numerical floor for log arguments.

    Returns
    -------
    Scalar loss tensor.
    """
    A_hat = soft_decoder(X, gamma)
    Q     = forest_matrix(A)
    A_hat = torch.clamp(A_hat, min=eps)
    Q     = torch.clamp(Q,     min=eps)
    return (A_hat * (A_hat / Q).log() + Q * (Q / A_hat).log()).sum()


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class PoincareMapsEmbedder(HyperbolicEmbedder):
    """
    Poincare Maps graph embedder with optional unknown-edge joint optimisation.

    Parameters
    ----------
    d:
        Embedding dimension (2 for visualisation, higher for downstream tasks).
    gamma:
        Soft-min temperature in the decoder.
    lr_X:
        Learning rate for Riemannian Adam on the embeddings.
    lr_a:
        Learning rate for Adam on the unknown edge weights.
    n_steps:
        Number of gradient steps.
    regularize_a:
        L2 regularisation on unknown edge weights. Recommended for
        tree-like graphs or when ``|Omega|`` is large.
    grad_clip:
        Maximum gradient norm. Set to ``0`` to disable clipping.
    log_every:
        Print loss every this many steps. ``0`` suppresses output.
    device:
        Torch device string.
    random_state:
        Seed for reproducible initialisation.

    Examples
    --------
    >>> import networkx as nx
    >>> from hypegrl.embedders.poincare_maps import PoincareMapsEmbedder
    >>> G = nx.karate_club_graph()
    >>> embedder = PoincareMapsEmbedder(d=2, n_steps=200, log_every=0)
    >>> embedder.fit(G, unknown_edges=list(G.edges())[:5])
    PoincareMapsEmbedder(d=2, gamma=1.0)
    >>> X = embedder.embeddings()
    >>> X.shape
    (34, 2)
    """

    def __init__(
        self,
        d: int = 2,
        gamma: float = 1.0,
        lr_X: float = 1e-2,
        lr_a: float = 1e-2,
        n_steps: int = 500,
        regularize_a: float = 0.0,
        grad_clip: float = 10.0,
        log_every: int = 50,
        device: str = "cpu",
        random_state: Optional[int] = None,
    ):
        self.d            = d
        self.gamma        = gamma
        self.lr_X         = lr_X
        self.lr_a         = lr_a
        self.n_steps      = n_steps
        self.regularize_a = regularize_a
        self.grad_clip    = grad_clip
        self.log_every    = log_every
        self.device       = device
        self.random_state = random_state

        self._X: Optional[np.ndarray]               = None
        self._a_omega: Optional[np.ndarray]         = None
        self._loss_history: Optional[list[float]]   = None
        self._unknown_edges: list[tuple[int, int]]  = []
        self._G: Optional[nx.Graph]                 = None

    # ------------------------------------------------------------------
    # HyperbolicEmbedder interface
    # ------------------------------------------------------------------

    def fit(
        self,
        G: nx.Graph,
        unknown_edges: Optional[list[tuple[int, int]]] = None,
        X_init: Optional[np.ndarray] = None,
        a_omega_init: Optional[np.ndarray] = None,
    ) -> "PoincareMapsEmbedder":
        """
        Fit Poincare Maps embeddings, optionally with unknown edges.

        Parameters
        ----------
        G:
            Input graph.
        unknown_edges:
            Edges treated as unknown. Their weights are jointly optimised
            with the embeddings.
        X_init:
            ``(N, d)`` initial embeddings inside the Poincare disk.
            Defaults to small random Gaussian, scaled to stay inside
            the disk.
        a_omega_init:
            Initial estimates for unknown edge weights in ``(0, 1)``.
            Defaults to ``0.5`` for all unknown edges.

        Returns
        -------
        self
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)

        N = G.number_of_nodes()
        unknown_edges = unknown_edges or []

        if X_init is None:
            X_init = np.random.randn(N, self.d) * 0.1

        # Closure capturing gamma for the loss function
        gamma = self.gamma
        # def _loss(X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        #     return symkl_loss_fn(X, A, gamma=gamma)

        result = joint_optimize(
            G           = G,
            loss_fn     = self.distance,
            X_init      = X_init,
            manifold    = POINCARE_BALL,
            unknown_edges   = unknown_edges,
            a_omega_init    = a_omega_init,
            lr_X         = self.lr_X,
            lr_a         = self.lr_a,
            n_steps      = self.n_steps,
            regularize_a = self.regularize_a,
            grad_clip    = self.grad_clip,
            log_every    = self.log_every,
            device       = self.device,
            verbose      = self.log_every > 0,
        )

        self._X             = result["X"]
        self._a_omega       = result["a_omega"]
        self._loss_history  = result["loss_history"]
        self._unknown_edges = unknown_edges
        self._G             = G
        return self

    def distance(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Symmetric KL divergence loss for Poincare Maps.

            \\mathcal{L} = \\sum_i \\mathrm{SymKL}(\\hat{A}_i,\\, Q_i)

        Parameters
        ----------
        X : torch.Tensor
            (N,d) Tensor with the embeddings.
        A : torch.Tensor
            (N,N) Tensor adjacency matrix.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.

        """
        return symkl_loss_fn(X, A, self.gamma)

    def embeddings(self) -> np.ndarray:
        """
        Return the ``(N, d)`` embedding matrix.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if self._X is None:
            raise RuntimeError("Call fit() before embeddings().")
        return self._X

    def structural_similarity(self, G: nx.Graph) -> np.ndarray:
        """
        Compute the forest matrix ``Q = (I + L)^{-1}`` for graph ``G``.

        Parameters
        ----------
        G:
            Input graph.

        Returns
        -------
        ``(N, N)`` NumPy array.
        """
        A = torch.tensor(
            nx.to_numpy_array(G, dtype=np.float64), dtype=torch.float64
        )
        return forest_matrix(A).numpy()

    def decode(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the soft-min decoder matrix ``A_hat`` from embeddings ``X``.

        Parameters
        ----------
        X:
            ``(N, d)`` embedding matrix on the Poincare disk.

        Returns
        -------
        ``(N, N)`` row-stochastic NumPy array.
        """
        X_t = torch.tensor(X, dtype=torch.float64)
        return soft_decoder(X_t, self.gamma).detach().numpy()

    # ------------------------------------------------------------------
    # Capability flags
    # ------------------------------------------------------------------

    def is_gradient_based(self) -> bool:
        return True

    def is_generative(self) -> bool:
        # Partial: generative via greedy rounding or Bernoulli sampling
        return False

    def supports_update(self) -> bool:
        return True

    def supports_node_update(self) -> bool:
        return True

    def update(
        self,
        added_edges   = None,
        removed_edges = None,
        revealed_edges = None,
        added_nodes   = None,
        removed_nodes = None,
        node_edges    = None,
    ) -> "PoincareMapsEmbedder":
        """
        Warm-start re-optimisation after graph changes.

        Currently performs a full warm-started refit. Woodbury-based
        incremental updates for the forest matrix are planned for a future
        release (see ``hypegrl.streaming.woodbury``).

        Parameters
        ----------
        added_edges:
            New edges; treated as unknown and added to ``Omega``.
        removed_edges:
            Edges to remove.
        revealed_edges:
            Dict ``{(i,j): weight}`` of previously unknown edges whose
            true weights are now known.
        added_nodes:
            New node IDs. Embedded via out-of-sample extension (planned).
        removed_nodes:
            Node IDs to remove.
        node_edges:
            Edges connecting new nodes to the existing graph.
        """
        if self._G is None:
            raise RuntimeError("Call fit() before update().")

        has_node_update = added_nodes or removed_nodes
        if has_node_update:
            warnings.warn(
                "Full out-of-sample node extension is not yet implemented. "
                "Falling back to warm-started full refit.",
                stacklevel=2,
            )

        # Build updated graph
        G_new = self._G.copy()

        if removed_nodes:
            self._check_connectivity_after_removal(G_new, nodes=removed_nodes)
            G_new.remove_nodes_from(removed_nodes)

        if removed_edges:
            self._check_connectivity_after_removal(G_new, edges=removed_edges)
            G_new.remove_edges_from(removed_edges)

        if added_edges:
            G_new.add_edges_from(added_edges)

        if revealed_edges:
            for (i, j), w in revealed_edges.items():
                G_new[i][j]["weight"] = w

        # Update unknown set
        new_unknown = list(self._unknown_edges)

        # Revealed edges leave Omega
        if revealed_edges:
            revealed_set = {(min(i,j), max(i,j)) for i,j in revealed_edges}
            new_unknown = [
                (m, n) for (m, n) in new_unknown
                if (min(m,n), max(m,n)) not in revealed_set
            ]

        # New edges enter Omega
        if added_edges:
            for e in added_edges:
                key = (min(e[0],e[1]), max(e[0],e[1]))
                if key not in {(min(m,n),max(m,n)) for m,n in new_unknown}:
                    new_unknown.append(key)

        # Warm-start from current embeddings (extended if nodes added)
        X_init = self._X
        if added_nodes and X_init is not None:
            n_new = len(added_nodes)
            new_rows = np.random.randn(n_new, self.d) * 0.1
            X_init = np.vstack([X_init, new_rows])

        return self.fit(G_new, unknown_edges=new_unknown, X_init=X_init)

    # ------------------------------------------------------------------
    # Extra accessors
    # ------------------------------------------------------------------

    @property
    def imputed_weights(self) -> Optional[np.ndarray]:
        """Imputed unknown edge weights after fitting."""
        return self._a_omega

    @property
    def loss_history(self) -> Optional[list[float]]:
        """Loss value at each optimisation step."""
        return self._loss_history

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_connectivity_after_removal(
        G: nx.Graph,
        nodes: Optional[list] = None,
        edges: Optional[list] = None,
    ) -> None:
        G_test = G.copy()
        if nodes:
            G_test.remove_nodes_from(nodes)
        if edges:
            G_test.remove_edges_from(edges)
        if not nx.is_connected(G_test):
            raise ValueError(
                "The requested removal would disconnect the graph. "
                "A disconnected graph produces a block-diagonal Laplacian "
                "whose forest matrix changes character. "
                "Consider removing one element at a time or handling "
                "connected components separately."
            )

    def __repr__(self) -> str:
        return f"PoincareMapsEmbedder(d={self.d}, gamma={self.gamma})"
