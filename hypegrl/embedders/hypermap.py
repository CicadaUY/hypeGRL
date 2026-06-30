"""
HyperMap embedder with gradient-descent refinement.

Implements the HyperMap maximum-likelihood embedding algorithm from:
    Papadopoulos et al., "Network Geometry Inference using Common Neighbors",
    Physical Review E, 2015.

The generative model is the S1/H2 model with Fermi-Dirac connection
probability:

    P_ij = 1 / (1 + exp(zeta/(2T) * (x_ij - R_i)))

where x_ij is the H^d hyperbolic distance between nodes i and j,
and R_i is a per-node threshold determined analytically from degree.

Workflow
--------
1. **Initialization** (greedy, C++-faithful, always 2D):
   radii are assigned analytically; angles are placed sequentially
   using either the common-neighbors (Phase 1) or Fermi-Dirac (Phase 2)
   log-likelihood. The result lives in H^2 polar coordinates.

2. **Lift to d dimensions**: the 2D polar coordinates are converted to
   Poincaré ball Cartesian via :func:`~hypegrl.manifolds.poincare.polar_to_poincare`
   (for d=2) or :func:`~hypegrl.manifolds.poincare.hyperspherical_to_poincare`
   (for d>2, with extra dimensions initialised to zero angle).

3. **Gradient refinement**: embeddings are wrapped as
   ``geoopt.ManifoldParameter`` and refined by minimising the Fermi-Dirac
   NLL via RiemannianAdam. Radii can be fixed or learnable (see
   ``fix_radii``). Global parameters (beta, T, zeta, R_i) are always
   fixed as in the original method.

Unknown edges
-------------
Unknown edges slot into the joint optimisation framework via
``joint_optimize``, with the Fermi-Dirac NLL as the loss function.
"""

from __future__ import annotations

import warnings
from typing import Optional

import geoopt
import networkx as nx
import numpy as np
import torch

from hypegrl.embedders.base import HyperbolicEmbedder
from hypegrl.manifolds.poincare import (
    POINCARE_BALL,
    polar_to_poincare,
    poincare_to_polar,
    hyperspherical_to_poincare,
    poincare_to_hyperspherical,
)
from hypegrl.inference.joint_optimizer import (
    graph_to_tensor,
    build_adjacency,
    logit_init,
)
from hypegrl.embedders._hypermap_init import hypermap_init, assign_radii
from hypegrl.inference.joint_optimizer import joint_optimize


def estimate_gamma(G, k_min=1):
    """
    Estimates the power-law exponent of a certain graph's degree distribution. Based on https://arxiv.org/pdf/0706.1062.
    k_min is the minimum degree after which we assume that the degree distribution is power-law
    """
    degrees = np.array([deg for _, deg in G.degree()])
    degrees = degrees[degrees >= k_min]

    if len(degrees) == 0:
        raise ValueError("No degrees >= k_min")

    n = len(degrees)
    # see Eq. 3.7 of https://arxiv.org/pdf/0706.1062. This is a reasonably good estimate, 
    # although if x_min<6 it should not provide accurate results.
    gamma_hat = 1 + n / np.sum(np.log(degrees / (k_min - 0.5)))
    return gamma_hat, degrees

# ---------------------------------------------------------------------------
# Fermi-Dirac loss (autograd-compatible, arbitrary d)
# ---------------------------------------------------------------------------

def fermi_dirac_nll(
    X: torch.Tensor,
    A: torch.Tensor,
    R,
    zeta_over_2T: float,
    manifold: geoopt.Manifold = POINCARE_BALL,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Negative Fermi-Dirac log-likelihood (see eq. (17) in Papadopoulos et al.).

    .. math::

        \\mathcal{L} = -\\sum_{i < j} \\left[
            A_{ij} \\log P_{ij} + (1 - A_{ij}) \\log(1 - P_{ij})
        \\right]

    where :math:`P_{ij} = 1/(1+exp(\\zeta/(2T)(x_{ij} - R_{ij})))` and
    :math:`x_{ij}` is the hyperbolic distance.

    The threshold :math:`R_{ij}` is **per node**, not global: in the model each
    link forms at the birth time of its *younger* endpoint, so the pair uses
    that node's threshold (Eqs. 1, 5-8 in Papadopoulos et al.). Rows are
    degree-descending, so the younger node of a pair is the one with the larger
    index, i.e. :math:`R_{ij} = R_{\\max(i, j)}` — matching ``decode()`` and the
    greedy initialisation.

    Parameters
    ----------
    X:
        ``(N, d)`` Poincaré ball embeddings.
    A:
        ``(N, N)`` adjacency matrix.
    R:
        Per-node thresholds. A ``(N,)`` array/tensor is interpreted as the
        per-node :math:`R_i` and expanded to the per-pair matrix
        :math:`R_{\\max(i, j)}`. A scalar is broadcast as a single global
        threshold (legacy behaviour, kept for convenience).
    zeta_over_2T:
        Fixed scalar :math:`\\zeta / (2T)`.
    manifold:
        A geoopt.Manifold indicating where the embeddings live.
    eps:
        Numerical floor for log arguments.

    Returns
    -------
    Scalar NLL tensor.
    """
    D = manifold.dist(X.unsqueeze(1), X.unsqueeze(0))  # (N, N)

    N   = X.shape[0]

    R_t = torch.as_tensor(R, dtype=X.dtype, device=X.device)
    if R_t.ndim == 0:
        R_pair = R_t                       # scalar: single global threshold
    else:
        # per-pair threshold = younger node's R = R[max(i, j)]
        idx    = torch.arange(N, device=X.device)
        R_pair = R_t[torch.maximum(idx.unsqueeze(1), idx.unsqueeze(0))]

    P = torch.sigmoid(zeta_over_2T * (R_pair - D))
    P = torch.clamp(P, eps, 1.0 - eps)

    mask = torch.triu(
        torch.ones(N, N, dtype=torch.bool, device=X.device), diagonal=1
    )
    nll = -(A[mask] * torch.log(P[mask])
            + (1.0 - A[mask]) * torch.log(1.0 - P[mask]))
    return nll.sum()


# ---------------------------------------------------------------------------
# Radius-preserving retraction
# ---------------------------------------------------------------------------

def retract_fixed_radius(
    X: torch.Tensor,
    rho_fixed: torch.Tensor,
    eps: float = 1e-5,
) -> None:
    """
    In-place retraction that rescales each row of X to its target
    Poincaré ball norm, preserving the encoded H^d radius.

    Used when ``fix_radii=True``: after each optimizer step the norms
    may drift; this snaps them back to ``tanh(zeta * r / 2)``.

    Parameters
    ----------
    X:
        ``(N, d)`` ManifoldParameter to retract in-place.
    rho_fixed:
        ``(N,)`` target norms.
    eps:
        Safety margin from the ball boundary.
    """
    with torch.no_grad():
        norms  = X.norm(dim=-1, keepdim=True).clamp(min=eps)
        target = rho_fixed.unsqueeze(1).clamp(max=1.0 - eps)
        X.copy_(X / norms * target)


# ---------------------------------------------------------------------------
# d-dimensional initialization
# ---------------------------------------------------------------------------

def _init_to_d_dimensions(
    thetas_2d: np.ndarray,
    r: np.ndarray,
    d: int,
    zeta: float,
) -> np.ndarray:
    """
    Lift 2D HyperMap initialization to d-dimensional Poincaré ball.

    For d=2: direct polar -> Poincaré disk conversion.
    For d>2: embed in d dimensions with extra hyperspherical angles = 0
             (places all nodes in the 2D subspace spanned by the first
             two coordinates; gradient refinement then moves them freely
             in the full d-dimensional space).

    Parameters
    ----------
    thetas_2d:
        ``(N,)`` angular coordinates from 2D initialization.
    r:
        ``(N,)`` radial coordinates.
    d:
        Target embedding dimension.
    zeta:
        Curvature.

    Returns
    -------
    ``(N, d)`` Poincaré ball coordinates.
    """
    if d == 2:
        return polar_to_poincare(thetas_2d, r, zeta)

    # d > 2: angles array of shape (N, d-1)
    # phi_0 = theta_2d (the HyperMap angle, in the first 2D subspace)
    # phi_1, ..., phi_{d-2} = 0  (collapses extra dims to the 2D plane)
    N = len(thetas_2d)
    angles = np.zeros((N, d - 1))
    angles[:, d - 2] = thetas_2d   # last angle = azimuthal, maps to (cos,sin)
    return hyperspherical_to_poincare(angles, r, zeta)


# ---------------------------------------------------------------------------
# HyperMap embedder
# ---------------------------------------------------------------------------

class HyperMapEmbedder(HyperbolicEmbedder):
    """
    HyperMap graph embedder with optional gradient-descent refinement.

    Supports arbitrary embedding dimension ``d`` and optionally frees
    the radial coordinates during gradient refinement.

    Parameters
    ----------
    d:
        Embedding dimension (default 2, matching the original HyperMap).
        For d > 2, the 2D greedy initialization is lifted to d dimensions
        with extra angles set to zero; gradient refinement then moves
        all coordinates freely.
    gamma:
        Power-law exponent of the degree distribution (gamma > 2).
    T:
        Temperature controlling clustering (0 < T < 1).
    zeta:
        Curvature of H^d (default 1.0).
    fix_radii:
        If ``True`` (default), radial coordinates are held fixed at their
        analytically determined values throughout gradient refinement —
        only angular positions are updated. This matches the original
        HyperMap paper.
        If ``False``, the full embedding (including radii) is free during
        refinement. The degree-based initialization still provides a good
        starting point, but the optimizer can move nodes radially. This
        gives more flexibility at the cost of potentially drifting from
        the degree-consistent solution.
    k_speedup:
        Degree threshold for the fast Phase-2 approximation in the greedy
        initialization. Nodes with degree < k_speedup use only their
        neighbors as comparison set. Set to 0 to disable (C++ default).
    corrections:
        If ``True``, run correction sweeps at degree thresholds 10/20/40/60
        during initialization.
    n_steps:
        Number of gradient refinement steps. Set to 0 to return the
        greedy initialization without refinement.
    lr_X:
        Learning rate for RiemannianAdam on the embeddings.
    lr_a:
        Learning rate for Adam on unknown edge weights.
    regularize_a:
        L2 regularisation coefficient on unknown edge weights.
    grad_clip:
        Maximum gradient norm for clipping. Set to 0 to disable.
    log_every:
        Print loss every this many steps during refinement. 0 suppresses.
    device:
        Torch device string.
    verbose_init:
        Print progress during greedy initialization.

    Examples
    --------
    >>> import networkx as nx
    >>> from hypegrl.embedders.hypermap import HyperMapEmbedder
    >>> G = nx.karate_club_graph()
    >>> emb = HyperMapEmbedder(d=2, gamma=2.5, T=0.5,
    ...                        n_steps=100, log_every=0)
    >>> emb.fit(G)
    HyperMapEmbedder(d=2, gamma=2.5, T=0.5, zeta=1.0)
    >>> emb.embeddings().shape
    (34, 2)
    """

    def __init__(
        self,
        d: int = 2,
        gamma: float = None,
        T: float = 0.5,
        zeta: float = 1.0,
        fix_radii: bool = True,
        k_speedup: int = 0,
        corrections: bool = True,
        n_steps: int = 200,
        lr_X: float = 1e-2,
        lr_a: float = 1e-2,
        regularize_a: float = 0.0,
        grad_clip: float = 10.0,
        log_every: int = 50,
        device: str = "cpu",
        verbose_init: bool = True,
    ):
        self.d            = d
        self.gamma        = gamma
        self.T            = T
        self.zeta         = zeta
        self.fix_radii    = fix_radii
        self.k_speedup    = k_speedup
        self.corrections  = corrections
        self.n_steps      = n_steps
        self.lr_X         = lr_X
        self.lr_a         = lr_a
        self.regularize_a = regularize_a
        self.grad_clip    = grad_clip
        self.log_every    = log_every
        self.device       = device
        self.verbose_init = verbose_init

        self._X             : Optional[np.ndarray]           = None
        self._a_omega       : Optional[np.ndarray]           = None
        self._loss_history  : Optional[list[float]]          = None
        self._unknown_edges : list[tuple[int, int]]          = []
        self._G             : Optional[nx.Graph]             = None
        self._params        : Optional[dict]                 = None
        self._nodes_sorted  : Optional[list]                 = None
        self._thetas_init   : Optional[np.ndarray]           = None
        self._r_init        : Optional[np.ndarray]           = None
        self._R             : Optional[np.ndarray]           = None

    # ------------------------------------------------------------------
    # HyperbolicEmbedder interface
    # ------------------------------------------------------------------

    def fit(
        self,
        G: nx.Graph,
        unknown_edges: Optional[list[tuple[int, int]]] = None,
        X_init: Optional[np.ndarray] = None,
        a_omega_init: Optional[np.ndarray] = None,
    ) -> "HyperMapEmbedder":
        """
        Fit HyperMap embeddings.

        Parameters
        ----------
        G:
            Input graph.
        unknown_edges:
            Edges treated as unknown; jointly optimised with embeddings.
        X_init:
            ``(N, d)`` initial Poincaré ball embeddings for the gradient
            refinement step.  Rows must be ordered to match
            :attr:`nodes_sorted` (degree-descending).  If ``None`` (or if
            this is the first call), the greedy HyperMap initialisation is
            run to produce the starting point.
        a_omega_init:
            Initial estimates for the unknown edges. If `None`, then this is
            estimated as the mean degree of the corresponding node.

        Returns
        -------
        self
        """
        unknown_edges       = unknown_edges or []
        self._G             = G
        self._unknown_edges = unknown_edges

        # estimate gamma if necessary
        if self.gamma is None:
            self.gamma, _ = estimate_gamma(G, k_min=3)

        # ── Stage 1: greedy initialization (always 2D) ───────────────────
        # Run when X_init is not provided or on the first call (no cached
        # node ordering yet).
        if X_init is None or self._nodes_sorted is None:
            thetas, r_final, nodes_sorted, params = hypermap_init(
                G,
                gamma       = self.gamma,
                T           = self.T,
                zeta        = self.zeta,
                k_speedup   = self.k_speedup,
                corrections = self.corrections,
                verbose     = self.verbose_init,
            )
            self._thetas_init  = thetas
            self._r_init       = r_final
            self._nodes_sorted = nodes_sorted
            self._params       = params

            _, R, _ = assign_radii(params)
            self._R = R

        # ── Lift to d dimensions (only when not warm-starting) ───────────
        if X_init is None:
            X_init = _init_to_d_dimensions(
                self._thetas_init, self._r_init, self.d, self.zeta
            )

        # ── Stage 2: gradient refinement ─────────────────────────────────
        if self.n_steps == 0:
            self._X            = X_init
            self._a_omega      = np.array([])
            self._loss_history = []
            return self

        # joint_optimize builds the adjacency from the graph in its node-iteration
        # order, but X_init, self._R and the returned embeddings are all in
        # degree-descending (nodes_sorted) order. Reorder the graph so the
        # adjacency rows line up with the embedding rows (and with the per-node
        # thresholds in self._R); otherwise the loss pairs each hyperbolic
        # distance with the wrong adjacency entry. unknown_edges are remapped to
        # the same order; their list order (and hence a_omega alignment) is kept.
        order = {node: idx for idx, node in enumerate(self._nodes_sorted)}
        G_sorted = nx.Graph()
        G_sorted.add_nodes_from(range(len(self._nodes_sorted)))
        G_sorted.add_edges_from(
            (order[u], order[v], data) for u, v, data in G.edges(data=True)
        )
        unknown_sorted = [(order[u], order[v]) for (u, v) in unknown_edges]

        result = joint_optimize(
            G           = G_sorted,
            loss_fn     = self.distance,
            X_init      = X_init,
            manifold    = POINCARE_BALL,
            unknown_edges   = unknown_sorted,
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
        # Per-node thresholds R_i (Eq. 3): the loss uses the younger node's R
        # for each pair, consistent with the greedy init and decode().
        return fermi_dirac_nll(
            X,
            A,
            self._R,
            self.zeta / (2.0 * self.T),
            POINCARE_BALL
        )

    def embeddings(self) -> np.ndarray:
        """
        Return ``(N, d)`` Poincaré ball embeddings.

        Row order matches :attr:`nodes_sorted` (degree-descending).

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
        Return the adjacency matrix as the structural similarity (``s(A) = A``).

        HyperMap uses the adjacency matrix directly as its structural
        target, unlike Poincaré Maps which uses the forest matrix.
        """
        return nx.to_numpy_array(G, dtype=np.float64)

    def decode(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the Fermi-Dirac connection probability matrix from embeddings.

        Works for any embedding dimension d.

        Parameters
        ----------
        X:
            ``(N, d)`` Poincaré ball embeddings.

        Returns
        -------
        ``(N, N)`` matrix of connection probabilities ``P_ij in (0, 1)``.
        """
        if self._R is None:
            raise RuntimeError("Call fit() before decode().")
        X_t      = torch.tensor(X, dtype=torch.float64)
        R_t      = torch.tensor(self._R, dtype=torch.float64)
        D        = POINCARE_BALL.dist(X_t.unsqueeze(1), X_t.unsqueeze(0))
        N        = X.shape[0]
        idx      = torch.arange(N)
        # Rows are degree-descending, so the larger index is the node that
        # appeared later ("younger"). The pairwise connection probability uses
        # the younger node's threshold R (Eqs. 1, 5-8 in Papadopoulos et al.):
        # each link forms at the birth time of its younger endpoint.
        R_pair   = R_t[torch.maximum(idx.unsqueeze(1), idx.unsqueeze(0))]
        P        = torch.sigmoid(
            -(self.zeta / (2.0 * self.T)) * (D - R_pair)
        )
        return P.detach().numpy()

    # ------------------------------------------------------------------
    # Capability flags
    # ------------------------------------------------------------------

    def is_gradient_based(self) -> bool:
        return True

    def is_generative(self) -> bool:
        return True

    def supports_update(self) -> bool:
        return True

    def supports_node_update(self) -> bool:
        return True

    def update(
        self,
        added_edges    = None,
        removed_edges  = None,
        revealed_edges = None,
        added_nodes    = None,
        removed_nodes  = None,
        node_edges     = None,
    ) -> "HyperMapEmbedder":
        """Warm-started refit after graph changes."""
        if self._G is None:
            raise RuntimeError("Call fit() before update().")

        if added_nodes or removed_nodes:
            warnings.warn(
                "Full out-of-sample node extension is not yet implemented. "
                "Falling back to warm-started full refit.",
                stacklevel=2,
            )

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

        new_unknown = list(self._unknown_edges)
        if revealed_edges:
            rev = {(min(i,j), max(i,j)) for i,j in revealed_edges}
            new_unknown = [
                e for e in new_unknown
                if (min(e[0],e[1]), max(e[0],e[1])) not in rev
            ]
        if added_edges:
            known = {(min(e[0],e[1]), max(e[0],e[1])) for e in new_unknown}
            for e in added_edges:
                key = (min(e[0],e[1]), max(e[0],e[1]))
                if key not in known:
                    new_unknown.append(key)

        return self.fit(G_new, unknown_edges=new_unknown, X_init=self._X)

    # ------------------------------------------------------------------
    # Extra accessors
    # ------------------------------------------------------------------

    @property
    def polar_coordinates(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """
        Current embeddings in H^2 polar coordinates ``(thetas, r)``.

        For d=2, converts from Poincaré disk directly.
        For d>2, projects to the first two dimensions before converting
        (approximate — the extra dimensions break the polar interpretation).
        """
        if self._X is None:
            return None
        X2 = self._X[:, :2]
        return poincare_to_polar(X2, self.zeta)

    @property
    def hyperspherical_coordinates(
        self,
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """
        Current embeddings in hyperspherical coordinates ``(angles, r)``.

        Returns ``(N, d-1)`` angles and ``(N,)`` radii.
        """
        if self._X is None:
            return None
        return poincare_to_hyperspherical(self._X, self.zeta)

    @property
    def nodes_sorted(self) -> Optional[list]:
        """Node IDs in degree-descending order (embedding row order)."""
        return self._nodes_sorted

    @property
    def loss_history(self) -> Optional[list[float]]:
        """Loss at each gradient refinement step."""
        return self._loss_history

    @property
    def imputed_weights(self) -> Optional[np.ndarray]:
        """Imputed unknown edge weights after fitting."""
        return self._a_omega

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_connectivity_after_removal(G, nodes=None, edges=None):
        G_test = G.copy()
        if nodes:
            G_test.remove_nodes_from(nodes)
        if edges:
            G_test.remove_edges_from(edges)
        if not nx.is_connected(G_test):
            raise ValueError(
                "The requested removal would disconnect the graph."
            )

    def __repr__(self) -> str:
        return (
            f"HyperMapEmbedder("
            f"d={self.d}, gamma={self.gamma}, "
            f"T={self.T}, zeta={self.zeta})"
        )
