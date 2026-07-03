"""
HYDRA embedder.

Implements a hyperbolic embedding of a graph distance matrix using the
HYDRA algorithm (Keller-Ressel & Nargang, 2021) within the
encoder-decoder framework of Chami et al. (2022).

Model
-----
::

    Structural similarity : All-pairs shortest-path distance matrix
                             D_ij = d_G(i, j)
    Encoder               : Spectral decomposition of the hyperbolic
                             Gram matrix  A_ij = cosh(sqrt(k) * D_ij)
                             → Poincaré disk coordinates (r, directional)
    Decoder               : Pairwise Poincaré distances
                             d_H(x_i, x_j)
    Loss                  : Stress (RMS distance error)
                             sqrt(sum_{i<j} (d_H(x_i,x_j) - D_ij)^2)

The embedding is closed-form (non-gradient). Curvature ``k`` may be
fixed by the caller or optimised via scalar minimisation of the stress.

References
----------
Keller-Ressel & Nargang, *Hydra: a method for strain-minimizing
hyperbolic embedding of network- and distance-based data*,
Journal of Complex Networks, 2021.
"""

from __future__ import annotations

import warnings
from math import pi
from typing import Optional

import networkx as nx
import numpy as np
import torch
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar

from hypegrl.embedders.base import HyperbolicEmbedder
from hypegrl.manifolds.poincare import poincare_distances_polar


# ---------------------------------------------------------------------------
# Internal spectral core
# ---------------------------------------------------------------------------

def _hydra_fixed_curvature(
    D: np.ndarray,
    dim: int,
    curvature: float,
    alpha: float,
    equi_adj: float,
    isotropic_adj: bool,
) -> dict:
    """
    Core HYDRA spectral embedding for a fixed curvature value.

    Parameters
    ----------
    D:
        ``(N, N)`` symmetric pairwise distance matrix (zeros on diagonal).
    dim:
        Target embedding dimension.
    curvature:
        Positive hyperbolic curvature ``k``.
    alpha:
        Radial rescaling factor (> 1 pushes points toward the boundary).
    equi_adj:
        Strength of equiangular angular adjustment; only used when
        ``dim == 2``.
    isotropic_adj:
        If ``True``, ignore the magnitude of the negative eigenvectors
        and use only their directions for the angular part.

    Returns
    -------
    dict with keys ``r``, ``directional``, ``theta`` (2-D only),
    ``stress``, ``curvature``, ``dim``.
    """
    n = D.shape[0]

    # --- Hyperbolic Gram matrix -------------------------------------------
    A = np.cosh(np.sqrt(curvature) * D)
    A_max = np.max(A)

    if A_max > 1e8:
        warnings.warn(
            "Gram matrix contains values > 1e8. "
            "Consider a smaller curvature or rescaled distances.",
            RuntimeWarning,
            stacklevel=3,
        )
    if np.isinf(A_max):
        raise ValueError(
            "Gram matrix contains infinite values. "
            "Reduce curvature or rescale distances."
        )

    # --- Spectral decomposition -------------------------------------------
    # eigh returns eigenvalues in ascending order.
    spec_vals, spec_vecs = eigh(A)

    # Largest eigenvector → Lorentzian (time-like) coordinate x0.
    lambda0 = spec_vals[-1]
    x0 = spec_vecs[:, -1] * np.sqrt(lambda0)
    if x0[0] < 0:
        x0 = -x0

    # Bottom `dim` eigenvectors → spatial (space-like) directions.
    X_dir = spec_vecs[:, :dim]          # (N, dim)
    spec_tail = spec_vals[:dim]         # negative or near-zero

    # --- Directional part -------------------------------------------------
    if not isotropic_adj:
        # Weight directions by the magnitude of the negative spectrum.
        X_dir = X_dir @ np.diag(np.sqrt(np.maximum(-spec_tail, 0)))

    norms = np.linalg.norm(X_dir, axis=1, keepdims=True)
    # Avoid division by zero for degenerate points.
    norms = np.where(norms == 0.0, 1.0, norms)
    directional = X_dir / norms         # (N, dim)  unit vectors

    # --- Radial part ------------------------------------------------------
    # Per Keller-Ressel & Nargang (2021) Theorem 3.1 step B2:
    # xmin := min(1, x0_1, ..., x0_n).  Including 1 ensures that for any
    # valid H^d embedding (where all x0_i >= 1) the formula reduces to the
    # exact stereographic projection sqrt((alpha*x0-1)/(alpha*x0+1)), giving
    # zero stress.  The R reference implementation omits the 1 (a bug that is
    # invisible on graph data where x0_min > 1 but breaks exact recovery).
    x_min = min(1.0, float(np.min(x0)))
    r = np.sqrt(np.maximum(0.0, (alpha * x0 - x_min) / (alpha * x0 + x_min)))

    # --- Angular adjustment (2-D only) ------------------------------------
    theta = None
    if dim == 2:
        theta = np.arctan2(X_dir[:, 1], X_dir[:, 0])
        if equi_adj > 0.0:
            delta = 2 * pi / n
            angles = np.linspace(-pi, pi - delta, n)
            ranks = np.argsort(np.argsort(theta))
            theta_equi = angles[ranks]
            theta = (1 - equi_adj) * theta + equi_adj * theta_equi
            directional = np.stack(
                (np.cos(theta), np.sin(theta)), axis=1
            )

    # --- Stress -----------------------------------------------------------
    stress = _compute_stress(r, directional, curvature, D)

    # --- Strain -----------------------------------------------------------
    # Per Keller-Ressel & Nargang (2019), strain is obtained from stress by
    # replacing distances with their hyperbolic cosines, i.e.
    #   strain = || A_hat - A ||_F
    # where A_ij = cosh(sqrt(k) * D_ij) is the original Gram matrix and
    # A_hat is its rank-(dim+1) reconstruction. Since the eigendecomposition
    # of A minimises exactly this Frobenius residual, the achieved minimum is
    #   strain = sqrt( sum_{discarded} lambda_i^2 )   (unnormalised)
    # Kept eigenvalues: bottom-dim (spatial directions) + top-1 (Lorentzian x0).
    kept_idx = list(range(dim)) + [len(spec_vals) - 1]
    sq_kept  = np.sum(spec_vals[kept_idx] ** 2)
    sq_all   = np.sum(spec_vals ** 2)
    strain   = float(np.sqrt(max(0.0, sq_all - sq_kept)))

    return {
        "r":           r,
        "directional": directional,
        "theta":       theta,
        "stress":      stress,
        "strain":      strain,
        "curvature":   curvature,
        "dim":         dim,
    }


def _compute_stress(
    r: np.ndarray,
    directional: np.ndarray,
    curvature: float,
    D: np.ndarray,
) -> float:
    """RMS stress between reconstructed and original distances."""
    D_hat = poincare_distances_polar(r, directional, curvature)
    mask  = np.triu(np.ones_like(D, dtype=bool), k=1)
    return float(np.sqrt(np.sum((D_hat[mask] - D[mask]) ** 2)))


def _poincare_cartesian_to_polar(
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decompose Cartesian Poincaré coordinates into ``(r, directional)``.

    Parameters
    ----------
    X:
        ``(N, d)`` Cartesian Poincaré coordinates.

    Returns
    -------
    r:
        ``(N,)`` radial coordinates.
    directional:
        ``(N, d)`` unit direction vectors.
    """
    r = np.linalg.norm(X, axis=1)                      # (N,)
    norms = np.where(r == 0.0, 1.0, r)
    directional = X / norms[:, None]                    # (N, d)
    return r, directional


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class HydraEmbedder(HyperbolicEmbedder):
    """
    HYDRA (Hyperbolic Distortion-minimising Radial Algorithm) graph embedder.

    A closed-form spectral method that embeds a graph into the Poincaré
    disk (or ball for ``dim > 2``) by decomposing the hyperbolic Gram
    matrix of the all-pairs shortest-path distance matrix.

    Since HYDRA is non-gradient-based, ``unknown_edges`` are not jointly
    optimised; they are zero-imputed with a warning, consistent with the
    convention established for non-gradient methods in this framework.

    Parameters
    ----------
    dim:
        Embedding dimension (2 for Poincaré disk, higher for ball).
    curvature:
        Fixed hyperbolic curvature ``k > 0``. Pass ``None`` to let
        :meth:`fit` optimise ``k`` via scalar minimisation of the stress.
    alpha:
        Radial rescaling factor (> 1 spreads points toward the boundary,
        reducing crowding at the origin).
    equi_adj:
        Equiangular adjustment strength in ``[0, 1]``; only active when
        ``dim == 2``. Blends spectral angles toward uniformly spaced
        angles, reducing angular overlap.
    weight:
        Edge attribute to use as distance. If ``None`` all edges have
        unit weight and Dijkstra operates on the unweighted graph.

    Examples
    --------
    >>> import networkx as nx
    >>> from hypegrl.embedders.hydra import HydraEmbedder
    >>> G = nx.karate_club_graph()
    >>> embedder = HydraEmbedder(dim=2, curvature=1.0)
    >>> embedder.fit(G)
    HydraEmbedder(dim=2, curvature=1.0)
    >>> X = embedder.embeddings()
    >>> X.shape
    (34, 2)
    """

    def __init__(
        self,
        dim:       int            = 2,
        curvature: Optional[float] = 1.0,
        alpha:     float          = 1.1,
        equi_adj:  float          = 0.5,
        weight:    Optional[str]  = None,
    ):
        self.dim       = dim
        self.curvature = curvature
        self.alpha     = alpha
        self.equi_adj  = equi_adj
        self.weight    = weight

        # Fitted state
        self._X:                Optional[np.ndarray] = None   # (N, dim) Cartesian
        self._r:                Optional[np.ndarray] = None   # (N,) radial
        self._directional:      Optional[np.ndarray] = None   # (N, dim) unit dirs
        self._theta:            Optional[np.ndarray] = None   # (N,) angles (2-D)
        self._D:                Optional[np.ndarray] = None   # (N, N) shortest-path distances
        self._curvature_fitted: Optional[float]      = None   # curvature used
        self._stress:           Optional[float]      = None
        self._strain:           Optional[float]      = None
        self._G:                Optional[nx.Graph]   = None

    # ------------------------------------------------------------------
    # HyperbolicEmbedder interface
    # ------------------------------------------------------------------

    def fit_distance(self, D: np.ndarray) -> "HydraEmbedder":
        """
        Embed an arbitrary distance matrix directly into hyperbolic space using HYDRA.

        Parameters
        ----------
        D:
            ``(N, N)`` symmetric pairwise distance matrix (zeros on diagonal).

        Returns
        -------
        self
        """
        # --- Run spectral embedding ---------------------------------------
        isotropic_adj = (self.dim != 2)

        if self.curvature is None:
            result = self._optimize_curvature(D, isotropic_adj)
        else:
            result = _hydra_fixed_curvature(
                D, self.dim, self.curvature,
                self.alpha, self.equi_adj, isotropic_adj,
            )

        # --- Store results ------------------------------------------------
        self._r             = result["r"]
        self._directional   = result["directional"]
        self._theta         = result["theta"]
        self._curvature_fitted = result["curvature"]
        self._stress        = result["stress"]
        self._strain        = result["strain"]
        self._D             = D
        self._G             = None  # Reset graph state since input is a raw distance matrix
        self._nodes         = list(range(D.shape[0]))  # rows follow matrix index order

        # Cartesian Poincaré coordinates: x_i = r_i * dir_i  ∈  B^dim
        self._X = self._r[:, None] * self._directional  # (N, dim)

        return self

    def fit(
        self,
        G: nx.Graph,
        unknown_edges: Optional[list[tuple[int, int]]] = None,
        X_init: Optional[np.ndarray] = None,
    ) -> "HydraEmbedder":
        """
        Embed the graph ``G`` into hyperbolic space using HYDRA.

        Parameters
        ----------
        G:
            Input graph.  Edge weights are used when ``self.weight`` is
            set and present; otherwise unit weights are assumed.
        unknown_edges:
            Unsupported for non-gradient methods.  A warning is issued
            and these edges are zero-imputed (treated as absent).
        X_init:
            Ignored.  HYDRA is a closed-form method with no gradient step.

        Returns
        -------
        self
        """
        # --- Handle unknown edges (non-gradient fallback) ----------------
        if unknown_edges:
            warnings.warn(
                "HydraEmbedder is a non-gradient method and does not support "
                "joint optimisation of unknown edges. "
                "Unknown edges are zero-imputed (treated as absent).",
                UserWarning,
                stacklevel=2,
            )

        # --- Compute structural similarity (shortest-path distances) ------
        D = self._shortest_path_matrix(G)

        # --- Run spectral embedding machinery via distance matrix --------
        self.fit_distance(D)

        # --- Keep track of the graph source -------------------------------
        self._G = G
        self._nodes = list(G.nodes())  # rows follow G.nodes() (no reorder)

        return self

    def embeddings(self) -> np.ndarray:
        """
        Return ``(N, dim)`` Cartesian Poincaré coordinates.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit` or :meth:`fit_distance`.
        """
        if self._X is None:
            raise RuntimeError("Call fit() or fit_distance() before embeddings().")
        return self._X

    def structural_similarity(self, G: nx.Graph) -> np.ndarray:
        """
        Compute the all-pairs shortest-path distance matrix ``D = s(A)``.

        Parameters
        ----------
        G:
            Input graph.

        Returns
        -------
        ``(N, N)`` NumPy array of shortest-path distances.
        """
        return self._shortest_path_matrix(G)

    def decode(self, X: np.ndarray) -> np.ndarray:
        """
        Reconstruct the pairwise hyperbolic distance matrix from embeddings.

        ``Dec(X)_{ij} = d_H(x_i, x_j)``  in the Poincaré model.

        Parameters
        ----------
        X:
            ``(N, dim)`` Cartesian Poincaré coordinates (i.e. the output
            of :meth:`embeddings`).

        Returns
        -------
        ``(N, N)`` symmetric distance matrix.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit` or :meth:`fit_distance` (curvature unknown).
        """
        if self._curvature_fitted is None:
            raise RuntimeError("Call fit() or fit_distance() before decode().")
        r, directional = _poincare_cartesian_to_polar(X)
        return poincare_distances_polar(r, directional, self._curvature_fitted)

    def distance(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Stress loss: RMS error between decoded and structural distances.

        .. math::

            \\mathcal{L} = \\sqrt{\\sum_{i < j}
                (\\hat{d}_{ij} - D_{ij})^2}

        Although HYDRA does not use gradient-based optimisation, this
        method is required by the :class:`~hypegrl.embedders.base.HyperbolicEmbedder`
        interface and may be used for evaluation.

        Parameters
        ----------
        X:
            ``(N, dim)`` Poincaré embeddings as a torch tensor.
        A:
            ``(N, N)`` adjacency matrix as a torch tensor. Shortest-path
            distances are recomputed internally via NetworkX using the
            stored graph.

        Returns
        -------
        Scalar torch tensor containing the stress.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if self._G is None or self._curvature_fitted is None:
            raise RuntimeError("Call fit() before distance().")

        X_np = X.detach().cpu().numpy()
        D    = self._shortest_path_matrix(self._G)
        D_hat = self.decode(X_np)

        mask = np.triu(np.ones_like(D, dtype=bool), k=1)
        stress = float(np.sqrt(np.sum((D_hat[mask] - D[mask]) ** 2)))
        return torch.tensor(stress, dtype=X.dtype, device=X.device)

    # ------------------------------------------------------------------
    # Capability flags
    # ------------------------------------------------------------------

    def is_gradient_based(self) -> bool:
        return False

    def is_generative(self) -> bool:
        return False

    def supports_update(self) -> bool:
        return False

    def supports_node_update(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Extra accessors
    # ------------------------------------------------------------------

    @property
    def stress(self) -> Optional[float]:
        """RMS stress of the fitted embedding (``None`` before fitting)."""
        return self._stress

    @property
    def strain(self) -> Optional[float]:
        """
        Frobenius residual of the rank-(dim+1) spectral approximation:

            strain = || A_hat - A ||_F = sqrt( sum_{discarded} lambda_i^2 )

        where ``A_ij = cosh(sqrt(k) * D_ij)`` is the hyperbolic Gram matrix.
        Per Keller-Ressel & Nargang (2019), strain is the objective that HYDRA
        minimises exactly via eigendecomposition; it equals the stress functional
        with all distances passed through ``cosh``. Independent of the Riemannian
        refinement in HYDRA+.
        """
        return self._strain

    @property
    def fitted_curvature(self) -> Optional[float]:
        """Curvature used in the fitted embedding."""
        return self._curvature_fitted

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _shortest_path_matrix(self, G: nx.Graph) -> np.ndarray:
        """
        Build the all-pairs shortest-path distance matrix.

        Nodes are assumed to be labelled ``0 … N-1`` (or whatever integer
        ordering NetworkX assigns via ``G.nodes()``).
        """
        nodes  = list(G.nodes())
        n      = len(nodes)
        index  = {v: i for i, v in enumerate(nodes)}
        D      = np.zeros((n, n), dtype=float)

        lengths = dict(
            nx.all_pairs_dijkstra_path_length(G, weight=self.weight)
        )
        for u, dists in lengths.items():
            for v, dist in dists.items():
                D[index[u], index[v]] = dist

        return D

    def _optimize_curvature(
        self,
        D: np.ndarray,
        isotropic_adj: bool,
    ) -> dict:
        """
        Find the curvature minimising the embedding stress via scalar
        optimisation, then return the corresponding embedding.

        The search interval is ``(eps, (8 / max(D))^2)`` — the same
        heuristic used in the original HYDRA R package.
        """
        eps     = np.finfo(float).eps
        k_upper = (8.0 / np.max(D)) ** 2
        k_bounds = (eps, k_upper)

        def _stress_at_k(k: float) -> float:
            return _hydra_fixed_curvature(
                D, self.dim, k,
                self.alpha, self.equi_adj, isotropic_adj,
            )["stress"]

        result  = minimize_scalar(_stress_at_k, bounds=k_bounds, method="bounded")
        k_opt   = result.x
        s_opt   = result.fun

        # Never worse than the default k = 1 baseline.
        s_k1 = _stress_at_k(1.0)
        if s_k1 < s_opt:
            k_opt = 1.0

        return _hydra_fixed_curvature(
            D, self.dim, k_opt,
            self.alpha, self.equi_adj, isotropic_adj,
        )

    def __repr__(self) -> str:
        k = self.curvature if self.curvature is not None else "optimized"
        return f"HydraEmbedder(dim={self.dim}, curvature={k})"