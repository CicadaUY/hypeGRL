"""
HYDRA+ embedder.

Extends the closed-form HYDRA embedding with a Riemannian gradient
refinement step on the Poincaré ball, using the HYDRA solution as a
warm start. This follows the encoder-decoder framework of Chami et al.
(2022), where the spectral step acts as the encoder and the Riemannian
optimiser refines the embeddings to further minimise the stress.

Model
-----
::

    Structural similarity : Pairwise distance matrix (from Graph or Matrix)
                             D_ij
    Encoder               : HYDRA spectral step (warm start)
                             → X_init ∈ B^dim
    Refinement            : Riemannian Adam on (B^dim, g_k) minimising
                             L(X) = sum_{i<j} (d_H(x_i,x_j) - D_ij)^2
    Decoder               : Pairwise Poincaré distances
                             d_H(x_i, x_j)
    Loss                  : Stress (squared distance error)
                             sum_{i<j} (d_H(x_i,x_j) - D_ij)^2

References
----------
Keller-Ressel & Nargang, *Hydra: a method for strain-minimizing
hyperbolic embedding of network- and distance-based data*,
Journal of Complex Networks, 2021.
"""

from __future__ import annotations

import warnings
from typing import Optional

import geoopt
import networkx as nx
import numpy as np
import torch

# Cleanly reuse the parent class and internal helper from hydra
from hypegrl.embedders.hydra import HydraEmbedder, _poincare_cartesian_to_polar
from hypegrl.inference.riemannian_optimizer import riemannian_optimize


# ---------------------------------------------------------------------------
# Stress loss (differentiable, used during Riemannian optimisation)
# ---------------------------------------------------------------------------

def _stress_loss(
    X: torch.Tensor,
    D_scaled: torch.Tensor,
    ball: geoopt.PoincareBall,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Differentiable stress loss on the unit Poincaré ball (c=1).

    We always optimise on the **unit** ball (``ball`` has ``c=1``) so that
    HYDRA's warm-start coordinates, which live in the unit disk, are valid
    without any rescaling. The curvature ``k`` is absorbed into the target
    distances: since ``d_k(x,y) = d_1(x,y) / sqrt(k)``, minimising

        sum_{i<j} (d_1(x_i,x_j) - D_ij * sqrt(k))^2

    is equivalent (up to the constant factor 1/k) to minimising the true
    stress ``sum_{i<j} (d_k(x_i,x_j) - D_ij)^2``. ``D_scaled`` must
    therefore be ``D * sqrt(k)``.
    """
    # geoopt broadcasts: dist(X[i], X[j]) over all pairs → (N, N)
    D_hat = ball.dist(X.unsqueeze(1), X.unsqueeze(0))  # (N, N)
    return torch.sum((D_hat[mask] - D_scaled[mask]) ** 2)


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class HydraPlusEmbedder(HydraEmbedder):
    """
    HYDRA+ graph embedder: spectral warm start + Riemannian refinement.

    Uses the closed-form HYDRA spectral embedding as initialisation,
    then refines the positions via Riemannian Adam on the Poincaré ball
    ``(B^dim, g_k)`` to further minimise the stress.

    Inherits all HYDRA parameters (``dim``, ``curvature``, ``alpha``,
    ``equi_adj``, ``weight``) and adds optimiser hyperparameters.
    """

    def __init__(
        self,
        dim:          int            = 2,
        curvature:    Optional[float] = 1.0,
        alpha:        float          = 1.1,
        equi_adj:     float          = 0.5,
        weight:       Optional[str]  = None,
        lr:           float          = 1e-2,
        n_steps:      int            = 500,
        grad_clip:    float          = 10.0,
        log_every:    int            = 50,
        device:       str            = "cpu",
        random_state: Optional[int]  = None,
    ):
        super().__init__(
            dim=dim,
            curvature=curvature,
            alpha=alpha,
            equi_adj=equi_adj,
            weight=weight,
        )
        self.lr           = lr
        self.n_steps      = n_steps
        self.grad_clip    = grad_clip
        self.log_every    = log_every
        self.device       = device
        self.random_state = random_state

        # Additional fitted state (refinement-specific)
        self._loss_history: Optional[list[float]] = None
        self._stress_init:  Optional[float]       = None  # HYDRA stress before refinement
        self._strain_init:  Optional[float]       = None  # HYDRA strain at warm start
        self._strain:       Optional[float]       = None  # strain of the refined embedding

    # ------------------------------------------------------------------
    # HyperbolicEmbedder interface
    # ------------------------------------------------------------------

    def fit_distance(
        self,
        D: np.ndarray,
        X_init: Optional[np.ndarray] = None,
    ) -> "HydraPlusEmbedder":
        """
        Embed an arbitrary distance matrix directly into hyperbolic space using HYDRA+,
        applying Riemannian gradient refinement on top of the warm start.

        Parameters
        ----------
        D:
            ``(N, N)`` symmetric pairwise distance matrix (zeros on diagonal).
        X_init:
            ``(N, dim)`` initial Poincaré coordinates for the Riemannian
            refinement step.  If ``None``, the HYDRA spectral solution is
            used as the warm start (default behaviour).  When provided, the
            spectral step is skipped; curvature defaults to ``self.curvature``
            or the previously fitted value (falls back to ``1.0`` if neither
            is available).

        Returns
        -------
        self
        """
        mask_np = np.triu(np.ones(len(D), dtype=bool), k=1)

        if X_init is None:
            # --- Step 1a: HYDRA spectral warm start -----------------------
            super().fit_distance(D)
            self._stress_init = self._stress
            self._strain_init = self._strain
            X_warm = self._X
        else:
            # --- Step 1b: external warm start -----------------------------
            # Determine curvature: prefer previously fitted, then constructor
            # value, fall back to 1.0.
            if self._curvature_fitted is None:
                self._curvature_fitted = self.curvature if self.curvature is not None else 1.0
            self._D = D
            D_hat = self.decode(X_init)
            self._stress_init = float(np.sqrt(np.sum((D_hat[mask_np] - D[mask_np]) ** 2)))
            A_hat_i = np.cosh(np.sqrt(self._curvature_fitted) * D_hat)
            A_i     = np.cosh(np.sqrt(self._curvature_fitted) * D)
            self._strain_init = float(np.sqrt(np.sum((A_hat_i - A_i) ** 2)))
            X_warm = X_init

        # --- Step 2: Riemannian refinement (shared) -----------------------
        ball  = geoopt.PoincareBall(c=1.0)
        n     = len(D)
        scale = float(np.sqrt(self._curvature_fitted))

        X_proj = ball.projx(torch.tensor(X_warm, dtype=torch.float64)).numpy()
        s_A    = D * scale
        mask_t = torch.as_tensor(np.triu(np.ones((n, n), dtype=bool), k=1))

        def loss_fn(X: torch.Tensor, s_A_t: torch.Tensor) -> torch.Tensor:
            return _stress_loss(X, s_A_t, ball, mask_t.to(X.device))

        result = riemannian_optimize(
            X_init    = X_proj,
            s_A       = s_A,
            loss_fn   = loss_fn,
            manifold  = ball,
            lr        = self.lr,
            n_steps   = self.n_steps,
            grad_clip = self.grad_clip,
            log_every = self.log_every,
            device    = self.device,
        )
        X_refined    = result["X"]
        loss_history = result["loss_history"]

        # --- Step 3: Update stored state ----------------------------------
        self._X            = X_refined
        self._loss_history = loss_history
        self._D            = D
        self._G            = None
        self._nodes        = list(range(D.shape[0]))  # rows follow matrix index order

        D_hat = self.decode(X_refined)
        self._stress = float(np.sqrt(np.sum((D_hat[mask_np] - D[mask_np]) ** 2)))
        A_hat = np.cosh(np.sqrt(self._curvature_fitted) * D_hat)
        A     = np.cosh(np.sqrt(self._curvature_fitted) * D)
        self._strain = float(np.sqrt(np.sum((A_hat - A) ** 2)))

        self._r, self._directional = _poincare_cartesian_to_polar(X_refined)
        if self.dim == 2:
            self._theta = np.arctan2(self._directional[:, 1], self._directional[:, 0])

        return self

    def fit(
        self,
        G: nx.Graph,
        unknown_edges: Optional[list[tuple[int, int]]] = None,
        X_init: Optional[np.ndarray] = None,
    ) -> "HydraPlusEmbedder":
        """
        Fit HYDRA+ embeddings from a graph.

        Parameters
        ----------
        G:
            Input graph.
        unknown_edges:
            Not supported for HYDRA+; a warning is issued and they are
            zero-imputed.
        X_init:
            ``(N, dim)`` initial Poincaré coordinates.  If ``None``, the
            HYDRA spectral solution is used as the warm start.  If provided,
            the spectral step is skipped and gradient refinement starts from
            ``X_init`` directly.

        Returns
        -------
        self
        """
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        if unknown_edges:
            warnings.warn(
                "HydraPlusEmbedder does not yet support joint optimisation "
                "of unknown edges (the stress loss requires a fixed distance "
                "matrix). Unknown edges are zero-imputed.",
                UserWarning,
                stacklevel=2,
            )

        # Compute structural similarity shortest-path matrix exactly once
        D = self._shortest_path_matrix(G)

        # Delegate everything to the clean matrix-based embedding workflow
        self.fit_distance(D, X_init=X_init)

        # Keep track of graph source for evaluation compliance
        self._G = G
        self._nodes = list(G.nodes())  # rows follow G.nodes() (no reorder)

        return self

    def distance(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """Differentiable stress loss for HYDRA+."""
        if self._D is None or self._curvature_fitted is None:
            raise RuntimeError("Call fit() before distance().")

        ball  = geoopt.PoincareBall(c=1.0)
        scale = float(np.sqrt(self._curvature_fitted))
        D_t   = torch.tensor(self._D * scale, dtype=X.dtype, device=X.device)
        n     = X.shape[0]
        mask  = torch.triu(
            torch.ones(n, n, dtype=torch.bool, device=X.device), diagonal=1
        )
        return _stress_loss(X, D_t, ball, mask) / self._curvature_fitted

    # ------------------------------------------------------------------
    # Capability flags
    # ------------------------------------------------------------------

    def is_gradient_based(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Public Accessor Properties (Fixes AttributeError)
    # ------------------------------------------------------------------

    @property
    def loss_history(self) -> Optional[list[float]]:
        """
        Scaled loss ``sum_{i<j}(d_1(x_i,x_j) - D_ij*sqrt(k))^2`` at each
        Riemannian Adam step. Divide by ``fitted_curvature`` to recover the
        true squared stress at each step.
        """
        return self._loss_history

    @property
    def stress_init(self) -> Optional[float]:
        """Stress of the HYDRA warm start before Riemannian refinement."""
        return self._stress_init

    @property
    def strain(self) -> Optional[float]:
        """
        Frobenius strain of the *refined* embedding:
            strain = || cosh(sqrt(k) * D_hat) - cosh(sqrt(k) * D) ||_F
        """
        return self._strain

    @property
    def strain_init(self) -> Optional[float]:
        """
        Frobenius strain of the HYDRA spectral warm start (before refinement):
            strain_init = || cosh(sqrt(k) * D_hat_init) - cosh(sqrt(k) * D) ||_F
        """
        return self._strain_init

    def __repr__(self) -> str:
        k = self.curvature if self.curvature is not None else "optimized"
        return (
            f"HydraPlusEmbedder(dim={self.dim}, curvature={k}, "
            f"lr={self.lr}, n_steps={self.n_steps})"
        )