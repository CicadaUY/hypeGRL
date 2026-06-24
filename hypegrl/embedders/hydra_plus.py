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

    def fit_distance(self, D: np.ndarray) -> "HydraPlusEmbedder":
        """
        Embed an arbitrary distance matrix directly into hyperbolic space using HYDRA+,
        applying Riemannian gradient refinement on top of the spectral solution.

        Parameters
        ----------
        D:
            ``(N, N)`` symmetric pairwise distance matrix (zeros on diagonal).

        Returns
        -------
        self
        """
        # --- Step 1: HYDRA spectral warm start via distance matrix --------
        super().fit_distance(D)

        self._stress_init = self._stress  # Snapshot spectral performance
        self._strain_init = self._strain  # Strain is fixed by the spectral step

        # --- Step 2: Riemannian refinement --------------------------------
        ball  = geoopt.PoincareBall(c=1.0)
        n     = len(D)
        scale = float(np.sqrt(self._curvature_fitted))

        # Project warm start strictly inside the ball before optimising.
        X_init_proj = ball.projx(
            torch.tensor(self._X, dtype=torch.float64)
        ).numpy()

        # Structural similarity: scaled distances so that d_1(x,y) matches
        # D*sqrt(k), avoiding a curvature factor inside the loss.
        s_A = D * scale

        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        mask_t = torch.as_tensor(mask)

        def loss_fn(X: torch.Tensor, s_A_t: torch.Tensor) -> torch.Tensor:
            return _stress_loss(X, s_A_t, ball, mask_t.to(X.device))

        result = riemannian_optimize(
            X_init    = X_init_proj,
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

        # --- Step 3: Update stored state with refined coordinates ---------
        self._X            = X_refined
        self._loss_history = loss_history
        self._D            = D
        self._G            = None  # Reset graph reference since input is a matrix

        # REUSE: Reconstruct distances efficiently using inherited decode()
        D_hat = self.decode(X_refined)
        mask  = np.triu(np.ones(len(D), dtype=bool), k=1)

        # Compute post-refinement true stress directly
        self._stress  = float(np.sqrt(np.sum((D_hat[mask] - D[mask])**2)))

        # Compute post-refinement strain directly
        A_hat = np.cosh(np.sqrt(self._curvature_fitted) * D_hat)
        A     = np.cosh(np.sqrt(self._curvature_fitted) * D)
        self._strain   = float(np.sqrt(np.sum((A_hat - A)**2)))

        # REUSE: Clean polar coordinates refresh using shared helper function
        self._r, self._directional = _poincare_cartesian_to_polar(X_refined)
        if self.dim == 2:
            self._theta    = np.arctan2(
                self._directional[:, 1], self._directional[:, 0]
            )

        return self

    def fit(
        self,
        G: nx.Graph,
        unknown_edges: Optional[list[tuple[int, int]]] = None,
    ) -> "HydraPlusEmbedder":
        """
        Fit HYDRA+ embeddings from a graph: spectral warm start, then Riemannian refinement.

        Parameters
        ----------
        G:
            Input graph.
        unknown_edges:
            Not supported for HYDRA+; a warning is issued and they are
            zero-imputed.

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
        self.fit_distance(D)

        # Keep track of graph source for evaluation compliance
        self._G = G

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