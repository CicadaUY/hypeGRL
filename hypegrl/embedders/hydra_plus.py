"""
HYDRA+ embedder.

Extends the closed-form HYDRA embedding with a Riemannian gradient
refinement step, using the HYDRA solution as a warm start. This follows the
encoder-decoder framework of Chami et al. (2022), where the spectral step acts
as the encoder and the Riemannian optimiser refines the embeddings to further
minimise the stress.

Where HYDRA minimises the *strain* (the Frobenius residual on the
cosh-linearised Gram matrices, which is what makes it an exactly solvable
eigenproblem), this refinement minimises the *stress* — the error in the actual
distances — which is the quantity one really cares about.

Model
-----
::

    Structural similarity : Pairwise distance matrix (from Graph or Matrix),
                             scaled by the curvature: D_ij * sqrt(k)
    Encoder               : HYDRA spectral step (warm start), then Riemannian
                             Adam over the chosen Representation (chart)
    Decoder               : Pairwise hyperbolic distances  d_H(x_i, x_j)
    Loss                  : Raw (squared) stress over the upper triangle
                             L(X) = sum_{i<j} (d_H(x_i,x_j) - D_ij*sqrt(k))^2

The square is deliberate: it shares its minimiser with the rooted stress
``sqrt(L)`` reported by the ``stress`` property, but has a smooth gradient
everywhere, whereas ``d/dL sqrt(L) = 1/(2 sqrt(L))`` blows up exactly as the fit
approaches perfect.

References
----------
Keller-Ressel & Nargang, *Hydra: a method for strain-minimizing
hyperbolic embedding of network- and distance-based data*,
Journal of Complex Networks, 2021.
"""

from __future__ import annotations

import warnings
from typing import Optional

import networkx as nx
import numpy as np
import torch

# Cleanly reuse the parent class and internal helper from hydra
from hypegrl.embedders.hydra import HydraEmbedder, _poincare_cartesian_to_polar
from hypegrl.inference.riemannian_optimizer import riemannian_optimize
from hypegrl.representations import (
    BallRepresentation,
    ExactPolarRepresentation,
    HyperboloidRepresentation,
    PolarRepresentation,
    build_representation,
)

# Chart in which the Step-2 gradient refinement runs. HYDRA+'s stress loss is a
# function of the pairwise distance only, so the chart is neutral — polar is the
# default (exact at all radii; the ball saturates past r≈12), matching the other
# neutral methods. The original HYDRA+ even refines on the hyperboloid; the ball
# was only an implementation choice.
_REPRESENTATIONS = {
    "polar": PolarRepresentation,
    "exact_polar": ExactPolarRepresentation,
    "ball": BallRepresentation,
    "hyperboloid": HyperboloidRepresentation,
}


# ---------------------------------------------------------------------------
# Stress loss (differentiable, used during Riemannian optimisation)
# ---------------------------------------------------------------------------

def _stress_loss_from_dist(
    D_hat: torch.Tensor,
    D_scaled: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Differentiable stress from a pairwise-distance matrix ``D_hat`` (e.g.
    ``rep.dist()``) against the target distances.

    The curvature ``k`` is absorbed into the target distances: since
    ``d_k(x,y) = d_1(x,y) / sqrt(k)`` and every chart here has unit curvature,
    minimising ``sum_{i<j} (d_1 - D_ij*sqrt(k))^2`` is equivalent (up to the
    constant ``1/k``) to the true stress. ``D_scaled`` must be ``D * sqrt(k)``.
    """
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
        representation: str          = "polar",
    ):
        super().__init__(
            dim=dim,
            curvature=curvature,
            alpha=alpha,
            equi_adj=equi_adj,
            weight=weight,
        )
        if representation not in _REPRESENTATIONS:
            raise ValueError(
                f"representation must be one of {sorted(_REPRESENTATIONS)}; "
                f"got {representation!r}.")
        self.lr             = lr
        self.n_steps        = n_steps
        self.grad_clip      = grad_clip
        self.log_every      = log_every
        self.device         = device
        self.random_state   = random_state
        self.representation = representation

        self._rep = None                          # fitted Representation
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
        # Build the representation (chosen chart) from the ball warm start (or an
        # incoming Representation of any chart). The stress loss pulls rep.dist()
        # — unit curvature in every chart — so the sqrt(k) target scaling below is
        # unchanged; embeddings() reads back the ball projection.
        n     = len(D)
        scale = float(np.sqrt(self._curvature_fitted))
        s_A    = D * scale
        mask_t = torch.as_tensor(np.triu(np.ones((n, n), dtype=bool), k=1))

        rep = build_representation(
            _REPRESENTATIONS[self.representation], X_warm,
            input_chart="ball", device=self.device,
        )

        def loss_fn(rep_, s_A_t: torch.Tensor) -> torch.Tensor:
            return _stress_loss_from_dist(rep_.dist(), s_A_t, mask_t.to(s_A_t.device))

        result = riemannian_optimize(
            representation = rep,
            s_A            = s_A,
            loss_fn        = loss_fn,
            lr             = self.lr,
            n_steps        = self.n_steps,
            grad_clip      = self.grad_clip,
            log_every      = self.log_every,
            device         = self.device,
        )
        X_refined    = rep.to_ball().detach().cpu().numpy()
        loss_history = result["loss_history"]

        # --- Step 3: Update stored state ----------------------------------
        self._X            = X_refined
        self._rep          = rep
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