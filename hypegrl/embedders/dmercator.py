"""
D-Mercator graph embedder.

Embeds a graph into hyperbolic space H^{D+1} (Poincaré ball B^d, d = D+1)
following the geometric network model of Jankowski et al. (2023). The original
D-Mercator pipeline produces the warm start in the S^D model; the embedding is
then refined directly on the **Poincaré ball**, where both the angular position
and the radial ("popularity") coordinate of every node move jointly under a
hyperbolic-distance Fermi-Dirac likelihood.

Encoder-decoder framework
-------------------------
::

    Structural similarity : Binary adjacency matrix A
    Encoder (init)        : full original D-Mercator pipeline
                            (``_dmercator_init.dmercator_init``):
                            κ/β inference → S^D-corrected Laplacian Eigenmaps
                            → likelihood-maximisation refinement → κ readjust
    Warm start            : x_i = tanh(r_i/2) · v_i ∈ B^d
                            with r_i = R̂ − (2/D) ln(κ_i/κ_min)  (Eq. 7)
    Refinement            : Riemannian Adam on the Poincaré ball minimising
                            the Fermi-Dirac NLL on hyperbolic distances
    Decoder               : p_ij = 1 / (1 + e^{(β/2)(d_H(x_i,x_j) − R̂)})
    Loss                  : -Σ_{i<j}[a_ij ln p_ij + (1−a_ij) ln(1−p_ij)]
    Output                : Poincaré ball coordinates

Why the Poincaré ball (and not the hyperboloid)
-----------------------------------------------
Both are exact models of H^{D+1}, and ``geoopt.PoincareBall.dist`` is the exact
hyperbolic distance — no approximation. The choice is numerical: low-degree
leaf nodes belong at large radius (``r ≈ R̂``, which reaches ~16–20 even for
tiny graphs), and on the hyperboloid that means a timelike coordinate
``cosh(r) ≈ 1e5``–``1e7`` that overflows off the manifold during optimisation
(catastrophic for the D=1/Mercator, leaf-heavy regime). The Poincaré ball keeps
coordinates bounded in (−1, 1), so it stays well-conditioned at those radii.

The Fermi-Dirac connection probability on the exact hyperbolic distance
reproduces the S^D model probability (Eq. 1) in the large-radius regime, since
``d_H ≈ r_i + r_j + 2 ln(Δθ_ij/2)`` and that expression makes
``e^{(β/2)(d_H − R̂)} = χ_ij^β`` exactly. β and R̂ are global parameters fixed
from the init; the refinement only moves the per-node positions, so the radial
coordinate (≈ hidden degree κ) is learned jointly with the angle.

References
----------
García-Pérez et al., "Mercator: uncovering faithful hyperbolic embeddings of
complex networks", New Journal of Physics, 2019.
Jankowski, Allard, Boguñá, Serrano, "The D-Mercator method for the
multidimensional hyperbolic embedding of real networks",
Nature Communications, 2023.
"""

from __future__ import annotations

import warnings
from typing import Optional

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import expit

from hypegrl.embedders._dmercator_init import dmercator_init
from hypegrl.embedders.base import HyperbolicEmbedder
from hypegrl.inference.riemannian_optimizer import riemannian_optimize
from hypegrl.manifolds.poincare import POINCARE_BALL
from hypegrl.representations import (
    BallRepresentation,
    HyperboloidRepresentation,
    PolarRepresentation,
    build_representation,
)

# Chart in which the Fermi-Dirac refinement runs. Polar is the default because
# it preserves the radial coordinate at the large radii D-Mercator assigns to
# leaves (the ball saturates at r≈12, the hyperboloid distance fails at r≈18).
_REPRESENTATIONS = {
    "polar": PolarRepresentation,
    "ball": BallRepresentation,
    "hyperboloid": HyperboloidRepresentation,
}

# ---------------------------------------------------------------------------
# Fermi-Dirac NLL on hyperbolic distances
# ---------------------------------------------------------------------------

def _fermi_dirac_nll(
    dist: torch.Tensor,
    A_t: torch.Tensor,
    half_beta: float,
    R_hat: float,
) -> torch.Tensor:
    """
    Binary cross-entropy NLL with the Fermi-Dirac connection probability

        p_ij = 1 / (1 + e^{(β/2)(d_ij − R̂)}) = σ(−z),  z = (β/2)(d_ij − R̂)

    evaluated over the upper triangle. ``logsigmoid`` keeps it stable for
    large ``|z|``.
    """
    N = dist.shape[0]
    z = half_beta * (dist - R_hat)
    mask = torch.triu(
        torch.ones(N, N, dtype=torch.bool, device=dist.device), diagonal=1)
    a = A_t[mask]
    zz = z[mask]
    nll = -(a * F.logsigmoid(-zz) + (1.0 - a) * F.logsigmoid(zz))
    return nll.sum()


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class DMercatorEmbedder(HyperbolicEmbedder):
    """
    D-Mercator graph embedder with hyperbolic (Poincaré ball) refinement.

    The original D-Mercator pipeline provides the warm start (hidden-degree /
    inverse-temperature inference, S^D-corrected Laplacian Eigenmaps, and
    likelihood-maximisation refinement). The embedding is then refined on the
    Poincaré ball by Riemannian Adam, minimising a Fermi-Dirac negative
    log-likelihood on the exact hyperbolic distances — so the radial
    ("popularity") and angular coordinates of every node move jointly.

    Parameters
    ----------
    d:
        Embedding dimension (Poincaré ball B^d). The similarity-space
        dimension is ``D = d − 1`` (sphere S^D). Minimum 2.
    beta:
        Inverse temperature β > D. Controls clustering.
        ``None`` → inferred from the empirical clustering during init.
    lr:
        Learning rate for RiemannianAdam.
    n_steps:
        Number of Riemannian gradient steps. ``0`` returns the pure init.
    grad_clip:
        Max gradient norm for clipping (0 = disabled).
    log_every:
        Print loss every this many steps (0 = silent).
    device:
        Torch device for optimisation.
    random_state:
        Seed for reproducibility.
    d1_init:
        Angular initialisation strategy, only effective when ``d == 2``
        (similarity dimension ``D == 1``):

        - ``"le"`` (default): the paper's Laplacian-Eigenmaps init.
        - ``"mercator"``: the classic Mercator ordering + expected-gap
          re-spacing that the reference C++ uses for D=1. Provided to compare
          the two initialisations; ignored (with a warning) for ``d > 2``.

        On leaf-heavy graphs (e.g. balanced trees) ``"mercator"`` is worth
        trying: it spaces degree-one nodes around their hub deterministically,
        whereas ``"le"`` reinserts each leaf at a random angle (and MLE does not
        move leaves). Measured on ``balanced_tree(2,4)``/``(2,5)`` over 8 seeds,
        ``"mercator"`` gives a ~6–10× larger minimum inter-leaf angle and
        comparable-to-better reconstruction (AUC 0.83 vs 0.77 on ``(2,4)``,
        ~tied on ``(2,5)``).
    """

    def __init__(
        self,
        d: int = 2,
        beta: Optional[float] = None,
        lr: float = 1e-2,
        n_steps: int = 500,
        grad_clip: float = 10.0,
        log_every: int = 50,
        device: str = "cpu",
        random_state: Optional[int] = None,
        d1_init: str = "le",
        representation: str = "polar",
    ):
        if d < 2:
            raise ValueError("d must be >= 2 (sphere dimension D = d-1 >= 1).")
        if d1_init not in ("le", "mercator"):
            raise ValueError(f"d1_init must be 'le' or 'mercator'; got {d1_init!r}.")
        if representation not in _REPRESENTATIONS:
            raise ValueError(
                f"representation must be one of {sorted(_REPRESENTATIONS)}; "
                f"got {representation!r}.")
        self.d = d
        self.beta = beta
        self.lr = lr
        self.n_steps = n_steps
        self.grad_clip = grad_clip
        self.log_every = log_every
        self.device = device
        self.random_state = random_state
        self.d1_init = d1_init
        self.representation = representation

        # Fitted state
        self._X: Optional[np.ndarray] = None           # (N, d) Poincaré ball
        self._r: Optional[np.ndarray] = None           # native radial coords (init)
        self._V: Optional[np.ndarray] = None           # native S^D unit vectors (init)
        self._kappa: Optional[np.ndarray] = None       # hidden degrees κ_i (init)
        self._kappa_min: Optional[float] = None        # κ_min from init
        self._nodes: Optional[list] = None
        self._beta_fitted: Optional[float] = None
        self._mu_fitted: Optional[float] = None
        self._R_sphere: Optional[float] = None
        self._R_hat: Optional[float] = None
        self._loss_history: Optional[list[float]] = None
        self._G: Optional[nx.Graph] = None
        self._rep = None                               # fitted Representation

    # ------------------------------------------------------------------
    # HyperbolicEmbedder interface
    # ------------------------------------------------------------------

    def fit(
        self,
        G: nx.Graph,
        unknown_edges: Optional[list[tuple[int, int]]] = None,
        X_init: Optional[np.ndarray] = None,
    ) -> "DMercatorEmbedder":
        """
        Fit D-Mercator embeddings from a graph.

        Parameters
        ----------
        G:
            Input graph (treated as unweighted; degrees drive κ_i).
        unknown_edges:
            Not yet supported; raises a warning and zero-imputes.
        X_init:
            ``(N, d)`` initial Poincaré ball coordinates for the hyperbolic
            refinement. If given, they are used in place of the original-method
            warm start (the init still runs when the model parameters β, R̂ have
            not yet been inferred). If ``None``, the full original D-Mercator
            pipeline provides the warm start.

        The angular initialisation strategy for ``d == 2`` is controlled by the
        ``d1_init`` constructor argument.
        """
        if unknown_edges:
            warnings.warn(
                "DMercatorEmbedder does not yet support joint optimisation of "
                "unknown edges. Unknown edges are zero-imputed.",
                UserWarning,
                stacklevel=2,
            )

        D = self.d - 1

        # ── Encoder: original D-Mercator init (warm start + model params) ──
        # Always run when X_init is None; when X_init is provided we still need
        # the inferred parameters (β, R̂, μ, R), so run unless already cached.
        need_init = X_init is None or self._R_hat is None or self._nodes is None
        if need_init:
            res = dmercator_init(
                G,
                D=D,
                beta=self.beta,
                random_state=self.random_state,
                d1_init=self.d1_init,
                verbose=self.log_every > 0,
            )
            self._beta_fitted = res["beta"]
            self._mu_fitted = res["mu"]
            self._R_sphere = res["R"]
            self._R_hat = res["R_hat"]
            self._kappa_min = float(np.min(res["kappa"]))
            self._nodes = res["nodes"]
            # Native (S^D × ℝ₊) warm-start coordinates from the init. These seed
            # the representation built below; the authoritative self._r/_V/_kappa
            # are read back from the (possibly refined) representation afterwards.
            self._r = res["r"]
            self._V = res["V"]
        # (cached refit when not need_init: self._r / self._V persist.)

        nodes = self._nodes
        self._G = G

        beta = self._beta_fitted
        R_hat = self._R_hat
        half_beta = beta / 2.0
        A = nx.to_numpy_array(G, nodelist=nodes, weight=None)

        # ── Build the representation (chosen chart) from the warm start ────
        # The native polar coords (self._r, self._V) are the natural warm start;
        # an external X_init (Poincaré-ball coordinates) overrides them via the
        # exact ball→polar map inside the representation.
        rep_cls = _REPRESENTATIONS[self.representation]
        if X_init is not None:
            rep = build_representation(
                rep_cls, X_init, input_chart="ball", device=self.device)
        else:
            rep = rep_cls.from_polar(self._r, self._V, device=self.device)

        # ── Refinement: Riemannian Adam in the chosen chart ────────────────
        # The decoder pulls the pairwise distance from the representation, so the
        # loss is chart-agnostic (same geometry in polar / ball / hyperboloid).
        if self.n_steps == 0:
            self._loss_history = []
        else:
            def loss_fn(rep_, A_t: torch.Tensor) -> torch.Tensor:
                return _fermi_dirac_nll(rep_.dist(), A_t, half_beta, R_hat)

            result = riemannian_optimize(
                representation=rep,
                s_A=A,
                loss_fn=loss_fn,
                lr=self.lr,
                n_steps=self.n_steps,
                grad_clip=self.grad_clip,
                log_every=self.log_every,
                device=self.device,
            )
            self._loss_history = result["loss_history"]

        # ── Read the refined embedding back in the charts the API needs ────
        # Native (authoritative) geometry from the representation; the ball image
        # is a (possibly saturating) projection for embeddings()/plotting. With
        # representation="polar" the radius is preserved at large r; with "ball"
        # it saturates — honestly reflecting what that chart can hold.
        r_ref, V_ref = rep.to_polar()
        self._r = r_ref.detach().cpu().numpy()
        self._V = V_ref.detach().cpu().numpy()
        self._kappa = self._kappa_min * np.exp((D / 2.0) * (R_hat - self._r))
        self._X = rep.to_ball().detach().cpu().numpy()
        self._rep = rep

        return self

    def embeddings(self) -> np.ndarray:
        """
        Return ``(N, d)`` Poincaré ball coordinates.

        Note: the ball chart cannot represent large hyperbolic radii — for
        ``r ≳ 12`` the map ``tanh(r/2)`` saturates to the boundary, so the
        leaf nodes D-Mercator places at ``r`` up to ~40 on larger graphs
        collapse to a common radius here. This is a lossy visualisation /
        interop projection; the authoritative geometry (unsaturated radius,
        κ, angles) is :meth:`native_coordinates`. Rows follow ``nodes()`` order.
        """
        if self._X is None:
            raise RuntimeError("Call fit() before embeddings().")
        return self._X

    def native_coordinates(self) -> dict:
        """
        Return the authoritative native (S^D × ℝ₊) coordinates, stored directly
        from the D-Mercator init (before the lossy Poincaré-ball projection):

        - ``"r"``     : ``(N,)``    hyperbolic radial coordinate (popularity),
                        does NOT saturate at large radius
        - ``"kappa"`` : ``(N,)``    hidden degree κ_i
        - ``"v"``     : ``(N, d)``  unit vectors on S^D (angular / similarity)

        Rows follow ``nodes()`` order. This is the faithful geometry to use for
        radius-based analysis; :meth:`embeddings` (ball) is for plotting/interop.
        """
        if self._r is None:
            raise RuntimeError("Call fit() before native_coordinates().")
        return {"r": self._r, "kappa": self._kappa, "v": self._V}

    def structural_similarity(self, G: nx.Graph) -> np.ndarray:
        """Return the binary adjacency matrix (D-Mercator ignores edge weights)."""
        nodelist = self._nodes if self._nodes is not None else list(G.nodes())
        return nx.to_numpy_array(G, nodelist=nodelist, weight=None)

    def decode(self, X) -> np.ndarray:
        """
        Fermi-Dirac connection probabilities from an embedding:

            p_ij = 1 / (1 + e^{(β/2)(d_H(x_i,x_j) − R̂)})

        Parameters
        ----------
        X:
            ``(N, d)`` Poincaré-ball coordinates, or a fitted
            :class:`~hypegrl.representations.Representation`. Passing the
            representation uses the exact ``rep.dist()`` — important here, since
            D-Mercator places leaves at large radius where the ball saturates.
        """
        if self._R_hat is None or self._beta_fitted is None:
            raise RuntimeError("Call fit() before decode().")

        if hasattr(X, "dist"):
            dist = X.dist().detach().cpu().numpy()
        else:
            Xt = torch.as_tensor(X, dtype=torch.float64)
            dist = POINCARE_BALL.dist(Xt.unsqueeze(1), Xt.unsqueeze(0)).numpy()
        return expit(-(self._beta_fitted / 2.0) * (dist - self._R_hat))

    # ------------------------------------------------------------------
    # Capability flags
    # ------------------------------------------------------------------

    def is_gradient_based(self) -> bool:
        return True

    def is_generative(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def kappa(self) -> Optional[np.ndarray]:
        """Hidden degrees κ_i, from the D-Mercator init (see native_coordinates)."""
        return self._kappa

    @property
    def beta_fitted(self) -> Optional[float]:
        """Inverse temperature β used during fitting."""
        return self._beta_fitted

    @property
    def mu_fitted(self) -> Optional[float]:
        """μ parameter derived from graph statistics (init stage)."""
        return self._mu_fitted

    @property
    def R_sphere(self) -> Optional[float]:
        """Sphere radius R."""
        return self._R_sphere

    @property
    def R_hat(self) -> Optional[float]:
        """Hyperbolic radial offset R̂ (Fermi-Dirac threshold)."""
        return self._R_hat

    @property
    def radial(self) -> Optional[np.ndarray]:
        """
        ``(N,)`` native hyperbolic radial coordinates r_i from the init — not
        read back from the saturating ball embedding. See native_coordinates.
        """
        return self._r

    @property
    def loss_history(self) -> Optional[list[float]]:
        """Fermi-Dirac NLL at each optimisation step."""
        return self._loss_history

    def __repr__(self) -> str:
        beta = self.beta if self.beta is not None else "auto"
        return (
            f"DMercatorEmbedder(d={self.d}, beta={beta}, "
            f"lr={self.lr}, n_steps={self.n_steps})"
        )
