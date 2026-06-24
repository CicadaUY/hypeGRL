"""
D-Mercator graph embedder.

Embeds a graph into the D-dimensional sphere S^D (D = d-1) following the S^D
geometric network model of García-Pérez et al. (2023).  The model assigns
each node a hidden degree κ_i and an angular position v_i ∈ S^D.  Two nodes
connect with probability

    p_ij = 1 / (1 + χ_ij^β),  χ_ij = R·Δθ_ij / (μ·κ_i·κ_j)^{1/D}

where Δθ_ij = arccos(v_i · v_j) is the geodesic angle and R, μ are derived
from graph statistics.

The optimisation follows the encoder-decoder framework:

    Structural similarity : Adjacency matrix A
    Encoder (init)        : S^D model-corrected Laplacian Eigenmaps
    Refinement            : Riemannian Adam on (S^D, round metric)
                            minimising NLL = -Σ_{i<j}[a_ij ln p_ij + (1-a_ij) ln(1-p_ij)]
    Decoder               : Fermi-Dirac connection probabilities p_ij
    Output                : Poincaré ball coords X_i = tanh(r_i/2) · v_i
                            where r_i = R̂ - (2/D) ln(κ_i/κ_min)

References
----------
García-Pérez et al., "Mercator: uncovering faithful hyperbolic embeddings of
complex networks", New Journal of Physics, 2019.
García-Pérez et al., "d-Mercator: multidimensional hyperbolic embedding of real
networks", Nature Communications, 2023.
"""

from __future__ import annotations

import warnings
from typing import Optional

import geoopt
import networkx as nx
import numpy as np
import torch
from scipy.special import gamma as sp_gamma

from hypegrl.embedders.base import HyperbolicEmbedder
from hypegrl.inference.riemannian_optimizer import riemannian_optimize


# ---------------------------------------------------------------------------
# Model parameter helpers
# ---------------------------------------------------------------------------

def _compute_R(N: int, D: int) -> float:
    """Sphere radius: R = [N·Γ((D+1)/2) / (2π^{(D+1)/2})]^{1/D}."""
    num = N * sp_gamma((D + 1) / 2.0)
    den = 2.0 * np.pi ** ((D + 1) / 2.0)
    return (num / den) ** (1.0 / D)


def _compute_mu(D: int, beta: float, avg_deg: float) -> float:
    """μ = β·Γ(D/2)·sin(Dπ/β) / (2π^{1+D/2}·⟨k⟩)."""
    num = beta * sp_gamma(D / 2.0) * np.sin(D * np.pi / beta)
    den = avg_deg * 2.0 * np.pi ** (1.0 + D / 2.0)
    return num / den


def _compute_R_hat(R: float, D: int, mu: float, kappa_min: float) -> float:
    """R̂ = 2·ln(2R / (μ·κ_min²)^{1/D})."""
    return 2.0 * np.log(2.0 * R / (mu * kappa_min ** 2) ** (1.0 / D))


# ---------------------------------------------------------------------------
# Initialization: S^D model-corrected Laplacian Eigenmaps
# ---------------------------------------------------------------------------

def _expected_angular_dist_connected(
    ki_kj: float,
    D: int,
    beta: float,
    mu: float,
    R: float,
    n_pts: int = 100,
) -> float:
    """
    E[Δθ|a_ij=1] via numerical integration of the S^D conditional distribution.

    For connected nodes i,j with product of hidden degrees κ_i·κ_j, the
    expected angular distance given connection is

        E[Δθ|a=1] = ∫ θ·sin^{D-1}(θ)·p(θ) dθ / ∫ sin^{D-1}(θ)·p(θ) dθ

    where p(θ) = 1/(1+(Rθ/(μκ_iκ_j)^{1/D})^β).
    """
    theta = np.linspace(1e-7, np.pi - 1e-7, n_pts)
    c = max((mu * ki_kj) ** (1.0 / D), 1e-10)
    chi = R * theta / c
    sin_d = np.sin(theta) ** max(D - 1, 0) if D > 1 else np.ones(n_pts)
    fermi = 1.0 / (1.0 + chi ** beta)
    w = sin_d * fermi
    den = np.trapz(w, theta)
    if den < 1e-10:
        return np.pi / 2.0
    return float(np.trapz(theta * w, theta) / den)


def _laplacian_eigenmaps_init(
    G: nx.Graph,
    nodes: list,
    kappa: np.ndarray,
    D: int,
    R: float,
    mu: float,
    beta: float,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    S^D model-corrected Laplacian Eigenmaps initialization.

    Builds a heat-kernel weight matrix using expected angular distances from
    the S^D model, then returns the D+1 smallest non-trivial eigenvectors
    row-normalised to S^{D} ⊂ R^{D+1}.

    Returns
    -------
    (N, D+1) array of unit-norm row vectors.
    """
    N = len(nodes)
    node_idx = {n: i for i, n in enumerate(nodes)}

    # --- Model-corrected heat-kernel weights ---------------------------------
    # First pass: compute expected angular distances for all edges
    raw_dists: list[float] = []
    edge_indices: list[tuple[int, int]] = []
    for u, v in G.edges():
        i, j = node_idx[u], node_idx[v]
        theta_exp = _expected_angular_dist_connected(
            kappa[i] * kappa[j], D, beta, mu, R
        )
        raw_dists.append(theta_exp)
        edge_indices.append((i, j))

    W = np.zeros((N, N))
    if raw_dists:
        t = max(float(np.mean(np.array(raw_dists) ** 2)), 1e-10)
        for (i, j), d in zip(edge_indices, raw_dists):
            w = float(np.exp(-(d ** 2) / t))
            W[i, j] = w
            W[j, i] = w

    # --- Normalised Laplacian eigenmaps --------------------------------------
    deg = W.sum(axis=1)
    inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = np.diag(inv_sqrt)
    L_sym = np.eye(N) - D_inv_sqrt @ W @ D_inv_sqrt
    L_sym = (L_sym + L_sym.T) / 2.0  # symmetrise numerically

    try:
        eigvals, eigvecs = np.linalg.eigh(L_sym)
        # Skip index 0 (constant vector), take next D+1
        idx = np.argsort(eigvals)
        selected = idx[1 : D + 2]
        V = eigvecs[:, selected]  # (N, D+1)
    except np.linalg.LinAlgError:
        rng = np.random.default_rng(random_state)
        V = rng.standard_normal((N, D + 1))

    # Row-normalise to unit sphere
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms = np.where(norms > 1e-10, norms, 1.0)
    return V / norms


# ---------------------------------------------------------------------------
# Differentiable NLL loss
# ---------------------------------------------------------------------------

def _sd_nll_loss(
    V: torch.Tensor,
    A_t: torch.Tensor,
    kappa_t: torch.Tensor,
    beta: float,
    mu: float,
    R: float,
    D: int,
) -> torch.Tensor:
    """
    Binary cross-entropy NLL for the S^D geometric network model.

    L = -Σ_{i<j} [ a_ij ln p_ij + (1-a_ij) ln(1-p_ij) ]
    p_ij = 1 / (1 + (R·θ_ij / (μ·κ_i·κ_j)^{1/D})^β)
    """
    eps = 1e-7
    N = V.shape[0]

    # Angular distances
    cos_sim = torch.clamp(V @ V.t(), -1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_sim)  # (N, N)

    # Connection probabilities
    kk = kappa_t.unsqueeze(0) * kappa_t.unsqueeze(1)  # (N, N)
    denom = torch.pow(torch.clamp(mu * kk, min=eps), 1.0 / D)
    chi = R * theta / denom
    p = 1.0 / (1.0 + torch.pow(torch.clamp(chi, min=eps), beta))
    p = torch.clamp(p, eps, 1.0 - eps)

    # Sum over upper triangle only
    mask = torch.triu(torch.ones(N, N, dtype=torch.bool, device=V.device), diagonal=1)
    a = A_t[mask]
    nll = -(a * torch.log(p[mask]) + (1.0 - a) * torch.log(1.0 - p[mask]))
    return nll.sum()


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class DMercatorEmbedder(HyperbolicEmbedder):
    """
    D-Mercator graph embedder: S^D geometric network model.

    Nodes are placed on the D-dimensional unit sphere S^D ⊂ R^{d} (D = d-1)
    with hidden degrees κ_i initialised from observed degrees.  Angular
    positions are initialised with model-corrected Laplacian Eigenmaps and
    refined by Riemannian Adam on S^D minimising the S^D model NLL.

    Output embeddings are in the Poincaré ball (for compatibility with the
    rest of the framework): X_i = tanh(r_i/2) · v_i where r_i is the
    hyperbolic radial coordinate derived from κ_i.

    Parameters
    ----------
    d:
        Ambient dimension of the embedding sphere (sphere dimension D = d-1).
        Minimum 2 (corresponding to S^1, a circle).
    beta:
        Inverse temperature β > D.  Controls clustering.
        ``None`` → auto-set to 2·D at fit time.
    lr:
        Learning rate for RiemannianAdam.
    n_steps:
        Number of Riemannian gradient steps.
    grad_clip:
        Max gradient norm for clipping (0 = disabled).
    log_every:
        Print loss every this many steps (0 = silent).
    device:
        Torch device for optimisation.
    random_state:
        Seed for reproducibility.
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
    ):
        if d < 2:
            raise ValueError("d must be >= 2 (sphere dimension D = d-1 >= 1).")
        self.d = d
        self.beta = beta
        self.lr = lr
        self.n_steps = n_steps
        self.grad_clip = grad_clip
        self.log_every = log_every
        self.device = device
        self.random_state = random_state

        # Fitted state
        self._X: Optional[np.ndarray] = None           # (N, d) Poincaré ball
        self._V: Optional[np.ndarray] = None           # (N, d) unit sphere
        self._kappa: Optional[np.ndarray] = None
        self._nodes: Optional[list] = None
        self._beta_fitted: Optional[float] = None
        self._mu_fitted: Optional[float] = None
        self._R_sphere: Optional[float] = None
        self._R_hat: Optional[float] = None
        self._loss_history: Optional[list[float]] = None
        self._G: Optional[nx.Graph] = None

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
            Input graph (unweighted or weighted; degrees used for κ_i).
        unknown_edges:
            Not yet supported; raises a warning and zero-imputes.
        X_init:
            ``(N, d)`` initial coordinates.  If given, rows are
            normalised to the unit sphere and used in place of the
            Laplacian-Eigenmaps initialisation.  If ``None``, model-
            corrected Laplacian Eigenmaps provide the warm start.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)

        if unknown_edges:
            warnings.warn(
                "DMercatorEmbedder does not yet support joint optimisation of "
                "unknown edges. Unknown edges are zero-imputed.",
                UserWarning,
                stacklevel=2,
            )

        N = G.number_of_nodes()
        D = self.d - 1
        nodes = list(G.nodes())
        node_idx = {n: i for i, n in enumerate(nodes)}

        # ── 1. κ initialisation (observed degrees, clamped to ≥ 1) ───────
        degrees = np.array([G.degree(n) for n in nodes], dtype=np.float64)
        kappa = np.maximum(degrees, 1.0)

        # ── 2. Derive model parameters from graph statistics ──────────────
        avg_deg = float(np.mean(degrees))
        if avg_deg == 0.0:
            avg_deg = 1.0  # isolated nodes: treat as ⟨k⟩ = 1

        beta = self.beta if self.beta is not None else float(2 * D)
        if beta <= D:
            beta = D + 0.5
            warnings.warn(
                f"beta must be > D={D}; clamped to {beta}.",
                UserWarning,
                stacklevel=2,
            )

        R = _compute_R(N, D)
        mu = _compute_mu(D, beta, avg_deg)
        kappa_min = float(kappa.min())
        R_hat = _compute_R_hat(R, D, mu, kappa_min)

        self._beta_fitted = beta
        self._mu_fitted = mu
        self._R_sphere = R
        self._R_hat = R_hat
        self._kappa = kappa
        self._nodes = nodes
        self._G = G

        # ── 3. Angular initialisation ──────────────────────────────────────
        if X_init is not None:
            norms = np.linalg.norm(X_init, axis=1, keepdims=True)
            norms = np.where(norms > 1e-10, norms, 1.0)
            V_init = (X_init / norms).astype(np.float64)
        else:
            V_init = _laplacian_eigenmaps_init(
                G, nodes, kappa, D, R, mu, beta,
                random_state=self.random_state,
            )

        # ── 4. Structural similarity (binary adjacency matrix) ───────────
        # D-Mercator is a binary model; ignore edge weights.
        A = nx.to_numpy_array(G, nodelist=nodes, weight=None)

        # ── 5. Riemannian optimisation on S^{d-1} ─────────────────────────
        sphere = geoopt.Sphere()
        kappa_t = torch.tensor(kappa, dtype=torch.float64)

        def loss_fn(V: torch.Tensor, A_t: torch.Tensor) -> torch.Tensor:
            return _sd_nll_loss(V, A_t, kappa_t.to(V.device), beta, mu, R, D)

        result = riemannian_optimize(
            X_init=V_init,
            s_A=A,
            loss_fn=loss_fn,
            manifold=sphere,
            lr=self.lr,
            n_steps=self.n_steps,
            grad_clip=self.grad_clip,
            log_every=self.log_every,
            device=self.device,
        )

        V_opt = result["X"]   # (N, d) unit vectors
        self._V = V_opt
        self._loss_history = result["loss_history"]

        # ── 6. Convert to Poincaré ball ───────────────────────────────────
        # r_i = R̂ - (2/D)·ln(κ_i/κ_min) is the hyperbolic radial coord
        # X_i = tanh(r_i/2) · v_i
        r_hyp = R_hat - (2.0 / D) * np.log(kappa / kappa_min)
        poincare_r = np.clip(np.tanh(r_hyp / 2.0), 0.0, 1.0 - 1e-5)
        self._X = poincare_r[:, None] * V_opt

        return self

    def embeddings(self) -> np.ndarray:
        """Return ``(N, d)`` Poincaré ball coordinates."""
        if self._X is None:
            raise RuntimeError("Call fit() before embeddings().")
        return self._X

    def structural_similarity(self, G: nx.Graph) -> np.ndarray:
        """Return the binary adjacency matrix (D-Mercator ignores edge weights)."""
        nodelist = self._nodes if self._nodes is not None else list(G.nodes())
        return nx.to_numpy_array(G, nodelist=nodelist, weight=None)

    def decode(self, X: np.ndarray) -> np.ndarray:
        """
        Compute S^D connection probabilities from embeddings.

        Each row of ``X`` is normalised to a unit vector before computing
        angular distances.  This means ``decode`` works identically for
        both Poincaré ball output (from ``embeddings()``) and raw unit
        sphere coordinates.
        """
        if self._kappa is None or self._beta_fitted is None:
            raise RuntimeError("Call fit() before decode().")

        D = self.d - 1
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        V = X / np.where(norms > 1e-10, norms, 1.0)

        cos_sim = np.clip(V @ V.T, -1.0 + 1e-7, 1.0 - 1e-7)
        theta = np.arccos(cos_sim)

        kk = np.outer(self._kappa, self._kappa)
        denom = np.maximum(self._mu_fitted * kk, 1e-10) ** (1.0 / D)
        chi = self._R_sphere * theta / denom
        return 1.0 / (1.0 + np.maximum(chi, 1e-10) ** self._beta_fitted)

    def distance(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        NLL of the S^D model.

        ``X`` may be unit sphere coordinates or Poincaré ball coordinates;
        rows are normalised before computing angular distances.
        """
        if self._kappa is None or self._beta_fitted is None:
            raise RuntimeError("Call fit() before distance().")

        D = self.d - 1
        norms = torch.norm(X, dim=-1, keepdim=True).clamp(min=1e-10)
        V = X / norms

        kappa_t = torch.tensor(self._kappa, dtype=X.dtype, device=X.device)
        return _sd_nll_loss(
            V, A, kappa_t,
            self._beta_fitted, self._mu_fitted, self._R_sphere, D,
        )

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
        """Hidden degree sequence κ_i (initialised from observed degrees)."""
        return self._kappa

    @property
    def beta_fitted(self) -> Optional[float]:
        """Inverse temperature β used during fitting."""
        return self._beta_fitted

    @property
    def mu_fitted(self) -> Optional[float]:
        """μ parameter derived from graph statistics."""
        return self._mu_fitted

    @property
    def R_sphere(self) -> Optional[float]:
        """Sphere radius R."""
        return self._R_sphere

    @property
    def angular_positions(self) -> Optional[np.ndarray]:
        """``(N, d)`` optimised unit vectors on S^{d-1}."""
        return self._V

    @property
    def nodes(self) -> Optional[list]:
        """Node ordering used during fitting (matches row order of embeddings)."""
        return self._nodes

    @property
    def loss_history(self) -> Optional[list[float]]:
        """NLL at each optimisation step."""
        return self._loss_history

    def __repr__(self) -> str:
        beta = self.beta if self.beta is not None else "auto"
        return (
            f"DMercatorEmbedder(d={self.d}, beta={beta}, "
            f"lr={self.lr}, n_steps={self.n_steps})"
        )
