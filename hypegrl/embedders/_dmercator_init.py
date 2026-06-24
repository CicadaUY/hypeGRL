"""
D-Mercator initialization: faithful reimplementation of the original method.

Reconstructs the embedding pipeline of:
    Jankowski, Allard, Boguñá, Serrano, "The D-Mercator method for the
    multidimensional hyperbolic embedding of real networks",
    Nature Communications, 2023.

The embedding lives in the **S^D model**: nodes sit on the D-sphere
(radius R in R^{D+1}, here stored as unit vectors since R is a global
scale that cancels under the angular distance) plus a hidden-degree
"popularity" coordinate κ_i. Connection probability:

    Δθ_ij = arccos(v_i · v_j)                      (unit vectors)
    χ_ij  = R · Δθ_ij / (μ κ_i κ_j)^{1/D}          (Eq. 1)
    p_ij  = 1 / (1 + χ_ij^β)                        (Eqs. 1, 15)

Pipeline (``dmercator_init``):

    Stage 1  INFER_KAPPA_AND_BETA   — couple hidden degrees κ and inverse
                                      temperature β; β chosen so the model
                                      clustering matches the empirical one.
    Stage 2  MODEL_CORRECTED_LE     — S^D-corrected Laplacian Eigenmaps for
                                      an initial angular guess.
    Stage 3  MLE_REFINE             — likelihood maximization, candidate
                                      proposals around hidden-degree-weighted
                                      neighbor means.
    Stage 4  FINAL_ADJUST_KAPPA     — readjust κ given the actual positions.
    (opt.)   RADIAL_MAP             — map κ to hyperbolic radial coords (Eq. 7).

Degree-one nodes are dropped before Stages 2–3 and reinserted afterward.

Equation numbers refer to the paper. Underspecified bits follow the official
implementation at github.com/networkgeometry/d-mercator (see comments). The full
step-by-step specification this module tracks — with the §-numbers cited in the
comments below — lives in ``docs/methods/dmercator_pseudocode.md``; keep the two
in sync.
"""

from __future__ import annotations

import warnings
from typing import Optional

import networkx as nx
import numpy as np
from scipy.special import gamma as sp_gamma

# ---------------------------------------------------------------------------
# Constants (mirroring the reference implementation)
# ---------------------------------------------------------------------------

_KAPPA_TOL_CLASS = 0.01    # ε for degree-class κ inference (Stage 1)
_KAPPA_TOL_FINAL = 0.5     # NUMERICAL_CONVERGENCE_THRESHOLD_3 (Stage 4)
_KAPPA_MAX_ITER = 500      # KAPPA_MAX_NB_ITER_CONV
_CLUSTERING_TOL = 0.01     # ε_c̄ on |c̄ - c̄_emp|
_CLUSTERING_MC = 600       # MC samples per degree class
_BETA_MAX_BISECT = 40      # cap on bisection iterations
_N_INT = 200               # trapezoid resolution over θ ∈ [0, π]
_TINY = 1e-12


# ---------------------------------------------------------------------------
# Model primitives (§0)
# ---------------------------------------------------------------------------

def compute_R(N: int, D: int) -> float:
    """Sphere radius (Eq. 2): R = [N·Γ((D+1)/2) / (2π^{(D+1)/2})]^{1/D}."""
    num = N * sp_gamma((D + 1) / 2.0)
    den = 2.0 * np.pi ** ((D + 1) / 2.0)
    return (num / den) ** (1.0 / D)


def compute_mu(beta: float, D: int, avg_deg: float) -> float:
    """μ (Eq. 3): β·Γ(D/2)·sin(Dπ/β) / (2π^{1+D/2}·⟨k⟩). Recompute when β changes."""
    num = beta * sp_gamma(D / 2.0) * np.sin(D * np.pi / beta)
    den = avg_deg * 2.0 * np.pi ** (1.0 + D / 2.0)
    return num / den


def _C_D(D: int) -> float:
    """Angular-measure normalizer C_D = Γ((D+1)/2) / (Γ(D/2)·√π) (Eq. 16)."""
    return sp_gamma((D + 1) / 2.0) / (sp_gamma(D / 2.0) * np.sqrt(np.pi))


def _theta_grid(n: int = _N_INT) -> np.ndarray:
    """Integration grid over θ ∈ (0, π)."""
    return np.linspace(1e-7, np.pi - 1e-7, n)


def _sin_pow(theta: np.ndarray, D: int) -> np.ndarray:
    """sin^{D-1}(θ), with the D=1 case (constant 1, uniform measure)."""
    return np.sin(theta) ** (D - 1) if D > 1 else np.ones_like(theta)


def _p_connect(
    dtheta: np.ndarray,
    ka: np.ndarray,
    kb: np.ndarray,
    beta: float,
    mu: float,
    R: float,
    D: int,
) -> np.ndarray:
    """Connection probability p = 1/(1+χ^β), elementwise (Eq. 1)."""
    c = np.maximum(mu * ka * kb, _TINY) ** (1.0 / D)
    chi = R * dtheta / c
    return 1.0 / (1.0 + np.maximum(chi, _TINY) ** beta)


def _random_unit_vectors(D: int, size: int, rng: np.random.Generator) -> np.ndarray:
    """``size`` uniform unit vectors on S^{D-1} ⊂ R^D (azimuth directions)."""
    g = rng.standard_normal((size, D))
    n = np.linalg.norm(g, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    return g / n


# ---------------------------------------------------------------------------
# Stage 1 — hidden degrees κ and inverse temperature β (§2)
# ---------------------------------------------------------------------------

def _connection_prob_matrix(
    kappa_classes: np.ndarray,
    beta: float,
    mu: float,
    R: float,
    D: int,
    theta: np.ndarray,
) -> np.ndarray:
    """
    Marginal connection probability between degree classes (Eq. 13 kernel):

        P_link(κ_a, κ_b) = C_D ∫_0^π sin^{D-1}θ / (1+(Rθ/(μκ_aκ_b)^{1/D})^β) dθ

    Returns a ``(C, C)`` matrix where ``C`` is the number of degree classes.
    """
    KK = np.outer(kappa_classes, kappa_classes)            # (C, C)
    c = np.maximum(mu * KK, _TINY) ** (1.0 / D)            # (C, C)
    sinp = _sin_pow(theta, D)                              # (T,)
    chi = R * theta[None, None, :] / c[:, :, None]         # (C, C, T)
    integrand = sinp[None, None, :] / (1.0 + chi ** beta)  # (C, C, T)
    return _C_D(D) * np.trapz(integrand, theta, axis=-1)   # (C, C)


def _expected_degrees_by_class(
    kappa_classes: np.ndarray,
    counts: np.ndarray,
    beta: float,
    mu: float,
    R: float,
    D: int,
    theta: np.ndarray,
) -> np.ndarray:
    """
    Expected degree of each class (Eq. 13): sum the marginal connection
    probability over all other nodes, excluding self.
    """
    Pmat = _connection_prob_matrix(kappa_classes, beta, mu, R, D, theta)
    # k̄(a) = Σ_b P[a,b]·counts[b] − P[a,a]   (drop the self term)
    return Pmat @ counts - np.diag(Pmat)


def _infer_kappa(
    degree_values: np.ndarray,
    counts: np.ndarray,
    beta: float,
    mu: float,
    R: float,
    D: int,
    theta: np.ndarray,
    rng: np.random.Generator,
    tol: float = _KAPPA_TOL_CLASS,
    max_iter: int = _KAPPA_MAX_ITER,
) -> np.ndarray:
    """
    Adjust κ per degree class so the expected degree matches the observed one
    (§2.2). ``κ_i ← |κ_i + (k_i − k̄_i)·u|`` with random step ``u ~ U(0,1)``.
    """
    kappa = degree_values.astype(np.float64).copy()
    for _ in range(max_iter):
        kbar = _expected_degrees_by_class(kappa, counts, beta, mu, R, D, theta)
        if np.max(np.abs(kbar - degree_values)) <= tol:
            break
        u = rng.random(len(kappa))
        kappa = np.abs(kappa + (degree_values - kbar) * u)
        kappa = np.maximum(kappa, _TINY)
    return kappa


def _sample_angle_given_connected(
    kappa_a: float,
    kappa_b: np.ndarray,
    beta: float,
    mu: float,
    R: float,
    D: int,
    rng: np.random.Generator,
    theta: np.ndarray,
) -> np.ndarray:
    """
    Sample Δθ ~ ρ(Δθ | connected; κ_a, κ_b) (Eq. 14) by inverse-CDF.

    ``kappa_b`` is an array (one target per sample); samples are drawn by
    grouping over its distinct values for efficiency.
    """
    sinp = _sin_pow(theta, D)
    out = np.empty(len(kappa_b))
    for kb in np.unique(kappa_b):
        c = max((mu * kappa_a * kb) ** (1.0 / D), _TINY)
        chi = R * theta / c
        dens = sinp / (1.0 + chi ** beta)           # ∝ p_connect · ρ
        cdf = np.concatenate(
            [[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(theta))]
        )
        if cdf[-1] <= _TINY:
            cdf = np.linspace(0.0, 1.0, len(theta))
        else:
            cdf /= cdf[-1]
        mask = kappa_b == kb
        out[mask] = np.interp(rng.random(int(mask.sum())), cdf, theta)
    return out


def _expected_clustering(
    degree_values: np.ndarray,
    kappa_classes: np.ndarray,
    counts: np.ndarray,
    neighbor_p: np.ndarray,
    beta: float,
    mu: float,
    R: float,
    D: int,
    rng: np.random.Generator,
    theta: np.ndarray,
    m: int = _CLUSTERING_MC,
) -> float:
    """
    Expected mean local clustering by Monte-Carlo (§2.4).

    For each degree class, repeatedly draw two neighbor degree classes from
    ``P(k'|k) = k'P(k')/⟨k⟩``, sample the two angular separations to the
    central node, place all three on the sphere (uniform azimuth on S^{D-1}),
    and average the connection probability between the two neighbors.
    """
    N = counts.sum()
    deg_to_kappa = dict(zip(degree_values, kappa_classes))
    total = 0.0
    for idx in range(len(degree_values)):
        kappa_c = kappa_classes[idx]
        # neighbor degree classes (uncorrelated): P(k'|k) = k'P(k')/⟨k⟩
        d1 = rng.choice(degree_values, size=m, p=neighbor_p)
        d2 = rng.choice(degree_values, size=m, p=neighbor_p)
        kappa1 = np.array([deg_to_kappa[d] for d in d1])
        kappa2 = np.array([deg_to_kappa[d] for d in d2])

        dth1 = _sample_angle_given_connected(
            kappa_c, kappa1, beta, mu, R, D, rng, theta)
        dth2 = _sample_angle_given_connected(
            kappa_c, kappa2, beta, mu, R, D, rng, theta)

        # place: central node at the pole; neighbors at polar angle dth with
        # uniform azimuth on S^{D-1}. cos(Δθ12) = cosθ1 cosθ2 + sinθ1 sinθ2 (w1·w2)
        w1 = _random_unit_vectors(D, m, rng)
        w2 = _random_unit_vectors(D, m, rng)
        cos12 = (
            np.cos(dth1) * np.cos(dth2)
            + np.sin(dth1) * np.sin(dth2) * np.sum(w1 * w2, axis=1)
        )
        dth12 = np.arccos(np.clip(cos12, -1.0, 1.0))

        p12 = _p_connect(dth12, kappa1, kappa2, beta, mu, R, D)
        total += float(np.mean(p12)) * counts[idx] / N
    return total


def infer_kappa_and_beta(
    degree_values: np.ndarray,
    counts: np.ndarray,
    avg_deg: float,
    c_emp: float,
    R: float,
    D: int,
    rng: np.random.Generator,
    theta: np.ndarray,
    beta_fixed: Optional[float] = None,
    verbose: bool = False,
) -> tuple[np.ndarray, float]:
    """
    Jointly infer hidden degrees κ (per degree class) and inverse temperature β
    (§2.1). β is found by bracket-then-bisect so the model clustering matches
    the empirical clustering; the κ inner loop is re-run for every β.

    If ``beta_fixed`` is given, β inference is skipped and only κ is inferred
    at that β.
    """
    if beta_fixed is not None:
        beta = float(beta_fixed)
        mu = compute_mu(beta, D, avg_deg)
        kappa = _infer_kappa(degree_values, counts, beta, mu, R, D, theta, rng)
        return kappa, beta

    # neighbor degree sampling distribution P(k'|k) = k'P(k')/⟨k⟩
    neighbor_p = degree_values * counts
    neighbor_p = neighbor_p / neighbor_p.sum()

    # initial guess β ∈ (D, D+1); quality is independent of it
    beta = float(rng.uniform(D, D + 1))
    mu = compute_mu(beta, D, avg_deg)
    kappa = _infer_kappa(degree_values, counts, beta, mu, R, D, theta, rng)
    cbar = _expected_clustering(
        degree_values, kappa, counts, neighbor_p, beta, mu, R, D, rng, theta
    )

    # bracket: grow β by ×1.5 until model clustering exceeds the empirical
    beta_lo = beta
    n_grow = 0
    while cbar < c_emp and n_grow < _BETA_MAX_BISECT:
        beta_lo = beta
        beta = 1.5 * beta
        mu = compute_mu(beta, D, avg_deg)
        kappa = _infer_kappa(degree_values, counts, beta, mu, R, D, theta, rng)
        cbar = _expected_clustering(
            degree_values, kappa, counts, neighbor_p, beta, mu, R, D, rng, theta
        )
        n_grow += 1
        if verbose:
            print(f"  [β bracket] β={beta:.4f}  c̄={cbar:.4f}  (target {c_emp:.4f})")
    beta_hi = beta

    # bisection on β ∈ [β_lo, β_hi]
    for _ in range(_BETA_MAX_BISECT):
        if abs(cbar - c_emp) < _CLUSTERING_TOL:
            break
        beta = 0.5 * (beta_lo + beta_hi)
        mu = compute_mu(beta, D, avg_deg)
        kappa = _infer_kappa(degree_values, counts, beta, mu, R, D, theta, rng)
        cbar = _expected_clustering(
            degree_values, kappa, counts, neighbor_p, beta, mu, R, D, rng, theta
        )
        if cbar < c_emp:
            beta_lo = beta
        else:
            beta_hi = beta
        if verbose:
            print(f"  [β bisect]  β={beta:.4f}  c̄={cbar:.4f}  (target {c_emp:.4f})")

    return kappa, beta


# ---------------------------------------------------------------------------
# Stage 2 — S^D-corrected Laplacian Eigenmaps (§3)
# ---------------------------------------------------------------------------

def _expected_angular_distance(
    kappa_prod: np.ndarray,
    beta: float,
    mu: float,
    R: float,
    D: int,
    theta: np.ndarray,
) -> np.ndarray:
    """
    Expected angular distance for connected pairs (Eqs. 19–20):

        ⟨Δθ⟩ = ∫ θ sin^{D-1}θ p(θ) dθ / ∫ sin^{D-1}θ p(θ) dθ

    Vectorised over an array of κ_i·κ_j products (one per edge).
    """
    sinp = _sin_pow(theta, D)                                   # (T,)
    c = np.maximum(mu * kappa_prod, _TINY) ** (1.0 / D)         # (E,)
    chi = R * theta[None, :] / c[:, None]                       # (E, T)
    w = sinp[None, :] / (1.0 + chi ** beta)                     # (E, T)
    num = np.trapz(theta[None, :] * w, theta, axis=1)
    den = np.trapz(w, theta, axis=1)
    den = np.where(den > _TINY, den, 1.0)
    return num / den


def model_corrected_le(
    A: np.ndarray,
    kappa: np.ndarray,
    beta: float,
    mu: float,
    R: float,
    D: int,
    theta: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    S^D-corrected Laplacian Eigenmaps (§3). Returns ``(n, D+1)`` unit vectors.

    Edge weights come from the model's expected angular distances mapped to
    unit-sphere chords ``d_ij = 2 sin(⟨Δθ⟩/2)`` (Eq. 5), with heat-kernel
    scale ``t = ⟨d_ij²⟩`` (note e). The random-walk-normalized Laplacian is
    solved via its symmetric form ``I − D_s^{-1/2} Ω D_s^{-1/2}`` (note f); the
    per-node D_s^{-1/2} scaling cancels under the final row normalization, so
    the symmetric eigenvectors are used directly.
    """
    n = A.shape[0]
    iu, ju = np.where(np.triu(A, k=1) > 0)
    if len(iu) == 0:
        return _random_unit_vectors(D + 1, n, rng)

    kk = kappa[iu] * kappa[ju]
    exp_dtheta = _expected_angular_distance(kk, beta, mu, R, D, theta)
    d_chord = 2.0 * np.sin(exp_dtheta / 2.0)        # unit-sphere chord (Eq. 5)

    # note (e): heat-kernel scale is the mean of squared chords
    t = float(np.mean(d_chord ** 2))
    t = max(t, _TINY)
    w = np.exp(-(d_chord ** 2) / t)                            # Eq. 5

    W = np.zeros((n, n))
    W[iu, ju] = w
    W[ju, iu] = w

    s = W.sum(axis=1)                                          # row strengths
    inv_sqrt = np.where(s > _TINY, 1.0 / np.sqrt(s), 0.0)
    L_sym = np.eye(n) - (inv_sqrt[:, None] * W * inv_sqrt[None, :])
    L_sym = 0.5 * (L_sym + L_sym.T)                            # symmetrise numerically

    try:
        _, eigvecs = np.linalg.eigh(L_sym)
        # request D+2 smallest, drop the trivial λ≈0 constant vector
        U = eigvecs[:, 1 : D + 2]                              # (n, D+1)
    except np.linalg.LinAlgError:
        U = rng.standard_normal((n, D + 1))

    norms = np.linalg.norm(U, axis=1, keepdims=True)
    norms = np.where(norms > _TINY, norms, 1.0)
    return U / norms                                          # unit vectors


# ---------------------------------------------------------------------------
# Stage 3 — likelihood maximization (§4)
# ---------------------------------------------------------------------------

def _onion_order(G_core: nx.Graph, rng: np.random.Generator) -> list[int]:
    """Onion-decomposition node order, outer→inner, random within a layer."""
    try:
        layers = nx.onion_layers(G_core)
    except Exception:
        layers = dict(G_core.degree())
    by_layer: dict[int, list[int]] = {}
    for node, layer in layers.items():
        by_layer.setdefault(layer, []).append(node)
    order: list[int] = []
    for layer in sorted(by_layer):
        nodes = by_layer[layer]
        rng.shuffle(nodes)
        order.extend(nodes)
    return order


def _local_loglik_candidates(
    candidates: np.ndarray,   # (m, D+1) unit
    V: np.ndarray,            # (n, D+1) unit
    a_row: np.ndarray,        # (n,) adjacency row for node i
    kappa_i: float,
    kappa: np.ndarray,        # (n,)
    i: int,
    beta: float,
    mu: float,
    R: float,
    D: int,
) -> np.ndarray:
    """Local log-likelihood (Eq. 23) for each candidate position of node i."""
    cos = np.clip(candidates @ V.T, -1.0 + 1e-12, 1.0 - 1e-12)   # (m, n)
    dtheta = np.arccos(cos)
    c = np.maximum(mu * kappa_i * kappa[None, :], _TINY) ** (1.0 / D)
    chi = R * dtheta / c
    p = 1.0 / (1.0 + np.maximum(chi, _TINY) ** beta)
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    ll = a_row[None, :] * np.log(p) + (1.0 - a_row[None, :]) * np.log(1.0 - p)
    ll[:, i] = 0.0                                               # skip self term
    return ll.sum(axis=1)


def mle_refine(
    A: np.ndarray,
    kappa: np.ndarray,
    beta: float,
    mu: float,
    R: float,
    D: int,
    V: np.ndarray,
    rng: np.random.Generator,
    max_sweeps: int = 10,
    tol: float = 1e-3,
) -> np.ndarray:
    """
    Refine angular positions by local likelihood maximization (§4).

    Visits nodes in onion order; for each, proposes candidates around the
    hidden-degree-weighted neighbor mean and keeps the most likely (the
    incumbent position is the baseline, so a node may stay put).
    """
    n = A.shape[0]
    G_core = nx.from_numpy_array(A)
    order = _onion_order(G_core, rng)
    neighbors = [np.where(A[i] > 0)[0] for i in range(n)]
    n_cand = int(round(100 * max(np.log(max(n, 2)), 1.0)))

    prev_total = -np.inf
    for _ in range(max_sweeps):
        for i in order:
            nbr = neighbors[i]
            if len(nbr) == 0:
                continue

            # (a) hidden-degree-weighted mean of neighbor positions (Eq. 21)
            wmean = np.sum(V[nbr] / (kappa[nbr] ** 2)[:, None], axis=0)
            nrm = np.linalg.norm(wmean)
            vbar = wmean / nrm if nrm > _TINY else V[i].copy()

            # (b) proposal spread (Eq. 22): σ = max(π, Δθ_max)/2 = max(π/2, Δθ_max/2)
            cos_nbr = np.clip(vbar @ V[nbr].T, -1.0, 1.0)
            dtheta_max = float(np.max(np.arccos(cos_nbr)))
            sigma = max(np.pi, dtheta_max) / 2.0

            # multivariate normal on the unit sphere: add N(0,σ²), renormalize
            noise = sigma * rng.standard_normal((n_cand, D + 1))
            cand = vbar[None, :] + noise
            cnorm = np.linalg.norm(cand, axis=1, keepdims=True)
            cnorm = np.where(cnorm > _TINY, cnorm, 1.0)
            cand = cand / cnorm
            cand = np.vstack([cand, V[i][None, :]])             # incumbent baseline

            ll = _local_loglik_candidates(
                cand, V, A[i], kappa[i], kappa, i, beta, mu, R, D
            )
            V[i] = cand[int(np.argmax(ll))]

        # convergence on total local log-likelihood
        cos = np.clip(V @ V.T, -1.0 + 1e-12, 1.0 - 1e-12)
        dtheta = np.arccos(cos)
        c = np.maximum(mu * np.outer(kappa, kappa), _TINY) ** (1.0 / D)
        p = np.clip(1.0 / (1.0 + np.maximum(R * dtheta / c, _TINY) ** beta),
                    1e-12, 1.0 - 1e-12)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        total = float(np.sum(A[mask] * np.log(p[mask])
                             + (1.0 - A[mask]) * np.log(1.0 - p[mask])))
        if abs(total - prev_total) <= tol * max(abs(prev_total), 1.0):
            break
        prev_total = total

    return V


# ---------------------------------------------------------------------------
# Stage 4 — final hidden-degree readjustment (§5)
# ---------------------------------------------------------------------------

def final_adjust_kappa(
    A: np.ndarray,
    kappa: np.ndarray,
    beta: float,
    mu: float,
    R: float,
    D: int,
    V: np.ndarray,
    rng: np.random.Generator,
    tol: float = _KAPPA_TOL_FINAL,
    max_iter: int = _KAPPA_MAX_ITER,
) -> np.ndarray:
    """
    Readjust κ per node using the actual angular distances (Eq. 24).

    ``κ_i ← |κ_i + (k_i − k̄_i)·u|`` — same routine as Stage 1 (the '+' is
    correct; the paper's IV B 5 '−' is a typo, per note (d)).
    """
    n = A.shape[0]
    k_obs = A.sum(axis=1)
    cos = np.clip(V @ V.T, -1.0 + 1e-12, 1.0 - 1e-12)
    dtheta = np.arccos(cos)
    np.fill_diagonal(dtheta, 0.0)
    kappa = kappa.copy()
    for _ in range(max_iter):
        c = np.maximum(mu * np.outer(kappa, kappa), _TINY) ** (1.0 / D)
        p = 1.0 / (1.0 + np.maximum(R * dtheta / c, _TINY) ** beta)
        np.fill_diagonal(p, 0.0)
        kbar = p.sum(axis=1)
        if np.max(np.abs(kbar - k_obs)) <= tol:
            break
        u = rng.random(n)
        kappa = np.abs(kappa + (k_obs - kbar) * u)
        kappa = np.maximum(kappa, _TINY)
    return kappa


# ---------------------------------------------------------------------------
# Stage 6 — radial map to hyperbolic coordinates (§6)
# ---------------------------------------------------------------------------

def radial_map(
    kappa: np.ndarray, mu: float, R: float, D: int
) -> tuple[np.ndarray, float]:
    """
    Map hidden degrees to hyperbolic radial coordinates (Eq. 7).

    Returns ``(r, R_hat)`` where ``r_i = R̂ − (2/D)·ln(κ_i/κ_0)`` and
    ``R̂ = 2·ln(2R / (μ κ_0²)^{1/D})``, κ_0 = min_i κ_i.
    """
    kappa_0 = float(np.min(kappa))
    R_hat = 2.0 * np.log(2.0 * R / (mu * kappa_0 ** 2) ** (1.0 / D))
    r = R_hat - (2.0 / D) * np.log(kappa / kappa_0)
    return r, R_hat


# ---------------------------------------------------------------------------
# Degree-one node reinsertion (§2.5)
# ---------------------------------------------------------------------------

def _place_at_angular_separation(
    v_ref: np.ndarray,
    dtheta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Unit vector at angular separation ``dtheta`` from ``v_ref`` (uniform azimuth)."""
    D1 = len(v_ref)
    r = rng.standard_normal(D1)
    t = r - (r @ v_ref) * v_ref            # orthogonal component
    tn = np.linalg.norm(t)
    if tn <= _TINY:
        t = np.eye(D1)[0] - (np.eye(D1)[0] @ v_ref) * v_ref
        tn = np.linalg.norm(t)
    t = t / tn
    return np.cos(dtheta) * v_ref + np.sin(dtheta) * t


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def dmercator_init(
    G: nx.Graph,
    D: int,
    beta: Optional[float] = None,
    random_state: Optional[int] = None,
    n_int: int = _N_INT,
    mle_max_sweeps: int = 10,
    verbose: bool = False,
) -> dict:
    """
    Run the full original D-Mercator pipeline to produce a warm start.

    Parameters
    ----------
    G:
        Input graph (treated as unweighted; edge weights ignored).
    D:
        Similarity-space dimension (sphere S^D ⊂ R^{D+1}; the Poincaré ball
        ambient dimension is ``D+1``).
    beta:
        Inverse temperature β > D. If ``None`` it is inferred via clustering
        matching (Stage 1); otherwise β is fixed and only κ is inferred.
    random_state:
        Seed for reproducibility.
    n_int:
        Trapezoid resolution for the angular integrals.
    mle_max_sweeps:
        Maximum likelihood-maximization sweeps (Stage 3).
    verbose:
        Print β-inference progress.

    Returns
    -------
    dict with keys:

    - ``V``        : ``(N, D+1)`` unit vectors on S^D (angular positions).
    - ``kappa``    : ``(N,)`` hidden degrees.
    - ``beta``     : inferred (or fixed) inverse temperature.
    - ``mu``       : μ parameter (consistent with the returned β).
    - ``R``        : sphere radius.
    - ``R_hat``    : hyperbolic radial offset R̂.
    - ``r``        : ``(N,)`` hyperbolic radial coordinates.
    - ``nodes``    : node ordering matching the rows of ``V``/``kappa``.
    """
    rng = np.random.default_rng(random_state)
    nodes = list(G.nodes())
    N = len(nodes)

    A = nx.to_numpy_array(G, nodelist=nodes, weight=None)
    np.fill_diagonal(A, 0.0)
    degrees = A.sum(axis=1)
    avg_deg = float(degrees.mean()) if N > 0 else 1.0
    if avg_deg <= 0.0:
        avg_deg = 1.0

    if beta is not None and beta <= D:
        warnings.warn(f"beta must be > D={D}; got {beta}. Clamped to {D + 0.5}.",
                      UserWarning, stacklevel=2)
        beta = D + 0.5

    R = compute_R(N, D)
    theta = _theta_grid(n_int)
    c_emp = nx.average_clustering(G) if N > 2 else 0.0

    # ── Stage 1: κ and β (per degree class, on the full graph) ───────────
    degree_values, inverse = np.unique(degrees, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    kappa_classes, beta = infer_kappa_and_beta(
        degree_values, counts, avg_deg, c_emp, R, D, rng, theta,
        beta_fixed=beta, verbose=verbose,
    )
    mu = compute_mu(beta, D, avg_deg)
    kappa = kappa_classes[inverse]                              # per node

    # ── Drop degree-one nodes for Stages 2–3 (§2.5) ──────────────────────
    core_mask = degrees >= 2
    core_idx = np.where(core_mask)[0]
    V = np.full((N, D + 1), np.nan)

    if len(core_idx) >= D + 2:
        A_core = A[np.ix_(core_idx, core_idx)]
        kappa_core = kappa[core_idx]
        # ── Stage 2: S^D-corrected Laplacian Eigenmaps ───────────────────
        V_core = model_corrected_le(A_core, kappa_core, beta, mu, R, D, theta, rng)
        # ── Stage 3: likelihood maximization ─────────────────────────────
        V_core = mle_refine(A_core, kappa_core, beta, mu, R, D, V_core, rng,
                            max_sweeps=mle_max_sweeps)
        V[core_idx] = V_core
    else:
        # too few core nodes for a meaningful eigenproblem
        V[core_idx] = _random_unit_vectors(D + 1, len(core_idx), rng)

    # ── Reinsert degree-one (and any unplaced) nodes (§2.5) ──────────────
    for i in np.where(~core_mask)[0]:
        nbr = np.where(A[i] > 0)[0]
        if len(nbr) == 0:
            V[i] = _random_unit_vectors(D + 1, 1, rng)[0]
            continue
        j = int(nbr[0])
        if np.any(np.isnan(V[j])):
            V[j] = _random_unit_vectors(D + 1, 1, rng)[0]
        dth = _sample_angle_given_connected(
            kappa[i], np.array([kappa[j]]), beta, mu, R, D, rng, theta
        )[0]
        V[i] = _place_at_angular_separation(V[j], dth, rng)

    # ── Stage 4: final κ readjustment with actual positions ──────────────
    kappa = final_adjust_kappa(A, kappa, beta, mu, R, D, V, rng)

    # ── Stage 6: hyperbolic radial coordinates ───────────────────────────
    r, R_hat = radial_map(kappa, mu, R, D)

    return {
        "V": V,
        "kappa": kappa,
        "beta": beta,
        "mu": mu,
        "R": R,
        "R_hat": R_hat,
        "r": r,
        "nodes": nodes,
    }
