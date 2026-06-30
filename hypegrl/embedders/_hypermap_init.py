"""
HyperMap initialization: greedy sequential angle placement.

Faithfully reimplements the two-phase algorithm from:
    Papadopoulos et al., "Network Geometry Inference using Common Neighbors",
    Physical Review E, 2015.

Phase 1 (common-neighbors likelihood, nodes 0..numCN-1):
    High-degree nodes placed by maximising a Gaussian log-likelihood on the
    number of common neighbors, using 48-point Gauss-Legendre integration.

Phase 2 (Fermi-Dirac MLE, nodes numCN..N-1):
    Lower-degree nodes placed by maximising the standard Fermi-Dirac
    log-likelihood over observed/non-observed edges.

Correction steps:
    Triggered at degree thresholds (10, 20, 40, 60): re-run Phase 2
    angle placement for all Phase-2 nodes, repeated kbar times.

Radii are NOT learned — they are assigned analytically from the node's
rank in the degree-sorted order and frozen.

All coordinates are returned in H^2 polar form (r_i, theta_i) and
converted to the Poincare disk by the caller.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Optional

import networkx as nx
import numpy as np
from scipy.special import roots_legendre


@lru_cache(maxsize=None)
def _gauss_legendre(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Cached Gauss-Legendre nodes/weights on [-1, 1] (same as scipy uses)."""
    return roots_legendre(n)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CORRECTION_DEGREES = {10, 20, 40, 60}
_LOGPROB_FLOOR      = -1e9
_MAXLOGL_INIT       = -1e10


# ---------------------------------------------------------------------------
# Global parameter estimation
# ---------------------------------------------------------------------------

def estimate_global_params(
    G: nx.Graph,
    gamma: float,
    zeta: float = 1.0,
    T: float = 0.5,
    m_in: float = -1.0,
    L_in: float = -1.0,
) -> dict:
    """
    Estimate global model parameters from the graph.

    Nodes are sorted by degree descending (highest degree = earliest
    arrival, as in the PSO/S1 growth model).

    Parameters
    ----------
    G:
        Input graph.
    gamma:
        Power-law exponent of the degree distribution (gamma > 2).
    zeta:
        Curvature parameter (zeta=1 gives standard H^2).
    T:
        Temperature parameter controlling clustering (0 < T < 1).
    m_in:
        Override for m, rate at which external external links appear at the graph.
        If -1, estimated from the graph as recommended in the paper (minimum of links per new node).
    L_in:
        Override for L, the rate at which interanl links appear in the graph. If -1, estimated as recommended in the paper.

    Returns
    -------
    dict with keys:
        N, beta, zeta, T, m, L, kbar, cteR, cteL_t,
        nodes_sorted (list of node ids, degree-descending),
        degrees (dict node->degree),
        adj (dict node->set of neighbors).
    """
    N     = G.number_of_nodes()
    E     = G.number_of_edges()
    kbar  = 2 * E / N  # mean degree

    # Degree-sorted node list (descending), matching C++ "sorted by degree"
    nodes_sorted = sorted(G.nodes(), key=lambda v: G.degree(v), reverse=True)
    degrees      = {v: G.degree(v) for v in G.nodes()}
    adj          = {v: set(G.neighbors(v)) for v in G.nodes()}

    beta = 1.0 / (gamma - 1.0)

    # m: average number of links brought by each new node
    if m_in > 0:
        m = m_in
    else:
        # from the paper: "If no historical data are available, m can be set, as
        # an approximation, to the minimum observed node degree
        # in the network"
        # m = kbar / 2.0
        m = min(degrees.values())

    # L: link-density parameter (C++ uses kbar and min-degree heuristic)
    if L_in > 0:
        L = L_in
    else:
        L = (kbar - 2.0 * m) / 2.0;
        if L<0:
            L = 0
        # if abs(beta - 1.0) < 1e-9:
        #     L = m
        # else:
        #     L = m * (1.0 - N ** (-(1.0 - beta))) ** 2 / (2.0 * (1.0 - beta))

    # Constant parts (C++ lines 98-101)
    cteL_t = 2.0 * L * (1.0 - beta) / ((1.0 - N ** (-(1.0 - beta))) ** 2 * (2.0 * beta - 1.0))
    # cteL_t = (
    #     2.0 * L * (1.0 - beta)
    #     / ((1.0 - N ** (-(1.0 - beta))) ** 2 * (2.0 * beta - 1.0))
    #     if abs(beta - 0.5) > 1e-9 and abs(beta - 1.0) > 1e-9
    #     else None          # handled separately in radius computation
    # )
    cteR = (2.0 / zeta) * np.log(2.0 * T / np.sin(T * np.pi))

    return dict(
        N=N, beta=beta, zeta=zeta, T=T, m=m, L=L, kbar=kbar,
        cteL_t=cteL_t, cteR=cteR,
        nodes_sorted=nodes_sorted, degrees=degrees, adj=adj,
    )


# ---------------------------------------------------------------------------
# Radius assignment (analytical, closed-form)
# ---------------------------------------------------------------------------

def assign_radii(params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assign initial radii r_i and threshold radii R_i to all nodes. This is basically 
    the step 5 of Fig. 1 in the paper, plus the computation of the corresponding R_i 
    in equation (1). Since we're already doing the computations, we also compute 
    num_cn, the number of nodes processed via common-neighbors in phase 1, by 
    replicating the paper's condition (m_i >= i-1) plus one in the c++ code (i == 1).

    Returns
    -------
    r_init : (N,) array — initial hyperbolic radius r_i = (2/zeta)*log(i)
    R      : (N,) array — threshold radius R_i for each node (Eq. 3 in the paper)
    num_cn : int with the number of nodes processed in phase 1.
    """
    N      = params["N"]
    beta   = params["beta"]
    zeta   = params["zeta"]
    m      = params["m"]
    L      = params["L"]
    # in cteR we expect to have the "mid-term" in eq. (3) of the paper. 
    # That is to say, separate the log of the multiplication as a sum of 
    # logs and the part that does not depend on i is expected at cteR.
    cteR   = params["cteR"]
    
    # in cteL_t we expect to have the first term in the multiplication of 
    # equation (4) of the paper.
    cteL_t = params["cteL_t"]
    oneOverOneMinusBeta = 1.0 / (1.0 - beta)

    r_init  = np.zeros(N)
    R       = np.zeros(N)

    num_cn = 0
    for i in np.arange(1,N+1):
        # i  1-indexed arrival time

        r_i = (2.0 / zeta) * np.log(i)
        r_init[i-1] = r_i

        # I_i: integral of density up to time t
        I_i = oneOverOneMinusBeta * (1.0 - i ** (-(1.0 - beta)))

        # L_t: expected number of links from node t to older nodes
        # First some border cases taken from the c++ code (not discussed in the paper)
        if abs(beta - 1.0) < 1e-9:
            L_t = 2.0 * L * (N - i) * np.log(i) / (i * np.log(N) ** 2)
        elif abs(beta - 0.5) < 1e-9:
            L_t = (
                L
                * ((1.0 - i ** (-0.5)) / (1.0 - N ** (-0.5)) ** 2)
                * np.log(N / i)
            )
        else:
            # now equation (4) in the paper
            L_t = cteL_t * (((N / i) ** (2.0 * beta - 1.0)) - 1.0) * (
                1.0 - i ** (-(1.0 - beta))
            )

        m_i_t = m + L_t
        
        if (m_i_t >= i - 1) or (i == 1):
            num_cn += 1
        
        ratio = I_i / m_i_t #if m_i_t > 0 else 1e-12
        ratio = max(ratio, 1e-12)  # guard log(0) at t=1
        R[i-1] = r_i - cteR - (2.0 / zeta) * np.log(ratio)

    # r_final: in C++ Phase 2, u->setRadius(u->getInitRadius(), zeta) (line 278)
    # meaning the embedding radius equals r_init
    # r_final = r_init.copy()

    return r_init, R, num_cn


# ---------------------------------------------------------------------------
# Hyperbolic distance helpers
# ---------------------------------------------------------------------------

def hyperbolic_dist(
    r1: float, r2: float, dtheta: float, zeta: float
) -> float:
    """
    Hyperbolic distance in H^2 between two points (r1, theta1), (r2, theta2).
    dtheta = angular separation (already computed as pi - |pi - |theta1-theta2||).
    """
    if dtheta == 0.0:
        return abs(r1 - r2)
    arg = (
        np.cosh(zeta * r1) * np.cosh(zeta * r2)
        - np.sinh(zeta * r1) * np.sinh(zeta * r2) * np.cos(dtheta)
    )
    # arg = max(arg, 1.0)   # numerical safety for acosh
    return (1.0 / zeta) * np.arccosh(arg)


def _hyperbolic_dist_vec(r1, r2, dtheta, zeta):
    """
    Vectorized H^2 distance, numerically matching :func:`hyperbolic_dist`.

    ``r1``, ``r2`` and ``dtheta`` broadcast against each other (any may be a
    scalar or an array). The arithmetic is term-for-term the same as the scalar
    version — same operation order and the same ``dtheta == 0 -> |r1 - r2|``
    special case — so the per-element results are identical to looping with
    :func:`hyperbolic_dist`. ``np.maximum(arg, 1.0)`` only guards ``arccosh``
    against round-off on the (discarded) ``dtheta == 0`` entries, where ``arg``
    can dip a hair below 1; for ``dtheta > 0`` we always have ``arg >= 1`` so it
    is a no-op there.
    """
    arg = (
        np.cosh(zeta * r1) * np.cosh(zeta * r2)
        - np.sinh(zeta * r1) * np.sinh(zeta * r2) * np.cos(dtheta)
    )
    return np.where(
        dtheta == 0.0,
        np.abs(r1 - r2),
        (1.0 / zeta) * np.arccosh(np.maximum(arg, 1.0)),
    )


def angular_sep(theta1: float, theta2: float) -> float:
    """Circular angular separation in [0, pi] (vectorizes over array inputs)."""
    return np.pi - abs(np.pi - abs(theta1 - theta2))


# ---------------------------------------------------------------------------
# Fermi-Dirac connection probability
# ---------------------------------------------------------------------------

def fermi_dirac(x: float, R: float, zeta_over_2T: float) -> float:
    """P(connect) = 1 / (1 + exp(zeta/(2T) * (x - R)))."""
    return 1.0 / (1.0 + np.exp(zeta_over_2T * (x - R)))


# ---------------------------------------------------------------------------
# Phase 1: common-neighbors log-likelihood
# ---------------------------------------------------------------------------

def _pair_terms(
    a: int,
    r_init: np.ndarray,
    R: np.ndarray,
    beta: float,
    zeta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Precompute the angle-independent ``cosh``/``sinh`` products for the
    ``(a, k)`` pair, for every node ``k``, plus the pair threshold ``R_{ak}``.

    The hyperbolic distance in the CN integrand is
    ``x = (1/zeta) arccosh(A - B cos(dtheta))`` with

        A = cosh(zeta r1) cosh(zeta r2),   B = sinh(zeta r1) sinh(zeta r2),

    where ``r1`` is the older node faded to the younger node's birth radius and
    ``r2`` the younger node's birth radius. ``A``, ``B`` and ``R_{ak}`` depend
    only on radii, not on any angle, so they are computed **once** here and
    reused across the entire angular grid search and all quadrature points —
    the per-node ``sinh``/``cosh`` caching the C++ does via
    ``Node::rinit_sinh``/``rinit_cosh``.

    Returns ``(A, B, R_ak)``, each of shape ``(N,)`` indexed by ``k``.
    """
    # Larger birth radius => later arrival, so younger_k[k] flags the nodes k
    # that are younger than a. The whole branch is done per-k with np.where.
    r_a            = r_init[a]
    one_minus_beta = 1.0 - beta
    younger_k      = r_init > r_a            # node k born after node a?

    # r1 = older node's radius, faded to the younger node's birth time;
    # r2 = younger node's own birth radius; R_ak = younger node's threshold.
    r1 = np.where(
        younger_k,
        beta * r_a + one_minus_beta * r_init,     # k younger: fade a to k's birth
        beta * r_init + one_minus_beta * r_a,     # a younger: fade k to a's birth
    )
    r2   = np.where(younger_k, r_init, r_a)
    R_ak = np.where(younger_k, R, R[a])

    # A, B are the radius-only parts of cosh(zeta*x) = A - B*cos(dtheta); since
    # they hold no angle, they are computed once here and reused for every angle.
    A = np.cosh(zeta * r1) * np.cosh(zeta * r2)
    B = np.sinh(zeta * r1) * np.sinh(zeta * r2)
    return A, B, R_ak


def expected_common_neighbors(
    i: int,
    j: int,
    theta_v: float,
    *,
    cn_terms: dict,
    angles: np.ndarray,
    params: dict,
    n_quad: int = 48,
) -> tuple[float, float]:
    """
    Compute expected (lambda) and variance of common neighbors between
    node i (being placed at theta_v) and already-placed node j.

    ``cn_terms[a]`` holds the precomputed ``(A, B, R_ak)`` arrays for anchor
    node ``a`` against every ``k`` (see :func:`_pair_terms`).

    The 48-point Gauss-Legendre integral is evaluated for **all** third nodes
    ``k`` at once: the integrand is a single ``(N, n_quad)`` NumPy array, and the
    quadrature contraction is one matrix reduction — instead of one
    ``fixed_quad`` call per ``k``. The arithmetic mirrors ``scipy``'s
    ``fixed_quad`` (same nodes/weights and node mapping), so the result is
    numerically identical to the per-``k`` version.

    Returns (lambda, variance).
    """
    N            = params["N"]
    zeta         = params["zeta"]
    zeta_over_2T = zeta / (2.0 * params["T"])
    theta_u      = angles[j]

    A_v, B_v, R_v = cn_terms[i]   # (i, k) cached terms, arrays over k
    A_u, B_u, R_u = cn_terms[j]   # (j, k) cached terms, arrays over k

    # Quadrature nodes mapped from [-1, 1] to [0, 2*pi] exactly as fixed_quad
    # does (w are the matching weights); phi has shape (Q=n_quad,).
    x_roots, w = _gauss_legendre(n_quad)
    a_lo, b_hi = 0.0, 2.0 * np.pi
    phi = (b_hi - a_lo) * (x_roots + 1.0) / 2.0 + a_lo        # (Q,)

    # Angular separation of the third node (at phi) from v and from u, per node.
    dtheta_v = np.pi - np.abs(np.pi - np.abs(phi - theta_v))  # (Q,)
    dtheta_u = np.pi - np.abs(np.pi - np.abs(phi - theta_u))  # (Q,)
    cos_dv = np.cos(dtheta_v)
    cos_du = np.cos(dtheta_u)

    # Connection probability at every (node k, quadrature node phi). The cached
    # per-pair terms are column vectors over k (``[:, None]``) and the angle
    # terms are row vectors over phi (``[None, :]``); broadcasting them gives
    # (N, Q) arrays in one shot. arccosh(A - B cos(dtheta)) = zeta * distance.
    x_v = (1.0 / zeta) * np.arccosh(A_v[:, None] - B_v[:, None] * cos_dv[None, :])
    x_u = (1.0 / zeta) * np.arccosh(A_u[:, None] - B_u[:, None] * cos_du[None, :])
    p_v = 1.0 / (1.0 + np.exp(zeta_over_2T * (x_v - R_v[:, None])))
    p_u = 1.0 / (1.0 + np.exp(zeta_over_2T * (x_u - R_u[:, None])))
    integ = p_v * p_u                                          # (N, Q)

    # Quadrature: integral over phi = (b-a)/2 * sum_q w_q f_q, contracting the
    # Q axis for all k at once; then the 1/(2*pi) normalisation (C++ line 221).
    prob = (b_hi - a_lo) / 2.0 * (integ * w[None, :]).sum(axis=-1)   # (N,)
    prob /= (2.0 * np.pi)

    # A node is never its own common neighbor: drop k == i and k == j.
    prob[i] = 0.0
    prob[j] = 0.0

    lam = prob.sum()
    var = (prob * (1.0 - prob)).sum()
    return lam, var


def cn_loglikelihood(
    i: int,
    theta_v: float,
    *,
    cn_terms: dict,
    angles: np.ndarray,
    adj: dict,
    nodes_sorted: list,
    params: dict,
) -> float:
    """
    Log-likelihood for node i placed at theta_v, Phase 1 (CN-based).
    Sums Gaussian log-likelihoods of empirical vs expected CN counts.

    ``cn_terms`` carries the precomputed per-pair ``cosh``/``sinh`` terms (see
    :func:`_pair_terms`), reused across all candidate angles.
    """
    logL = 0.0
    node_v = nodes_sorted[i]

    for j in range(i):
        node_u = nodes_sorted[j]

        # Empirical common neighbors
        empirical_cn = len(adj[node_v] & adj[node_u])

        lam, var = expected_common_neighbors(
            i, j, theta_v,
            cn_terms=cn_terms, angles=angles, params=params,
        )

        if var < 1e-12:
            var = 1e-12

        log_prob = (
            -(empirical_cn - lam) ** 2 / (2.0 * var)
            - 0.5 * np.log(2.0 * np.pi * var)
        )

        # C++ lines 234-239: cap positive log-probs
        if log_prob > 0:
            if abs(empirical_cn - lam) > 1:
                log_prob = _LOGPROB_FLOOR
            else:
                log_prob = 0.0

        logL += log_prob

    return logL


# ---------------------------------------------------------------------------
# Phase 2: Fermi-Dirac MLE angle placement
# ---------------------------------------------------------------------------

def fd_loglikelihood(
    i: int,
    theta_v: float,
    *,
    r_final: np.ndarray,
    R: np.ndarray,
    angles: np.ndarray,
    is_here: np.ndarray,
    adj: dict,
    nodes_sorted: list,
    params: dict,
    nodes2compare: Optional[list[int]] = None,
) -> float:
    """
    Fermi-Dirac log-likelihood for node i placed at theta_v.

    nodes2compare: indices of nodes to compare against.
                   If None, uses all earlier nodes (0..i-1).

    Vectorized over the comparison nodes: one ``(M,)`` array of distances /
    probabilities instead of a Python loop of scalar calls. The math is
    element-wise identical to the scalar path (see :func:`_hyperbolic_dist_vec`);
    only the summation order changes, which can shift the total by a last ULP.
    """
    zeta         = params["zeta"]
    zeta_over_2T = zeta / (2.0 * params["T"])
    node_v       = nodes_sorted[i]
    compare      = nodes2compare if nodes2compare is not None else range(i)

    # Comparison nodes that are already placed in the embedding.
    J = np.fromiter((j for j in compare if is_here[j]), dtype=np.intp)
    if J.size == 0:
        return 0.0

    # node i is the youngest here, so every pair uses its threshold R[i].
    dtheta = angular_sep(theta_v, angles[J])
    x      = _hyperbolic_dist_vec(r_final[i], r_final[J], dtheta, zeta)
    P      = fermi_dirac(x, R[i], zeta_over_2T)
    P      = np.clip(P, 1e-12, 1.0 - 1e-12)

    # Edge -> log P ; non-edge -> log(1 - P).
    is_edge = np.fromiter(
        (nodes_sorted[j] in adj[node_v] for j in J), dtype=bool, count=J.size
    )
    return np.where(is_edge, np.log(P), np.log(1.0 - P)).sum()


def fd_loglikelihood_correction(
    i: int,
    theta_v: float,
    *,
    r_init: np.ndarray,
    R: np.ndarray,
    angles: np.ndarray,
    adj: dict,
    nodes_sorted: list,
    params: dict,
    nodes2compare: list[int],
) -> float:
    """
    Fermi-Dirac log-likelihood for correction steps, mirroring C++ exactly.

    Unlike fd_loglikelihood, this recomputes pair-wise radii from r_init for
    every (v, l) pair based on who arrived later (larger r_init = later arrival),
    and uses R of the later-arriving node — matching C++ lines 451-468.

    Vectorized over the comparison nodes: the per-pair "who arrived later" branch
    becomes an ``np.where`` mask, element-wise identical to the scalar loop.
    """
    zeta         = params["zeta"]
    beta         = params["beta"]
    zeta_over_2T = zeta / (2.0 * params["T"])
    node_v       = nodes_sorted[i]

    J = np.fromiter((j for j in nodes2compare if j != i), dtype=np.intp)
    if J.size == 0:
        return 0.0

    # i_after[k]: node i arrived after node J[k] (larger r_init = later arrival).
    # The faded radius always pulls the older node toward the younger one's birth
    # radius, and the threshold is the younger node's R — selected per pair.
    ri      = r_init[i]
    rj      = r_init[J]
    i_after = ri > rj
    r_v     = np.where(i_after, ri, beta * ri + (1.0 - beta) * rj)
    r_l     = np.where(i_after, beta * rj + (1.0 - beta) * ri, rj)
    R_use   = np.where(i_after, R[i], R[J])

    dtheta = angular_sep(theta_v, angles[J])
    x      = _hyperbolic_dist_vec(r_v, r_l, dtheta, zeta)
    P      = fermi_dirac(x, R_use, zeta_over_2T)
    P      = np.clip(P, 1.0e-12, 1.0 - 1.0e-12)

    is_edge = np.fromiter(
        (nodes_sorted[j] in adj[node_v] for j in J), dtype=bool, count=J.size
    )
    return np.where(is_edge, np.log(P), np.log(1.0 - P)).sum()


def _grid_search_angle(
    logL_fn,
    step: float,
    theta_start: float = 0.0,
    theta_end: float = 2.0 * np.pi,
    strict: bool = False,
) -> tuple[float, float]:
    """
    Grid search over [theta_start, theta_end) with given step.
    Returns (best_angle, best_logL).

    strict=True uses > (keep first occurrence on ties) — matches C++ Phase 1.
    strict=False uses >= (keep last occurrence on ties) — matches C++ Phase 2 / corrections.
    """
    best_angle = theta_start
    best_logL  = _MAXLOGL_INIT
    theta      = theta_start

    while theta <= theta_end:
        logL = logL_fn(theta)
        if (strict and logL > best_logL) or (not strict and logL >= best_logL):
            best_angle = theta
            best_logL  = logL
        theta += step

    return best_angle, best_logL


# ---------------------------------------------------------------------------
# Main initialization entry point
# ---------------------------------------------------------------------------

def hypermap_init(
    G: nx.Graph,
    gamma: float,
    T: float,
    zeta: float = 1.0,
    k_speedup: int = 0,
    m_in: float = -1.0,
    L_in: float = -1.0,
    corrections: bool = True,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list, dict]:
    """
    Run the HyperMap greedy initialization and return H^2 polar coordinates.

    Parameters
    ----------
    G:
        Input graph.
    gamma:
        Power-law exponent (gamma > 2).
    T:
        Temperature (0 < T < 1).
    zeta:
        Curvature (default 1.0).
    k_speedup:
        Degree threshold for the fast approximation in Phase 2.
        Nodes with degree < k_speedup use only their neighbors as
        comparison set (not all earlier nodes). Set to 0 to disable,
        matching C++ default.
    m_in, L_in:
        Optional overrides for m and L.
    corrections:
        If True, run correction steps at degree thresholds 10/20/40/60.
    verbose:
        Print progress.

    Returns
    -------
    thetas  : (N,) array of angular coordinates in [0, 2*pi).
    r_final : (N,) array of radial coordinates in H^2.
    nodes_sorted : list of node IDs in degree-descending order
                   (index i in thetas/r_final corresponds to nodes_sorted[i]).
    params  : dict of global parameters (for use by HyperMapEmbedder).
    """
    params       = estimate_global_params(G, gamma, zeta, T, m_in, L_in)
    r_init, R, num_cn    = assign_radii(params)
    N            = params["N"]
    nodes_sorted = params["nodes_sorted"]
    adj          = params["adj"]
    kbar         = params["kbar"]
    degrees      = params["degrees"]

    angles   = np.zeros(N)
    is_here  = np.zeros(N, dtype=bool)

    if verbose:
        print(f"N={N}, numCN={num_cn}, kbar={kbar:.2f}, "
              f"beta={params['beta']:.4f}, T={T}, zeta={zeta}")

    # Precompute the angle-independent cosh/sinh products for every CN anchor.
    # The nodes that appear as i or j in Phase 1 are exactly 0..num_cn-1, each
    # against all k; these terms are reused across the whole angular grid search.
    cn_terms = {
        a: _pair_terms(a, r_init, R, params["beta"], zeta)
        for a in range(num_cn)
    }

    # ── Phase 1: common-neighbors ─────────────────────────────────────────
    for i in range(num_cn):
        is_here[i] = True
        t = i + 1

        if t == 1:
            angles[i] = np.pi
            if verbose:
                print(f"[Phase1] t={t} node={nodes_sorted[i]} "
                      f"theta={angles[i]:.6f} r={r_init[i]}")
            continue

        step = min(1.0 / t, 0.01)

        def logL_cn(theta_v, _i=i):
            return cn_loglikelihood(
                _i, theta_v,
                cn_terms=cn_terms, angles=angles,
                adj=adj, nodes_sorted=nodes_sorted, params=params,
            )

        best_angle, _ = _grid_search_angle(logL_cn, step, strict=True)
        angles[i] = best_angle

        if verbose:
            print(f"[Phase1] t={t} node={nodes_sorted[i]} "
                  f"theta={best_angle:.6f} r={r_init[i]}")

    r_final = np.zeros_like(r_init)
    # ── Phase 2: Fermi-Dirac MLE ──────────────────────────────────────────
    for i in range(num_cn, N):
        t = i + 1
        is_here[i] = True

        # In C++, r_final for Phase-2 nodes = r_init (line 278)
        r_final[i] = r_init[i]

        # Update Phase-2 radii of earlier nodes: beta*r_l + (1-beta)*r_t
        # This reflects "node t arrives now" in the growth model
        for j in range(i):
            r_final[j] = (
                params["beta"] * r_init[j]
                + (1.0 - params["beta"]) * r_init[i]
            )

        if t == 1:
            angles[i] = np.pi
            continue

        step = min(1.0 / t, 0.01)

        # k_speedup: low-degree nodes compare only against neighbors
        if k_speedup > 0 and degrees[nodes_sorted[i]] < k_speedup:
            node_v       = nodes_sorted[i]
            nodes2compare = [
                j for j in range(i)
                if nodes_sorted[j] in adj[node_v] and is_here[j]
            ]
        else:
            nodes2compare = None   # all earlier nodes

        def logL_fd(theta_v, _i=i, _n2c=nodes2compare):
            return fd_loglikelihood(
                _i, theta_v,
                r_final=r_final, R=R, angles=angles,
                is_here=is_here, adj=adj,
                nodes_sorted=nodes_sorted, params=params,
                nodes2compare=_n2c,
            )

        best_angle, _ = _grid_search_angle(logL_fd, step)
        angles[i] = best_angle

        # k_speedup refinement: narrow search around best angle
        if k_speedup > 0 and degrees[nodes_sorted[i]] < k_speedup:
            C         = 200.0
            Delta     = C * step
            theta_min = max(best_angle - Delta, 0.0)
            theta_max = min(best_angle + Delta, 2.0 * np.pi)

            def logL_fd_full(theta_v, _i=i):
                return fd_loglikelihood(
                    _i, theta_v,
                    r_final=r_final, R=R, angles=angles,
                    is_here=is_here, adj=adj,
                    nodes_sorted=nodes_sorted, params=params,
                    nodes2compare=None,   # all nodes for refinement
                )

            refined_angle, _ = _grid_search_angle(
                logL_fd_full, step, theta_min, theta_max
            )
            angles[i] = refined_angle

        if verbose:
            print(f"[Phase2] t={t} node={nodes_sorted[i]} "
                  f"theta={angles[i]:.6f} r={r_final[i]:.6f} "
                  f"k={degrees[nodes_sorted[i]]}")

        # ── Correction steps (triggered at degree thresholds) ─────────────
        if corrections and i + 1 < N:
            deg_curr = degrees[nodes_sorted[i]]
            deg_next = degrees[nodes_sorted[i + 1]]

            if deg_next < deg_curr and deg_curr in _CORRECTION_DEGREES:
                if verbose:
                    print(f"Running correction steps for k>={deg_curr}")

                # Phase-2 nodes (indices num_cn..i)
                phase2_indices = list(range(num_cn, i + 1))
                all_indices    = list(range(i + 1))

                for rnd in range(int(round(kbar))):
                    if verbose:
                        print(f"  round {rnd + 1}")

                    for jj in phase2_indices:
                        step_c = min(1.0 / (jj + 1), 0.01)

                        def logL_corr(theta_v, _jj=jj):
                            return fd_loglikelihood_correction(
                                _jj, theta_v,
                                r_init=r_init, R=R, angles=angles,
                                adj=adj,
                                nodes_sorted=nodes_sorted, params=params,
                                nodes2compare=all_indices,
                            )

                        new_angle, _ = _grid_search_angle(logL_corr, step_c)
                        angles[jj]   = new_angle

    return angles, r_final, nodes_sorted, params
