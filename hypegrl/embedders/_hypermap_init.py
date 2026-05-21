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
from typing import Optional

import networkx as nx
import numpy as np
from scipy.integrate import fixed_quad


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


def angular_sep(theta1: float, theta2: float) -> float:
    """Circular angular separation in [0, pi]."""
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

def _cn_integrand(
    phi: float,
    *,
    zeta: float,
    zeta_over_2T: float,
    theta_u: float,
    theta_v: float,
    ruk1: float, ruk2: float,   # radii for u-k pair
    rvk1: float, rvk2: float,   # radii for v-k pair
    R_u_k: float,
    R_v_k: float,
) -> float:
    """
    Integrand for expected common neighbors (toIntegrate::operator() in C++).
    phi is the angle of the third node k being integrated over.
    """
    dtheta_v = angular_sep(phi, theta_v)
    dtheta_u = angular_sep(phi, theta_u)

    x_v = hyperbolic_dist(rvk1, rvk2, dtheta_v, zeta)
    x_u = hyperbolic_dist(ruk1, ruk2, dtheta_u, zeta)

    p_v = fermi_dirac(x_v, R_v_k, zeta_over_2T)
    p_u = fermi_dirac(x_u, R_u_k, zeta_over_2T)
    return p_v * p_u


def expected_common_neighbors(
    i: int,
    j: int,
    theta_v: float,
    *,
    r_init: np.ndarray,
    R: np.ndarray,
    angles: np.ndarray,
    params: dict,
    n_quad: int = 48,
) -> tuple[float, float]:
    """
    Compute expected (lambda) and variance of common neighbors between
    node i (being placed at theta_v) and already-placed node j.

    Returns (lambda, variance).
    """
    N            = params["N"]
    zeta         = params["zeta"]
    beta         = params["beta"]
    zeta_over_2T = zeta / (2.0 * params["T"])
    theta_u      = angles[j]

    lam = 0.0
    var = 0.0

    for k in range(N):
        if k == i or k == j:
            continue

        r_l = r_init[k]

        # Determine radii for (v, k) pair — who arrived later?
        if r_l > r_init[i]:   # k arrived after v
            rvk2 = r_l
            R_v_k = R[k]
            rvk1  = beta * r_init[i] + (1.0 - beta) * r_l
        else:                  # v arrived after k
            rvk2  = r_init[i]
            R_v_k = R[i]
            rvk1  = beta * r_l + (1.0 - beta) * r_init[i]

        # Determine radii for (u, k) pair
        if r_l > r_init[j]:   # k arrived after u
            ruk2  = r_l
            R_u_k = R[k]
            ruk1  = beta * r_init[j] + (1.0 - beta) * r_l
        else:                  # u arrived after k
            ruk2  = r_init[j]
            R_u_k = R[j]
            ruk1  = beta * r_l + (1.0 - beta) * r_init[j]

        # 48-point Gauss-Legendre integration over phi in [0, 2*pi]
        prob, _ = fixed_quad(
            lambda phi: np.array([
                _cn_integrand(
                    p,
                    zeta=zeta, zeta_over_2T=zeta_over_2T,
                    theta_u=theta_u, theta_v=theta_v,
                    ruk1=ruk1, ruk2=ruk2,
                    rvk1=rvk1, rvk2=rvk2,
                    R_u_k=R_u_k, R_v_k=R_v_k,
                )
                for p in np.atleast_1d(phi)
            ]),
            0.0, 2.0 * np.pi,
            n=n_quad,
        )
        prob /= (2.0 * np.pi)   # normalize (C++ line 221)
        lam += prob
        var += prob * (1.0 - prob)

    return lam, var


def cn_loglikelihood(
    i: int,
    theta_v: float,
    *,
    r_init: np.ndarray,
    R: np.ndarray,
    angles: np.ndarray,
    adj: dict,
    nodes_sorted: list,
    params: dict,
) -> float:
    """
    Log-likelihood for node i placed at theta_v, Phase 1 (CN-based).
    Sums Gaussian log-likelihoods of empirical vs expected CN counts.
    """
    logL = 0.0
    node_v = nodes_sorted[i]

    for j in range(i):
        node_u = nodes_sorted[j]

        # Empirical common neighbors
        empirical_cn = len(adj[node_v] & adj[node_u])

        lam, var = expected_common_neighbors(
            i, j, theta_v,
            r_init=r_init, R=R, angles=angles, params=params,
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
    """
    zeta_over_2T = params["zeta"] / (2.0 * params["T"])
    node_v       = nodes_sorted[i]
    compare      = nodes2compare if nodes2compare is not None else range(i)

    logL = 0.0
    for j in compare:
        if not is_here[j]:
            continue
        node_u = nodes_sorted[j]

        dtheta = angular_sep(theta_v, angles[j])
        x = hyperbolic_dist(r_final[i], r_final[j], dtheta, params["zeta"])
        P = fermi_dirac(x, R[i], zeta_over_2T)
        P = np.clip(P, 1e-12, 1.0 - 1e-12)

        if node_u in adj[node_v]:
            logL += np.log(P)
        else:
            logL += np.log(1.0 - P)

    return logL


def _grid_search_angle(
    logL_fn,
    step: float,
    theta_start: float = 0.0,
    theta_end: float = 2.0 * np.pi,
) -> tuple[float, float]:
    """
    Grid search over [theta_start, theta_end) with given step.
    Returns (best_angle, best_logL).
    """
    best_angle = theta_start
    best_logL  = _MAXLOGL_INIT
    theta      = theta_start

    while theta <= theta_end:
        logL = logL_fn(theta)
        if logL >= best_logL:
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

    # ── Phase 1: common-neighbors ─────────────────────────────────────────
    for i in range(num_cn):
        is_here[i] = True
        t = i + 1

        if t == 1:
            angles[i] = np.pi
            continue

        step = min(1.0 / t, 0.01)

        def logL_cn(theta_v, _i=i):
            return cn_loglikelihood(
                _i, theta_v,
                r_init=r_init, R=R, angles=angles,
                adj=adj, nodes_sorted=nodes_sorted, params=params,
            )

        best_angle, _ = _grid_search_angle(logL_cn, step)
        angles[i] = best_angle

        if verbose:
            print(f"[Phase1] t={t} node={nodes_sorted[i]} "
                  f"theta={best_angle:.6f}")

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
                            return fd_loglikelihood(
                                _jj, theta_v,
                                r_final=r_final, R=R, angles=angles,
                                is_here=is_here, adj=adj,
                                nodes_sorted=nodes_sorted, params=params,
                                nodes2compare=all_indices,
                            )

                        new_angle, _ = _grid_search_angle(logL_corr, step_c)
                        angles[jj]   = new_angle

    return angles, r_final, nodes_sorted, params
