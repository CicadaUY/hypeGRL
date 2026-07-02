"""Estimate scalar parameters of the latent-geometry model from an observed graph.

Scope: this module is the single, method-agnostic home for estimating the
**scalar hyperparameters** a method needs *before* it embeds — e.g. HyperMap /
E-PSO's ``(m, L, gamma, T, zeta)``. It imports no embedders so any method can
reuse it. It does *not* estimate the unknown adjacency *entries* ``a_Omega``
(that is ``imputation.py``), solve for embeddings (the optimizers), or hold any
one embedder's warm-start logic (e.g. ``_hypermap_init.py``, ``_dmercator_init``).

Right now this covers the power-law exponent ``gamma`` of the degree
distribution together with an automatic choice of its lower cutoff ``k_min``.

Both follow Clauset, Shalizi & Newman, "Power-law distributions in empirical
data", SIAM Review 51, 661 (2009), arXiv:0706.1062:

- the discrete maximum-likelihood estimator of the exponent (their Eq. 3.7,
  Section 3.2), and
- the choice of the lower cutoff ``x_min`` by Kolmogorov-Smirnov minimisation
  (Section 3.3).

This is the intended home for the other E-PSO / HyperMap parameters too: the
temperature ``T`` (via clustering matching) and the ``m`` / ``L`` average-degree
heuristics, so that all of ``(m, L, gamma, T, zeta)`` can be inferred in one
place rather than inside a specific embedder.
"""

from __future__ import annotations

import warnings

import networkx as nx
import numpy as np
from scipy.special import zeta


def _gamma_mle(tail_degrees: np.ndarray, k_min: float) -> float:
    """Discrete power-law MLE for the exponent (CSN Eq. 3.7).

    ``tail_degrees`` are the degrees ``>= k_min``. The ``- 0.5`` is CSN's
    continuity correction for discrete (integer) data.
    """
    n = tail_degrees.size
    return 1.0 + n / np.sum(np.log(tail_degrees / (k_min - 0.5)))


def _ks_distance(tail_degrees: np.ndarray, k_min: float, gamma: float) -> float:
    """Kolmogorov-Smirnov distance between the empirical tail and the fitted
    discrete power law, over degrees ``>= k_min``.

    Compares complementary CDFs ``P(X >= k)``: the empirical one is the fraction
    of the tail at least ``k``; the fitted one is the Hurwitz-zeta ratio
    ``zeta(gamma, k) / zeta(gamma, k_min)`` (``scipy.special.zeta(gamma, k)`` is
    the Hurwitz zeta ``sum_{i>=0} (k + i)**-gamma``). Taking ``1 - CDF`` vs.
    ``CDF`` does not change the maximum absolute difference, so this is the KS
    statistic.
    """
    vals = np.unique(tail_degrees)
    emp_ccdf = np.array([(tail_degrees >= k).mean() for k in vals])
    fit_ccdf = zeta(gamma, vals) / zeta(gamma, k_min)
    return float(np.max(np.abs(emp_ccdf - fit_ccdf)))


def choose_kmin_ks(degrees) -> dict | None:
    """Choose the power-law lower cutoff ``k_min`` by KS minimisation (CSN §3.3).

    For every candidate cutoff, fit ``gamma`` on the degrees at or above it and
    score that fit by its KS distance to the data; the best cutoff is the one with
    the smallest KS distance. Too small a cutoff pulls the non-power-law body of
    the distribution into the fit (biasing ``gamma``); too large a cutoff throws
    away tail data (inflating its variance) — the KS distance trades these off.

    The candidates are the distinct degree values **with the largest dropped**,
    following the ``powerlaw`` package's ``Fit.find_xmin`` ("Don't look at last
    xmin, as that's also the xmax"). The largest value's tail is a single point,
    which any distribution fits perfectly (KS = 0); scoring it would spuriously
    select a meaningless cutoff (e.g. on a tree, whose top degree is shared by many
    nodes). Equivalently: every scored cutoff has a tail spanning at least two
    distinct degrees. With fewer than two candidates left (fewer than three
    distinct degrees), the cutoff is undefined and this returns ``None``.

    Parameters
    ----------
    degrees:
        Iterable of node degrees (e.g. ``[d for _, d in G.degree()]``).

    Returns
    -------
    dict or None
        ``{"k_min", "gamma", "ks", "n_tail"}`` for the best cutoff, where ``ks``
        is the (unitless, in ``[0, 1]``) KS distance — a large value means the
        degree distribution is not well described by a power law (e.g. a tree).
        Returns ``None`` when there are fewer than three distinct degrees.
    """
    degrees = np.asarray([d for d in degrees if d >= 1], dtype=float)
    candidates = np.unique(degrees)[:-1]  # drop the largest (its tail is one value)
    if candidates.size < 2:
        return None

    best: dict | None = None
    for k_min in candidates:
        tail = degrees[degrees >= k_min]
        gamma = _gamma_mle(tail, k_min)
        ks = _ks_distance(tail, k_min, gamma)
        if best is None or ks < best["ks"]:
            best = {
                "k_min": int(k_min),
                "gamma": gamma,
                "ks": ks,
                "n_tail": int(tail.size),
            }
    return best


def estimate_gamma(G: nx.Graph, k_min: int | None = None):
    """Estimate the power-law exponent ``gamma`` of ``G``'s degree distribution.

    The exponent is the discrete MLE (:func:`_gamma_mle`). The only free choice is
    the lower cutoff ``k_min`` above which the degree distribution is assumed to be
    power-law:

    - ``k_min=None`` (default) selects it automatically by KS minimisation
      (:func:`choose_kmin_ks`).
    - passing an explicit integer uses that cutoff directly.

    When automatic selection is not possible (fewer than three distinct degrees,
    so :func:`choose_kmin_ks` returns ``None``), it falls back to ``k_min=1`` with
    a warning — the exponent is then biased and should be treated with suspicion.

    Returns
    -------
    (gamma_hat, tail_degrees):
        the estimated exponent and the degrees actually used in the fit (those
        ``>= k_min``).
    """
    degrees = np.array([deg for _, deg in G.degree()], dtype=float)
    degrees = degrees[degrees >= 1]
    if degrees.size == 0:
        raise ValueError("graph has no nodes with degree >= 1")

    if k_min is None:
        chosen = choose_kmin_ks(degrees)
        if chosen is None:
            k_min = 1
            warnings.warn(
                "estimate_gamma: too few distinct degrees to select k_min by KS "
                "minimisation (need at least three); falling back to k_min=1, which "
                "biases gamma. Pass k_min explicitly if you know the power-law cutoff.",
                stacklevel=2,
            )
        else:
            k_min = chosen["k_min"]

    tail = degrees[degrees >= k_min]
    if tail.size == 0:
        raise ValueError(f"no nodes with degree >= k_min={k_min}")
    gamma_hat = _gamma_mle(tail, k_min)
    return gamma_hat, tail
