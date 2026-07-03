"""Estimate scalar parameters of the latent-geometry model from an observed graph.

Scope: this module is the single, method-agnostic home for estimating the
**scalar hyperparameters** a method needs *before* it embeds — e.g. HyperMap /
E-PSO's ``(m, L, gamma, T, zeta)``. It imports no embedders so any method can
reuse it. It does *not* estimate the unknown adjacency *entries* ``a_Omega``
(that is ``imputation.py``), solve for embeddings (the optimizers), or hold any
one embedder's warm-start logic (e.g. ``_hypermap_init.py``, ``_dmercator_init``).

Each estimator documents its own method and source in its own docstring. So far
this covers the power-law exponent ``gamma`` of the degree distribution with an
automatic choice of the lower cutoff ``k_min`` (:func:`estimate_gamma`,
:func:`choose_kmin_ks`) and a goodness-of-fit test for the power-law hypothesis
(:func:`power_law_gof`). The ``m`` / ``L`` average-degree heuristics could also
land here (pure graph → scalar). The E-PSO temperature ``T`` does **not**: its
estimator (E-PSO §V connection-probability slope fit) needs an embedding, so it
lives on the embedder as ``HyperMapEmbedder.estimate_temperature``, not here.
"""

from __future__ import annotations

import warnings

import networkx as nx
import numpy as np
from scipy.special import zeta

# Fallback exponent when gamma cannot be estimated (degree distribution not
# power-law). A neutral midpoint of the usual empirical scale-free range [2, 3];
# valid for the PSO model, which requires gamma >= 2.
DEFAULT_GAMMA = 2.5

# Minimum number of degrees a candidate tail must retain to be an eligible cutoff.
# CSN §3.2: "n ≳ 50 is a reasonable rule of thumb for extracting reliable parameter
# estimates. Data sets smaller than this should be treated with caution." CSN give
# 50 as advisory guidance on estimate reliability; using it as a hard filter on
# candidate cutoffs (below) is our design choice. It bounds tail size, not the
# exponent — gamma is left uncapped (HyperMap/E-PSO admits any gamma >= 2).
DEFAULT_MIN_TAIL = 50


def _gamma_mle(tail_degrees: np.ndarray, k_min: float) -> float:
    """Discrete power-law MLE for the exponent (CSN Eq. 3.7).

    CSN = Clauset, Shalizi & Newman, "Power-law distributions in empirical data",
    SIAM Review 51, 661 (2009), arXiv:0706.1062 (the shorthand is reused below).
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
    den = zeta(gamma, k_min)
    if not np.isfinite(den) or den == 0.0:
        # gamma so large the fit is a point mass at k_min (den underflows to 0,
        # which would give 0/0 = nan). CCDF: P(X>=k_min)=1, P(X>k_min)=0.
        fit_ccdf = (vals <= k_min).astype(float)
    else:
        fit_ccdf = zeta(gamma, vals) / den
    return float(np.max(np.abs(emp_ccdf - fit_ccdf)))


def _discrete_power_law_sampler(gamma: float, k_min: int, kmax: int):
    """Return a ``draw(size, rng)`` that samples integers from the discrete power
    law ``P(k) ∝ k**-gamma``, ``k >= k_min``, by inverse-CDF lookup.

    The CDF ``F(k) = P(X <= k) = 1 - zeta(gamma, k+1)/zeta(gamma, k_min)`` is
    tabulated once over ``k in [k_min, kmax]``; a uniform draw is mapped through it
    with ``searchsorted``. Values are **capped at ``kmax``** — a numerical guard we
    add (not from CSN) so the tail is finite; with ``kmax`` well past the observed
    degrees the truncated probability is negligible and its effect on the KS
    statistic is immaterial.
    """
    ks = np.arange(k_min, kmax + 1)
    cdf = 1.0 - zeta(gamma, ks + 1) / zeta(gamma, k_min)  # P(X <= k), ascending

    def draw(size, rng):
        idx = np.searchsorted(cdf, rng.random(size), side="left")
        return ks[np.clip(idx, 0, ks.size - 1)]

    return draw


def choose_kmin_ks(degrees, min_tail: int = DEFAULT_MIN_TAIL) -> dict | None:
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

    Candidates whose tail retains fewer than ``min_tail`` degrees are also skipped:
    pure KS minimisation keeps shrinking as the tail shrinks, so on a bell-shaped
    (non-power-law) degree sequence it would otherwise wander far into the sparse
    tail and report a spuriously steep exponent off a handful of points. The floor
    keeps the selected fit statistically supported (see ``DEFAULT_MIN_TAIL``). It
    bounds tail size only — the exponent stays uncapped. When no candidate retains
    ``min_tail`` degrees (e.g. a graph too small to have a well-supported tail),
    this returns ``None``.

    Parameters
    ----------
    degrees:
        Iterable of node degrees (e.g. ``[d for _, d in G.degree()]``).
    min_tail:
        Minimum number of degrees a candidate tail must retain to be eligible
        (default :data:`DEFAULT_MIN_TAIL`). Pass ``1`` to disable the floor.

    Returns
    -------
    dict or None
        ``{"k_min", "gamma", "ks", "n_tail"}`` for the best cutoff, where ``ks``
        is the (unitless, in ``[0, 1]``) KS distance — a large value means the
        degree distribution is not well described by a power law (e.g. a tree).
        Returns ``None`` when fewer than three distinct degrees exist, or when no
        candidate cutoff retains at least ``min_tail`` degrees.
    """
    degrees = np.asarray([d for d in degrees if d >= 1], dtype=float)
    candidates = np.unique(degrees)[:-1]  # drop the largest (its tail is one value)
    if candidates.size < 2:
        return None

    best: dict | None = None
    for k_min in candidates:
        tail = degrees[degrees >= k_min]
        if tail.size < min_tail:  # too few points for a reliable fit (CSN §3.2)
            continue
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


def estimate_gamma(
    G: nx.Graph, k_min: int | None = None, fallback_gamma: float = DEFAULT_GAMMA
):
    """Estimate the power-law exponent ``gamma`` of ``G``'s degree distribution.

    The exponent is the discrete MLE (:func:`_gamma_mle`). The only free choice is
    the lower cutoff ``k_min`` above which the degree distribution is assumed to be
    power-law:

    - ``k_min=None`` (default) selects it automatically by KS minimisation
      (:func:`choose_kmin_ks`).
    - passing an explicit integer uses that cutoff directly.

    When a cutoff cannot be selected (:func:`choose_kmin_ks` returns ``None`` —
    too few distinct degrees, or no cutoff with a well-supported tail), no
    exponent is meaningful. Rather than fit a biased — possibly invalid, i.e.
    ``< 2`` — value, this returns ``fallback_gamma`` with a warning.

    Returns
    -------
    (gamma_hat, tail_degrees):
        the estimated exponent and the degrees actually used in the fit (those
        ``>= k_min``); on fallback, ``fallback_gamma`` and the full degree array.
    """
    degrees = np.array([deg for _, deg in G.degree()], dtype=float)
    degrees = degrees[degrees >= 1]
    if degrees.size == 0:
        raise ValueError("graph has no nodes with degree >= 1")

    if k_min is None:
        chosen = choose_kmin_ks(degrees)
        if chosen is None:
            warnings.warn(
                "estimate_gamma: cannot select a power-law cutoff (too few distinct "
                "degrees, or no cutoff with a well-supported tail) — using fallback "
                f"gamma={fallback_gamma}. Pass k_min (or gamma) explicitly to "
                "override.",
                stacklevel=2,
            )
            return fallback_gamma, degrees
        k_min = chosen["k_min"]

    tail = degrees[degrees >= k_min]
    if tail.size == 0:
        raise ValueError(f"no nodes with degree >= k_min={k_min}")
    gamma_hat = _gamma_mle(tail, k_min)
    return gamma_hat, tail


# Below this p-value the power-law hypothesis is rejected (CSN §4).
GOF_PLAUSIBLE_P = 0.1


def power_law_gof(degrees, n_bootstrap: int = 2000, seed=None) -> dict | None:
    """Goodness-of-fit test for the power-law hypothesis (CSN §4 bootstrap).

    Answers *"is a power law even a plausible model for this degree sequence?"* —
    a separate question from which cutoff fits best (:func:`choose_kmin_ks`) or what
    the exponent is (:func:`estimate_gamma`). Run it deliberately; it is never
    invoked automatically.

    Method (semiparametric bootstrap; Clauset, Shalizi & Newman, arXiv:0706.1062,
    §4):

    1. Fit the data → ``(k_min, gamma, D)`` via :func:`choose_kmin_ks` (``D`` is the
       observed KS distance).
    2. Generate ``n_bootstrap`` synthetic degree sequences of the same length. Each
       point is drawn, with probability ``n_tail/n``, from the *fitted* power law
       (``>= k_min``), and otherwise sampled uniformly (with replacement) from the
       *observed* degrees *below* ``k_min``. This preserves the real non-power-law
       body and grafts on a genuine power-law tail.
    3. Refit each synthetic sequence from scratch (its own KS-selected cutoff) and
       record its KS distance ``D_synth``.
    4. ``p = fraction of synthetic sequences with D_synth >= D``.

    Interpretation (the commonly-misread part, see CSN §4): a **small** ``p`` (CSN
    use the threshold ``< 0.1``) **rejects** the power law; a **large** ``p`` only
    means the power law is *plausible*, not that it is correct or the best of
    competing models.

    Parameters
    ----------
    degrees:
        Iterable of node degrees (e.g. ``[d for _, d in G.degree()]``).
    n_bootstrap:
        Number of synthetic sequences. CSN note ~2500 gives ``p`` accurate to
        ~±0.01; the default 1000 (~±0.015) is faster and usually enough to act on.
    seed:
        Seed for the bootstrap RNG; set it for reproducible ``p``.

    Returns
    -------
    dict or None
        ``{"p_value", "plausible", "D", "k_min", "gamma", "n_tail", "n_bootstrap"}``
        where ``plausible = p_value >= 0.1``. Returns ``None`` when the data cannot be
        fit at all (:func:`choose_kmin_ks` returns ``None``).
    """
    degrees = np.asarray([d for d in degrees if d >= 1], dtype=float)
    n = degrees.size
    fit = choose_kmin_ks(degrees)
    if fit is None:
        return None
    k_min, gamma, D_obs, n_tail = fit["k_min"], fit["gamma"], fit["ks"], fit["n_tail"]

    rng = np.random.default_rng(seed)
    body = degrees[degrees < k_min]   # observed non-tail values, preserved verbatim
    p_tail = n_tail / n               # P(a synthetic point comes from the fitted tail)
    kmax = int(max(1000, 100 * degrees.max()))
    draw_tail = _discrete_power_law_sampler(gamma, k_min, kmax)

    exceed = valid = 0
    for _ in range(n_bootstrap):
        from_tail = rng.random(n) < p_tail
        synth = np.empty(n)
        synth[from_tail] = draw_tail(int(from_tail.sum()), rng)
        n_body = n - int(from_tail.sum())
        if n_body:
            synth[~from_tail] = rng.choice(body, size=n_body, replace=True)
        sfit = choose_kmin_ks(synth)
        if sfit is None:            # synthetic sequence too degenerate to refit; skip
            continue
        valid += 1
        exceed += sfit["ks"] >= D_obs
    if valid == 0:
        warnings.warn(
            "power_law_gof: no synthetic sequence could be refit; p-value undefined.",
            stacklevel=2,
        )
        p_value = float("nan")
    else:
        p_value = exceed / valid

    return {
        "p_value": p_value,
        "plausible": bool(p_value >= GOF_PLAUSIBLE_P),
        "D": D_obs,
        "k_min": k_min,
        "gamma": gamma,
        "n_tail": n_tail,
        "n_bootstrap": n_bootstrap,
    }
