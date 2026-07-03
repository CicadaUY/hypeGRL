"""Tests for graph-parameter estimation (``hypegrl.inference.parameters``).

Focus: the power-law exponent ``gamma`` and the automatic KS choice of the lower
cutoff ``k_min`` (Clauset-Shalizi-Newman, arXiv:0706.1062). The KS selection is
checked on a synthetic sample whose true exponent is known, plus the small-graph
fallback and the explicit-cutoff path.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from hypegrl.inference.parameters import (
    _gamma_mle,
    choose_kmin_ks,
    estimate_gamma,
    power_law_gof,
)


def test_gamma_mle_matches_closed_form():
    """The exponent estimator is exactly the CSN discrete MLE formula."""
    tail = np.array([3.0, 4, 5, 8, 12, 20])
    k_min = 3
    expected = 1.0 + tail.size / np.sum(np.log(tail / (k_min - 0.5)))
    assert _gamma_mle(tail, k_min) == pytest.approx(expected)


def test_choose_kmin_recovers_known_exponent():
    """On a Zipf sample with exponent a, KS selection recovers gamma ≈ a.

    ``numpy`` ``zipf(a)`` draws integers with ``P(k) ∝ k**-a`` (k >= 1), i.e. a
    discrete power law of exponent ``a``. With a proper lower cutoff the MLE
    should return an exponent close to the truth.
    """
    a_true = 2.5
    rng = np.random.default_rng(0)
    degrees = rng.zipf(a_true, size=20_000)

    res = choose_kmin_ks(degrees)
    assert res is not None
    assert res["gamma"] == pytest.approx(a_true, abs=0.2)
    assert res["k_min"] >= 1
    assert 0.0 <= res["ks"] <= 1.0


def test_choose_kmin_excludes_degenerate_max_cutoff():
    """The max-degree cutoff (single-value tail, KS=0) is never selected.

    A balanced binary tree has degrees {1, 2, 3} with the top value (3) shared by
    many nodes; scoring k_min=3 would fit that single-value tail perfectly (KS=0)
    and return a meaningless gamma (~6.5). Dropping the largest candidate prevents
    that: the chosen cutoff is below the max degree and gamma is the honest MLE.
    (Depth 6 keeps every candidate tail above the min_tail floor.)
    """
    degrees = [d for _, d in nx.balanced_tree(2, 6).degree()]
    res = choose_kmin_ks(degrees)
    assert res is not None
    assert res["k_min"] < max(degrees)   # not the degenerate top cutoff
    assert res["gamma"] < 3.0            # not the spurious ~6.5


def test_choose_kmin_returns_none_with_too_few_distinct_degrees():
    """Fewer than three distinct degrees leaves < 2 candidate cutoffs -> None."""
    degs = lambda G: [d for _, d in G.degree()]  # noqa: E731
    assert choose_kmin_ks(degs(nx.cycle_graph(10))) is None  # 1 distinct degree
    assert choose_kmin_ks(degs(nx.star_graph(6))) is None    # 2 distinct degrees
    # a large graph with many distinct degrees and a well-supported tail selects one
    assert choose_kmin_ks(degs(nx.barabasi_albert_graph(500, 2, seed=0))) is not None


def test_choose_kmin_min_tail_floor_without_capping_gamma():
    """The min_tail floor rejects under-supported cutoffs but leaves gamma uncapped.

    Pure KS minimisation keeps shrinking as the tail shrinks, so on a bell-shaped
    (non-power-law) sequence it drifts into a handful of tail points and reports a
    spuriously steep exponent. The floor (CSN §3.2 rule of thumb, n >= 50) keeps
    the selected tail statistically supported. It bounds tail *size*, not the
    exponent: HyperMap/E-PSO admits any gamma >= 2, so gamma is left uncapped.
    """
    degrees = (np.random.default_rng(1).poisson(5, size=3000) + 1).astype(float)

    # Disabling the floor lets KS drift into a tiny, unreliable tail.
    assert choose_kmin_ks(degrees, min_tail=1)["n_tail"] < 50

    tight = choose_kmin_ks(degrees)          # default floor
    assert tight["n_tail"] >= 50             # selected tail stays well-supported
    assert tight["gamma"] > 3.0              # ...yet gamma is not capped at 3


def test_choose_kmin_returns_none_when_no_supported_tail():
    """A graph too small for any tail to reach the floor returns None (tunable)."""
    karate = [d for _, d in nx.karate_club_graph().degree()]  # N=34 < min_tail
    assert choose_kmin_ks(karate) is None
    assert choose_kmin_ks(karate, min_tail=5) is not None  # lowering it recovers a fit


def test_estimate_gamma_auto_on_scale_free_graph():
    """Auto k_min on a Barabási–Albert graph gives a plausible exponent (~3)."""
    G = nx.barabasi_albert_graph(3000, 2, seed=1)
    gamma, tail = estimate_gamma(G)  # k_min chosen automatically
    assert 2.0 < gamma < 4.0
    assert tail.size > 0
    # returned degrees are the tail actually used in the fit
    assert tail.min() >= 1


def test_estimate_gamma_falls_back_to_default_with_warning():
    """Too few distinct degrees -> warn and return the fixed fallback exponent."""
    from hypegrl.inference.parameters import DEFAULT_GAMMA

    G = nx.cycle_graph(10)  # every node degree 2: a single distinct degree
    with pytest.warns(UserWarning, match="fallback gamma"):
        gamma, _ = estimate_gamma(G)
    assert gamma == DEFAULT_GAMMA
    assert gamma >= 2.0  # valid for the PSO model, unlike the old k_min=1 MLE (~1.72)

    # the default is overridable
    with pytest.warns(UserWarning, match="fallback gamma"):
        gamma2, _ = estimate_gamma(G, fallback_gamma=3.0)
    assert gamma2 == 3.0


def test_estimate_gamma_explicit_kmin_unchanged():
    """Passing an explicit k_min bypasses selection and matches the raw MLE."""
    G = nx.barabasi_albert_graph(500, 2, seed=2)
    degrees = np.array([d for _, d in G.degree()], dtype=float)
    tail = degrees[degrees >= 3]
    gamma, used = estimate_gamma(G, k_min=3)
    assert gamma == pytest.approx(_gamma_mle(tail, 3))
    assert used.size == tail.size


# ── Goodness-of-fit (CSN §4 bootstrap) ───────────────────────────────────────

def test_gof_returns_none_when_unfittable():
    """A degree sequence that can't be fit at all -> None (like choose_kmin_ks)."""
    assert power_law_gof([d for _, d in nx.cycle_graph(20).degree()]) is None


def test_gof_dict_shape_and_p_range():
    """The result carries the fit summary and a p-value in [0, 1]."""
    rng = np.random.default_rng(0)
    res = power_law_gof(rng.zipf(2.5, size=1000), n_bootstrap=100, seed=0)
    expected_keys = {"p_value", "plausible", "D", "k_min",
                     "gamma", "n_tail", "n_bootstrap"}
    assert set(res) == expected_keys
    assert 0.0 <= res["p_value"] <= 1.0
    assert res["plausible"] == (res["p_value"] >= 0.1)


def test_gof_does_not_reject_true_power_law():
    """A genuine power-law sample is judged plausible (large p)."""
    rng = np.random.default_rng(1)
    res = power_law_gof(rng.zipf(2.5, size=2000), n_bootstrap=200, seed=1)
    assert res["plausible"]
    assert res["p_value"] >= 0.1


def test_gof_rejects_non_power_law():
    """A tree (body-dominated, not heavy-tailed) is robustly rejected (p ~ 0).

    A balanced tree's degrees pile up at the low end (mostly leaves), so the
    KS-selected cutoff stays at the bottom where the fit is poor (large D), and the
    bootstrap rejects across seeds. (A bell-shaped sequence like Poisson/ER is *not*
    a robust reject case for this single-distribution GOF: its sparse upper tail is
    weakly power-law-compatible, so the test can call it plausible — that comparison
    is the job of a likelihood-ratio test against an alternative, not of CSN §4.)
    """
    degrees = [d for _, d in nx.balanced_tree(2, 7).degree()]  # N=255
    res = power_law_gof(degrees, n_bootstrap=200, seed=0)
    assert not res["plausible"]
    assert res["p_value"] < 0.1


def test_gof_reproducible_with_seed():
    """Same seed -> identical p-value (the test is otherwise stochastic)."""
    degrees = np.random.default_rng(3).zipf(2.5, size=1000)
    a = power_law_gof(degrees, n_bootstrap=100, seed=7)
    b = power_law_gof(degrees, n_bootstrap=100, seed=7)
    assert a["p_value"] == b["p_value"]
