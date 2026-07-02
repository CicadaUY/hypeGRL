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
    """
    degrees = [d for _, d in nx.balanced_tree(2, 4).degree()]
    res = choose_kmin_ks(degrees)
    assert res is not None
    assert res["k_min"] < max(degrees)   # not the degenerate top cutoff
    assert res["gamma"] < 3.0            # not the spurious ~6.5


def test_choose_kmin_returns_none_with_too_few_distinct_degrees():
    """Fewer than three distinct degrees leaves < 2 candidate cutoffs -> None."""
    degs = lambda G: [d for _, d in G.degree()]  # noqa: E731
    assert choose_kmin_ks(degs(nx.cycle_graph(10))) is None  # 1 distinct degree
    assert choose_kmin_ks(degs(nx.star_graph(6))) is None    # 2 distinct degrees
    # karate has many distinct degrees, so it does select a cutoff
    assert choose_kmin_ks(degs(nx.karate_club_graph())) is not None


def test_estimate_gamma_auto_on_scale_free_graph():
    """Auto k_min on a Barabási–Albert graph gives a plausible exponent (~3)."""
    G = nx.barabasi_albert_graph(3000, 2, seed=1)
    gamma, tail = estimate_gamma(G)  # k_min chosen automatically
    assert 2.0 < gamma < 4.0
    assert tail.size > 0
    # returned degrees are the tail actually used in the fit
    assert tail.min() >= 1


def test_estimate_gamma_falls_back_with_warning_when_cutoff_undefined():
    """Too few distinct degrees to select a cutoff -> warn and fall back to k_min=1."""
    G = nx.cycle_graph(10)  # every node degree 2: a single distinct degree
    with pytest.warns(UserWarning, match="falling back to k_min=1"):
        gamma, _ = estimate_gamma(G)
    assert np.isfinite(gamma)


def test_estimate_gamma_explicit_kmin_unchanged():
    """Passing an explicit k_min bypasses selection and matches the raw MLE."""
    G = nx.barabasi_albert_graph(500, 2, seed=2)
    degrees = np.array([d for _, d in G.degree()], dtype=float)
    tail = degrees[degrees >= 3]
    gamma, used = estimate_gamma(G, k_min=3)
    assert gamma == pytest.approx(_gamma_mle(tail, 3))
    assert used.size == tail.size
