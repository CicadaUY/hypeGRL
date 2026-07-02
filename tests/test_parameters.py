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

    res = choose_kmin_ks(degrees, min_tail=50)
    assert res is not None
    assert res["gamma"] == pytest.approx(a_true, abs=0.2)
    assert res["k_min"] >= 1
    assert res["n_tail"] >= 50
    assert 0.0 <= res["ks"] <= 1.0


def test_choose_kmin_returns_none_when_tail_too_small():
    """No candidate cutoff can satisfy an unreachably large ``min_tail``."""
    degrees = [d for _, d in nx.karate_club_graph().degree()]
    assert choose_kmin_ks(degrees, min_tail=1000) is None


def test_estimate_gamma_auto_on_scale_free_graph():
    """Auto k_min on a Barabási–Albert graph gives a plausible exponent (~3)."""
    G = nx.barabasi_albert_graph(3000, 2, seed=1)
    gamma, tail = estimate_gamma(G)  # k_min chosen automatically
    assert 2.0 < gamma < 4.0
    assert tail.size > 0
    # returned degrees are the tail actually used in the fit
    assert tail.min() >= 1


def test_estimate_gamma_small_graph_falls_back_with_warning():
    """A graph too small to select a cutoff warns and falls back to k_min=1."""
    G = nx.karate_club_graph()
    with pytest.warns(UserWarning, match="falling back to k_min=1"):
        gamma, _ = estimate_gamma(G, min_tail=1000)
    assert np.isfinite(gamma)


def test_estimate_gamma_explicit_kmin_unchanged():
    """Passing an explicit k_min bypasses selection and matches the raw MLE."""
    G = nx.barabasi_albert_graph(500, 2, seed=2)
    degrees = np.array([d for _, d in G.degree()], dtype=float)
    tail = degrees[degrees >= 3]
    gamma, used = estimate_gamma(G, k_min=3)
    assert gamma == pytest.approx(_gamma_mle(tail, 3))
    assert used.size == tail.size
