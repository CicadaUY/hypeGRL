# -*- coding: utf-8 -*-
"""
Tests for generative models — currently :class:`FermiDiracGenerator`.

Two branches are exercised separately, since they compute the probability
matrix in completely different ways: reusing a natively generative embedder's
own decoder, and the generic global ``r``/``t`` model on pairwise distances.

The distances behind the generic model must come from the fitted
``Representation`` (exact at every radius). The large-radius test guards the
one path that has no representation to fall back on — a
``PrecomputedEmbedder``'s raw ball coordinates.
"""

import networkx as nx
import numpy as np
import pytest
import torch

from hypegrl.embedders.hypermap import HyperMapEmbedder
from hypegrl.embedders.poincare_maps import PoincareMapsEmbedder
from hypegrl.embedders.precomputed import PrecomputedEmbedder
from hypegrl.generation.fermi_dirac import FermiDiracGenerator
from hypegrl.manifolds.polar import polar_distances

torch.set_default_dtype(torch.float64)


def _karate():
    return nx.karate_club_graph()


def _ball_coords(radii, angles):
    """Poincaré-ball coordinates at the given *hyperbolic* radii and angles."""
    rho = np.tanh(np.asarray(radii, dtype=np.float64) / 2.0)
    angles = np.asarray(angles, dtype=np.float64)
    return np.stack([rho * np.cos(angles), rho * np.sin(angles)], axis=1)


def _disk_sample(n, seed=0, max_ball_radius=0.9):
    """``n`` points uniform in area on the Poincaré disk."""
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0, 2 * np.pi, n)
    rho = np.sqrt(rng.uniform(0, 1, n)) * max_ball_radius
    return np.stack([rho * np.cos(angle), rho * np.sin(angle)], axis=1)


# ---------------------------------------------------------------------------
# probabilities(): the two branches
# ---------------------------------------------------------------------------

def test_probabilities_generative_reuses_embedder_decoder():
    """With r and t unset, a generative embedder's own decoder is used."""
    G = _karate()
    emb = HyperMapEmbedder(d=2, n_steps=0, log_every=0).fit(G)
    assert emb.is_generative()

    P = FermiDiracGenerator(emb).probabilities()
    expected = np.asarray(emb.decode(emb.embeddings_representation()))

    assert P.shape == (G.number_of_nodes(),) * 2
    np.testing.assert_allclose(P, np.clip(expected, 0.0, 1.0) * (1 - np.eye(len(P))))
    np.testing.assert_array_equal(np.diag(P), 0.0)


def test_probabilities_generic_global_model():
    """With r and t given, P is the Fermi-Dirac formula on rep distances."""
    G = _karate()
    emb = PoincareMapsEmbedder(d=2, n_steps=50, log_every=0).fit(G)
    assert not emb.is_generative()

    r, t = 2.0, 0.5
    P = FermiDiracGenerator(emb, r=r, t=t).probabilities()

    D = emb.embeddings_representation().dist().detach().cpu().numpy()
    expected = 1.0 / (np.exp((D - r) / t) + 1.0)
    np.fill_diagonal(expected, 0.0)

    np.testing.assert_allclose(P, expected)
    assert P.min() >= 0.0 and P.max() <= 1.0


def test_probabilities_symmetric_with_zero_diagonal():
    emb = PrecomputedEmbedder(_disk_sample(40, seed=1))
    P = FermiDiracGenerator(emb, r=1.0, t=0.2).probabilities()
    np.testing.assert_allclose(P, P.T)
    np.testing.assert_array_equal(np.diag(P), 0.0)


def test_probabilities_warns_when_uncalibrated():
    """Non-generative embedder with neither r nor t set falls back, loudly."""
    emb = PrecomputedEmbedder(_disk_sample(20, seed=2))
    with pytest.warns(UserWarning, match="not natively generative"):
        FermiDiracGenerator(emb).probabilities()


# ---------------------------------------------------------------------------
# The distance matrix behind the generic model
# ---------------------------------------------------------------------------

def test_distance_matrix_uses_the_representation():
    """A fitted gradient embedder's distances come from rep.dist() exactly."""
    G = _karate()
    emb = PoincareMapsEmbedder(d=2, n_steps=50, log_every=0).fit(G)

    D = FermiDiracGenerator(emb)._distance_matrix()
    expected = emb.embeddings_representation().dist().detach().cpu().numpy()

    np.testing.assert_allclose(D, expected)


def test_distance_matrix_exact_at_large_radius_without_a_representation():
    """
    A ``PrecomputedEmbedder`` has no ``Representation``, so the generator must
    re-chart its ball coordinates rather than measure in the ball — whose
    ``geoopt`` distance is hard-capped at ``2·artanh(1 − 1e-7) = 16.811243``.
    Three points at hyperbolic radius 20, pairwise far apart, sit well past
    that cap.
    """
    radii = [20.0, 20.0, 20.0]
    angles = [0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0]
    X = _ball_coords(radii, angles)
    assert np.all(np.linalg.norm(X, axis=1) < 1.0)

    truth = polar_distances(
        np.asarray(radii),
        np.stack([np.cos(angles), np.sin(angles)], axis=1),
    )
    off = ~np.eye(3, dtype=bool)
    assert truth[off].min() > 16.811243        # every pair is past the cap

    D = FermiDiracGenerator(PrecomputedEmbedder(X))._distance_matrix()

    # rtol is 1e-7, not machine epsilon: the radius survives the ball only to
    # ~9 significant digits at r = 20, since it is stored as tanh(r/2) and
    # 1 − tanh(10) ≈ 4e-9 is itself resolved to ~1e-16. That residual is the
    # cost of routing through ball coordinates at all; the 22-unit error of the
    # cap is what this test guards against.
    np.testing.assert_allclose(D, truth, rtol=1e-7)
    assert D[off].min() > 16.811243


# ---------------------------------------------------------------------------
# fit(): maximum-likelihood calibration of (r, t)
# ---------------------------------------------------------------------------

def test_fit_recovers_known_r_and_t():
    """Sample under a known (r, t), then recover them by MLE."""
    emb = PrecomputedEmbedder(_disk_sample(300, seed=3))
    r_true, t_true = 1.0, 0.3

    G = FermiDiracGenerator(emb, r=r_true, t=t_true, random_state=0).sample(1)[0]
    refit = FermiDiracGenerator(emb).fit(G)

    assert refit.r == pytest.approx(r_true, abs=0.1)
    assert refit.t == pytest.approx(t_true, abs=0.1)


def test_fit_ignores_nodes_outside_the_embedding():
    """Edges touching unknown nodes are dropped, not an error."""
    emb = PrecomputedEmbedder(_disk_sample(30, seed=4))
    G = nx.Graph()
    G.add_nodes_from(range(30))
    G.add_edges_from([(0, 1), (2, 3), (0, "ghost")])

    gen = FermiDiracGenerator(emb).fit(G)
    assert np.isfinite(gen.r) and gen.t > 0.0


# ---------------------------------------------------------------------------
# sample()
# ---------------------------------------------------------------------------

def test_sample_shape_order_and_isolated_nodes():
    nodes = [f"n{i}" for i in range(25)]
    emb = PrecomputedEmbedder(_disk_sample(25, seed=5), nodes=nodes)
    # r far below every pairwise distance => essentially no edges survive
    graphs = FermiDiracGenerator(emb, r=-40.0, t=1.0, random_state=0).sample(3)

    assert len(graphs) == 3
    for g in graphs:
        assert list(g.nodes()) == nodes      # isolated nodes kept, order kept
        assert g.number_of_edges() == 0


def test_sample_is_reproducible_and_seed_sensitive():
    emb = PrecomputedEmbedder(_disk_sample(60, seed=6))
    a = FermiDiracGenerator(emb, r=1.0, t=0.2, random_state=7).sample(1)[0]
    b = FermiDiracGenerator(emb, r=1.0, t=0.2, random_state=7).sample(1)[0]
    c = FermiDiracGenerator(emb, r=1.0, t=0.2, random_state=8).sample(1)[0]

    assert set(a.edges()) == set(b.edges())
    assert set(a.edges()) != set(c.edges())


def test_sample_before_fit_raises():
    emb = PoincareMapsEmbedder(d=2, n_steps=10, log_every=0)
    with pytest.raises(RuntimeError, match="fit"):
        FermiDiracGenerator(emb).sample(1)


# ---------------------------------------------------------------------------
# from_embeddings shortcut
# ---------------------------------------------------------------------------

def test_from_embeddings_matches_explicit_precomputed():
    X = _disk_sample(30, seed=9)
    direct = FermiDiracGenerator(PrecomputedEmbedder(X), r=1.0, t=0.2)
    shortcut = FermiDiracGenerator.from_embeddings(X, r=1.0, t=0.2)

    np.testing.assert_allclose(direct.probabilities(), shortcut.probabilities())
    assert shortcut.embedder.nodes() == list(range(30))


def test_from_embeddings_rejects_points_outside_the_ball():
    with pytest.raises(ValueError, match="strictly inside"):
        FermiDiracGenerator.from_embeddings(np.array([[1.5, 0.0]]))
