# -*- coding: utf-8 -*-
"""
Tests for :class:`~hypegrl.representations.tangent.TangentRepresentation` — the
Euclidean tangent-space chart ``z = r·v`` (Mishne et al. 2023, §4).

Same geometry as the polar charts (``dist()`` routes through the exact
``polar_distances``); the distinctive properties are the Cartesian ``z``
parametrisation (a single ``Euclidean`` parameter, ``r = ‖z‖``), exactness at the
large radii where the ambient charts fail, and a finite gradient at the origin.
"""

import networkx as nx
import numpy as np
import pytest
import torch

torch.set_default_dtype(torch.float64)

import geoopt  # noqa: E402

from hypegrl.representations import (  # noqa: E402
    PolarRepresentation,
    TangentRepresentation,
)


def _unit(rng, n, d):
    V = rng.normal(size=(n, d))
    return V / np.linalg.norm(V, axis=1, keepdims=True)


@pytest.mark.parametrize("d", [2, 3])
def test_from_polar_to_polar_roundtrip(d):
    rng = np.random.default_rng(0)
    r = rng.uniform(0.1, 6.0, size=7)
    V = _unit(rng, 7, d)
    r2, v2 = TangentRepresentation.from_polar(r, V).to_polar()
    assert np.allclose(r2.numpy(), r, atol=1e-12)
    assert np.allclose(v2.numpy(), V, atol=1e-12)


def test_parameter_is_a_single_euclidean_z_equal_to_r_times_v():
    """``z`` is one ``Euclidean`` parameter of shape ``(N, d)`` with ``z = r·v``,
    so ``‖z‖ = r`` — not the ``Euclidean × Sphere`` pair of the polar chart."""
    r = np.array([1.0, 2.0, 5.0])
    V = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    rep = TangentRepresentation.from_polar(r, V)
    params = rep.parameters()
    assert len(params) == 1
    assert isinstance(params[0].manifold, geoopt.Euclidean)
    assert params[0].shape == (3, 2)
    z = params[0].detach().numpy()
    assert np.allclose(z, r[:, None] * V)                 # z = r·v
    assert np.allclose(np.linalg.norm(z, axis=1), r)      # ‖z‖ = r


def test_distance_matches_the_polar_chart_at_large_radius():
    """Same chart, same geometry — and unlike the ball/hyperboloid it stays exact
    at the large radii real graphs need (here up to r ≈ 40)."""
    rng = np.random.default_rng(3)
    r = rng.uniform(0.1, 40.0, size=10)
    V = _unit(rng, 10, 2)
    tan = TangentRepresentation.from_polar(r, V).dist().detach().numpy()
    pol = PolarRepresentation.from_polar(r, V).dist().detach().numpy()
    # relative agreement (distances themselves reach ~80 at these radii)
    assert np.allclose(tan, pol, rtol=1e-9, atol=1e-6)


def test_dist_finite_with_node_at_origin():
    """A node at ``z = 0`` (``r = 0``) must not produce a NaN distance/gradient —
    the ``1e-30`` floor inside ``r = ‖z‖`` guards the ``0/0`` in ``∂r/∂z``."""
    r = np.array([0.0, 1.0, 3.0])
    V = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    rep = TangentRepresentation.from_polar(r, V)
    D = rep.dist()
    assert torch.isfinite(D).all()
    D.sum().backward()
    assert torch.isfinite(rep.parameters()[0].grad).all()


def test_euclidean_adam_optimises_and_stays_finite():
    """End-to-end: plain Euclidean RiemannianAdam on ``z`` drives the loss down,
    exercising the ``r = ‖z‖`` / ``v = z/‖z‖`` autograd path at large radius."""
    rng = np.random.default_rng(5)
    r = rng.uniform(1.0, 3.0, size=6)
    V = _unit(rng, 6, 2)
    rep = TangentRepresentation.from_polar(r, V)
    opt = geoopt.optim.RiemannianAdam(rep.parameters(), lr=1e-2)

    def loss_fn():
        rr, _ = rep._polar()
        return ((rr - 5.0) ** 2).sum()           # pull all points out to r = 5

    first = loss_fn().item()
    for _ in range(500):
        opt.zero_grad()
        loss_fn().backward()
        opt.step()
    assert loss_fn().item() < first * 0.5
    assert torch.isfinite(rep.parameters()[0]).all()


def test_embedder_accepts_the_tangent_chart():
    """Selectable as ``representation="tangent"`` on a gradient embedder, and fits."""
    from hypegrl.embedders import PoincareMapsEmbedder

    G = nx.balanced_tree(2, 3)
    emb = PoincareMapsEmbedder(d=2, representation="tangent", n_steps=60,
                               lr_X=3e-2, random_state=0, log_every=0).fit(G)
    X = emb.embeddings()
    assert X.shape == (G.number_of_nodes(), 2)
    assert np.isfinite(X).all()
    assert emb._loss_history[-1] < emb._loss_history[0]
