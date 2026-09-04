# -*- coding: utf-8 -*-
"""
The curved polar chart: the warped metric ``dr² + w_c(r)²·g_{S^D}`` with
``w_c(r) = sinh(√c·r)/√c``, and the representation built on it.

Two independent references are used, so no test checks the implementation
against its own reasoning:

- the geodesic is compared with one obtained by numerically integrating the
  warped-product geodesic equations ``r'' = w_c w_c' θ'²``,
  ``θ'' = −2 (w_c'/w_c) r' θ'`` with RK4;
- ``chart_curvature = 1`` is compared with the already-verified exact chart.

The distinction the chart rests on is also pinned here: ``chart_curvature``
changes the metric the optimiser steps under and must leave the distance the
loss decodes — the curvature ``−1`` one — untouched.
"""

import networkx as nx
import numpy as np
import pytest
import torch

from hypegrl.embedders.hydra_plus import HydraPlusEmbedder, _stress_loss_from_dist
from hypegrl.inference.riemannian_optimizer import riemannian_optimize
from hypegrl.manifolds.polar import WarpedPolarHyperboloid, polar_distances
from hypegrl.representations import (
    CurvedPolarRepresentation,
    ExactPolarRepresentation,
    PolarRepresentation,
    build_representation,
)

torch.set_default_dtype(torch.float64)

CURVATURES = [0.04, 0.25, 1.0, 2.0]


def _pack(r, v):
    return torch.cat([torch.as_tensor([r], dtype=torch.float64),
                      torch.as_tensor(v, dtype=torch.float64)])


def _frame(theta):
    """A unit direction on S¹ and a unit vector orthogonal to it."""
    v = torch.tensor([np.cos(theta), np.sin(theta)])
    e = torch.tensor([-np.sin(theta), np.cos(theta)])
    return v, e


def _warp(r, c):
    return np.sinh(np.sqrt(c) * r) / np.sqrt(c)


def _integrated_geodesic(r0, theta0, dr0, dtheta0, c, n=40000):
    """
    The geodesic of ``dr² + w_c(r)²dθ²`` at ``t = 1``, by RK4 on

        r''     = w_c(r)·w_c'(r)·θ'²
        θ''     = −2·(w_c'(r)/w_c(r))·r'·θ'

    Reference for :meth:`WarpedPolarHyperboloid.expmap`, derived from the metric
    rather than from the implementation's ``ρ = √c·r`` substitution.
    """
    sq = np.sqrt(c)

    def deriv(state):
        r, th, dr, dth = state
        w, dw = np.sinh(sq * r) / sq, np.cosh(sq * r)
        return np.array([dr, dth, w * dw * dth ** 2, -2.0 * dw / w * dr * dth])

    y = np.array([r0, theta0, dr0, dtheta0], dtype=float)
    h = 1.0 / n
    for _ in range(n):
        k1 = deriv(y)
        k2 = deriv(y + 0.5 * h * k1)
        k3 = deriv(y + 0.5 * h * k2)
        k4 = deriv(y + h * k3)
        y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y[0], y[1]


# ----------------------------------------------------------------- geometry


@pytest.mark.parametrize("c", [0.04, 0.25, 2.0])
@pytest.mark.parametrize("r0", [0.7, 2.0, 5.0])
def test_expmap_matches_integrated_geodesic(c, r0):
    """The exponential map is the geodesic of the metric it advertises."""
    theta0, dr0 = 0.4, 0.03
    dtheta0 = 0.02 / _warp(r0, c)          # a realistic geodesic step length
    manifold = WarpedPolarHyperboloid(chart_curvature=c)

    v0, e = _frame(theta0)
    y = manifold.expmap(_pack(r0, v0), torch.cat([torch.tensor([dr0]), e * dtheta0]))
    r_ref, theta_ref = _integrated_geodesic(r0, theta0, dr0, dtheta0, c)

    assert y[0].item() == pytest.approx(r_ref, abs=1e-9)
    v_ref = torch.tensor([np.cos(theta_ref), np.sin(theta_ref)])
    assert torch.allclose(y[1:], v_ref, atol=1e-9)


@pytest.mark.parametrize("c", CURVATURES)
@pytest.mark.parametrize("r", [0.5, 3.0, 9.0])
def test_egrad2rgrad_divides_the_angular_part_by_the_warp_squared(c, r):
    """``G⁻¹ = diag(1, w_c⁻²·I)`` — the whole difference from a product metric."""
    manifold = WarpedPolarHyperboloid(chart_curvature=c)
    v, e = _frame(1.1)
    g = manifold.egrad2rgrad(_pack(r, v), torch.cat([torch.tensor([0.7]), e]))
    assert g[0].item() == pytest.approx(0.7, abs=1e-14)        # radial untouched
    assert torch.allclose(g[1:], e / _warp(r, c) ** 2, rtol=1e-12)


@pytest.mark.parametrize("c", CURVATURES)
def test_inner_is_the_warped_metric(c):
    """``⟨u,u⟩ = u_r² + w_c(r)²‖u_v‖²``, read off the metric directly."""
    manifold = WarpedPolarHyperboloid(chart_curvature=c)
    r = 2.5
    v, e = _frame(0.0)
    u = torch.cat([torch.tensor([0.3]), e * 0.2])
    expected = 0.3 ** 2 + _warp(r, c) ** 2 * 0.2 ** 2
    assert manifold.inner(_pack(r, v), u).item() == pytest.approx(expected, rel=1e-12)


def test_curvature_one_reproduces_the_exact_chart():
    """``chart_curvature = 1`` is the exact metric, not merely close to it."""
    exact = WarpedPolarHyperboloid()
    curved = WarpedPolarHyperboloid(chart_curvature=1.0)
    rng = np.random.default_rng(0)
    for r0 in (0.4, 2.0, 7.0, 15.0):
        v, e = _frame(rng.uniform(0, 2 * np.pi))
        x = _pack(r0, v)
        u = torch.cat([torch.tensor([rng.normal() * 1e-2]),
                       e * (rng.normal() * 1e-2 / np.sinh(r0))])
        assert torch.allclose(curved.expmap(x, u), exact.expmap(x, u), atol=1e-14)
        assert torch.allclose(curved.egrad2rgrad(x, u), exact.egrad2rgrad(x, u),
                              atol=1e-14)
        y = _pack(r0 + 0.3, _frame(0.2)[0])
        assert curved.dist(x, y).item() == pytest.approx(exact.dist(x, y).item(),
                                                         abs=1e-12)


@pytest.mark.parametrize("c", [0.04, 0.25, 2.0])
def test_manifold_dist_is_the_chart_metric_not_the_embedding_distance(c):
    """
    The manifold reports *its own* geodesic distance, so it stays consistent
    with :meth:`inner` and :meth:`expmap`.

    Since ``g_c = (1/c)·g₁`` in ``ρ = √c·r``, that distance is
    ``d₁(ρ₀,ρ₁)/√c`` — a different function of ``(r, v)`` from the curvature
    ``−1`` distance the embedding is scored by, which is why the representation
    decodes the latter itself rather than delegating here.
    """
    manifold = WarpedPolarHyperboloid(chart_curvature=c)
    v0, _ = _frame(0.0)
    v1, _ = _frame(0.9)
    r0, r1 = 1.3, 2.6
    got = manifold.dist(_pack(r0, v0), _pack(r1, v1)).item()

    rho = np.array([np.sqrt(c) * r0, np.sqrt(c) * r1])
    unit = WarpedPolarHyperboloid()
    expected = unit.dist(_pack(rho[0], v0), _pack(rho[1], v1)).item() / np.sqrt(c)
    assert got == pytest.approx(expected, rel=1e-12)

    embedding_distance = polar_distances(np.array([r0, r1]),
                                         np.stack([v0.numpy(), v1.numpy()]))[0, 1]
    assert got != pytest.approx(embedding_distance, rel=1e-6)


@pytest.mark.parametrize("c", [0.04, 0.25, 1.0])
def test_a_metric_step_is_consistent_with_inner_and_dist(c):
    """A step of metric length ``s`` lands ``s`` away in the manifold's own metric."""
    manifold = WarpedPolarHyperboloid(chart_curvature=c)
    v, e = _frame(0.3)
    x = _pack(4.0, v)
    u = torch.cat([torch.tensor([0.02]), e * 0.01 / _warp(4.0, c)])
    s = torch.sqrt(manifold.inner(x, u)).item()
    assert manifold.dist(x, manifold.expmap(x, u)).item() == pytest.approx(s, rel=1e-9)


@pytest.mark.parametrize("r", [8.0, 15.0])
def test_a_smaller_chart_curvature_turns_the_angle_further(r):
    """
    The point of the chart: for the same geodesic step budget, a flatter warp
    buys more angular coordinate motion at large radius.

    A step of metric length ``s`` spent purely angularly changes the angle by
    ``s/w_c(r)``, and ``w_c`` grows with ``c``.

    The angle is read off the chord rather than from ``arccos`` of a dot
    product: at ``c = 1`` and ``r = 15`` the turn is ``~3e-8`` radians — the
    stall this chart exists to relieve — and ``arccos`` has no resolution left
    that close to 1.
    """
    step, angles = 0.05, []
    for c in (0.04, 0.25, 1.0):
        manifold = WarpedPolarHyperboloid(chart_curvature=c)
        v, e = _frame(0.0)
        y = manifold.expmap(_pack(r, v), torch.cat([torch.tensor([0.0]),
                                                    e * step / _warp(r, c)]))
        angle = 2.0 * torch.arcsin(0.5 * (y[1:] - v).norm()).item()
        assert angle == pytest.approx(step / _warp(r, c), rel=2e-2)
        angles.append(angle)
    assert angles[0] > angles[1] > angles[2]


@pytest.mark.parametrize("r0", [20.0, 40.0, 80.0])
def test_expmap_stable_at_large_radius(r0):
    """No ``e^r`` forms, so the chart survives radii the ambient ones cannot hold."""
    manifold = WarpedPolarHyperboloid(chart_curvature=0.25)
    y = manifold.expmap(_pack(r0, [1.0, 0.0]), torch.tensor([0.05, 0.0, 0.0]))
    assert torch.isfinite(y).all()
    assert y[0].item() == pytest.approx(r0 + 0.05, abs=1e-9)


def test_non_positive_chart_curvature_is_rejected():
    """``c → 0`` is the tangent chart, a different parametrisation, not a limit."""
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="chart_curvature must be positive"):
            WarpedPolarHyperboloid(chart_curvature=bad)
    with pytest.raises(ValueError, match="chart_curvature must be positive"):
        CurvedPolarRepresentation.from_polar(
            torch.tensor([1.0]), torch.tensor([[1.0, 0.0]]), chart_curvature=0.0)


# ----------------------------------------------------------- representation


def _sample(n=9, seed=0):
    rng = np.random.default_rng(seed)
    r = torch.as_tensor(rng.uniform(0.2, 6.0, size=n))
    v = torch.as_tensor(rng.normal(size=(n, 2)))
    return r, v / v.norm(dim=-1, keepdim=True)


@pytest.mark.parametrize("c", CURVATURES)
def test_distance_does_not_depend_on_chart_curvature(c):
    """
    The chart is a preconditioner: the objective must be identical at every
    ``c``, so two runs differing only in ``chart_curvature`` optimise the same
    function.
    """
    r, v = _sample()
    ref = ExactPolarRepresentation.from_polar(r, v).dist()
    got = CurvedPolarRepresentation.from_polar(r, v, chart_curvature=c).dist()
    assert torch.allclose(got, ref, atol=1e-12)


@pytest.mark.parametrize("c", [0.25, 1.0])
def test_dist_between_matches_the_full_matrix(c):
    rep = CurvedPolarRepresentation.from_polar(*_sample(), chart_curvature=c)
    i = torch.tensor([0, 3, 5])
    j = torch.tensor([[1, 2], [4, 6], [7, 8]])
    assert torch.allclose(rep.dist_between(i[:, None], j), rep.dist()[i[:, None], j],
                          atol=1e-12)


def test_polar_roundtrip_is_lossless():
    r, v = _sample()
    got_r, got_v = CurvedPolarRepresentation.from_polar(
        r, v, chart_curvature=0.25).to_polar()
    assert torch.allclose(got_r, r, atol=1e-12)
    assert torch.allclose(got_v, v, atol=1e-12)


def test_zero_diagonal():
    rep = CurvedPolarRepresentation.from_polar(*_sample())
    assert torch.equal(rep.dist().diagonal(), torch.zeros(9, dtype=torch.float64))


def test_default_chart_curvature_is_a_quarter():
    """
    The member at which the arc formula's angular factor ``sinh(r)/w_c(r)²``
    stops depending on radius, in the crowded regime where a pairwise-distance
    loss has ``∂_θ d ≈ sinh r``.
    """
    rep = CurvedPolarRepresentation.from_polar(*_sample())
    assert rep._manifold.chart_curvature == 0.25


@pytest.mark.parametrize("r", [5.0, 10.0, 20.0, 40.0])
def test_the_angular_arc_factor_is_flat_in_radius_at_a_quarter(r):
    """
    ``sinh(r)/w_c(r)² = ½·coth(r/2) → ½`` at ``c = 1/4``, while it blows up
    below and decays above — the reason that value is the default.
    """
    assert np.sinh(r) / _warp(r, 0.25) ** 2 == pytest.approx(0.5, abs=1e-2)
    assert np.sinh(r) / _warp(r, 0.04) ** 2 > 1.0                    # grows
    assert np.sinh(r) / _warp(r, 1.0) ** 2 < 0.5                     # decays


@pytest.mark.parametrize("c", [0.04, 0.25, 1.0])
def test_optimisation_reduces_stress(c):
    """The chart trains: a stress fit on a small tree improves and stays finite."""
    G = nx.balanced_tree(2, 3)
    n = G.number_of_nodes()
    D = np.array(nx.floyd_warshall_numpy(G), dtype=np.float64)
    mask = torch.as_tensor(np.triu(np.ones((n, n), dtype=bool), k=1))
    rng = np.random.default_rng(0)
    r = torch.as_tensor(rng.uniform(0.5, 3.0, size=n))
    v = torch.as_tensor(rng.normal(size=(n, 2)))
    rep = CurvedPolarRepresentation.from_polar(
        r, v / v.norm(dim=-1, keepdim=True), chart_curvature=c)

    history = riemannian_optimize(
        representation=rep, s_A=D,
        loss_fn=lambda rep_, s_: _stress_loss_from_dist(rep_.dist(), s_, mask),
        lr=1e-2, n_steps=200, log_every=0,
    )["loss_history"]

    assert np.isfinite(history).all()
    assert history[-1] < history[0]
    assert torch.isfinite(rep.dist()).all()


# --------------------------------------------------------------- plumbing


def test_representation_kwargs_reach_the_chart():
    rep = build_representation(
        CurvedPolarRepresentation, np.array([[0.1, 0.2], [0.3, -0.1]]),
        input_chart="ball", chart_curvature=0.09)
    assert rep._manifold.chart_curvature == pytest.approx(0.09)


def test_an_option_the_selected_chart_ignores_is_rejected():
    """A chart option is silently dropped by ``**_`` otherwise — loud is better."""
    with pytest.raises(TypeError, match="chart_curvature"):
        build_representation(PolarRepresentation, np.array([[0.1, 0.2]]),
                             input_chart="ball", chart_curvature=0.25)


def test_embedder_accepts_the_curved_chart():
    """End to end through an embedder, which is how the chart is selected."""
    G = nx.balanced_tree(2, 3)
    emb = HydraPlusEmbedder(
        dim=2, n_steps=30, random_state=0, representation="curved_polar",
        representation_kwargs={"chart_curvature": 0.25},
    ).fit(G)
    X = emb.embeddings()
    assert X.shape == (G.number_of_nodes(), 2)
    assert np.isfinite(X).all()
    assert emb.embeddings_representation()._manifold.chart_curvature == 0.25


def test_embedder_rejects_an_option_for_another_chart():
    with pytest.raises(TypeError, match="chart_curvature"):
        HydraPlusEmbedder(
            dim=2, n_steps=5, representation="polar",
            representation_kwargs={"chart_curvature": 0.25},
        ).fit(nx.balanced_tree(2, 2))
