# -*- coding: utf-8 -*-
"""
Stage-A chart layer: polar canonical coordinate, direct converters, and the
stable polar distance, checked against a high-precision (mpmath) oracle.

These guard the numerical claim behind the Representation refactor: the polar
chart computes hyperbolic distances correctly at the large radii (r up to ~40)
where the ambient hyperboloid / ball charts fail.
"""

import mpmath as mp
import numpy as np
import pytest
import torch

from hypegrl.manifolds.conversions import (
    ball_to_hyperboloid,
    ball_to_polar,
    hyperboloid_to_ball,
    hyperboloid_to_polar,
    polar_to_ball,
    polar_to_hyperboloid,
)
from hypegrl.manifolds.polar import polar_distances, polar_distances_torch

torch.set_default_dtype(torch.float64)
mp.mp.dps = 50  # 50 significant digits for the oracle


def _random_polar(n, d, r_lo, r_hi, seed=0):
    """n points in polar coords: radii in [r_lo, r_hi], unit dirs in S^{d-1}."""
    rng = np.random.default_rng(seed)
    r = rng.uniform(r_lo, r_hi, size=n)
    V = rng.standard_normal((n, d))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return r, V


def _oracle_distance(ri, rj, vi, vj):
    """
    Exact H^{D+1} geodesic distance at 50 digits (law of cosines).

    The angular term is computed as an mpmath dot of the vector components —
    NOT ``float(np.dot(vi, vj))`` — so ``1 − cos`` keeps full precision for
    near-coincident directions (a float64 dot rounds it to garbage).
    """
    cos_ang = sum(mp.mpf(float(a)) * mp.mpf(float(b)) for a, b in zip(vi, vj))
    ri, rj = mp.mpf(float(ri)), mp.mpf(float(rj))
    val = mp.cosh(ri) * mp.cosh(rj) - mp.sinh(ri) * mp.sinh(rj) * cos_ang
    return float(mp.acosh(val if val > 1 else mp.mpf(1)))


# ---------------------------------------------------------------------------
# Round-trips (each within its chart's valid r-range)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("d", [2, 3, 5])
def test_polar_ball_roundtrip(d):
    # ball saturates past r~12, so round-trip only in the ball-safe range.
    r, V = _random_polar(200, d, 0.5, 10.0, seed=d)
    r2, V2 = ball_to_polar(polar_to_ball(r, V))
    assert np.allclose(r, r2, atol=1e-9)
    assert np.allclose(V, V2, atol=1e-9)


@pytest.mark.parametrize("d", [2, 3, 5])
def test_polar_hyperboloid_roundtrip(d):
    # hyperboloid is exact to r~350; test well past the ball's ceiling.
    r, V = _random_polar(200, d, 0.5, 300.0, seed=d)
    H = polar_to_hyperboloid(r, V)
    # on-sheet constraint x0^2 - ||x'||^2 = 1, relative to the coordinate scale
    # cosh^2 r (the residual's rounding grain grows with the coordinates).
    resid = H[:, 0] ** 2 - (H[:, 1:] ** 2).sum(1) - 1.0
    assert (np.abs(resid) / np.cosh(r) ** 2).max() < 1e-10
    r2, V2 = hyperboloid_to_polar(H)
    assert np.allclose(r, r2, atol=1e-9)
    assert np.allclose(V, V2, atol=1e-9)


@pytest.mark.parametrize("d", [2, 3])
def test_ball_hyperboloid_consistency(d):
    # ball -> hyperboloid -> polar agrees with ball -> polar (moderate r).
    r, V = _random_polar(200, d, 0.5, 8.0, seed=d)
    X = polar_to_ball(r, V)
    r_via_h, V_via_h = hyperboloid_to_polar(ball_to_hyperboloid(X))
    r_direct, V_direct = ball_to_polar(X)
    assert np.allclose(r_via_h, r_direct, atol=1e-8)
    assert np.allclose(V_via_h, V_direct, atol=1e-8)
    assert np.allclose(hyperboloid_to_ball(ball_to_hyperboloid(X)), X, atol=1e-8)


# ---------------------------------------------------------------------------
# polar_distances vs the mpmath oracle
# ---------------------------------------------------------------------------

def test_polar_distances_matches_oracle_large_radius():
    # random well-separated points out to r=40, where the ambient chart fails.
    r, V = _random_polar(40, 3, 5.0, 40.0, seed=1)
    D = polar_distances(r, V)
    err = 0.0
    for i in range(len(r)):
        for j in range(i + 1, len(r)):
            o = _oracle_distance(r[i], r[j], V[i], V[j])
            err = max(err, abs(D[i, j] - o) / max(1.0, o))
    assert err < 1e-10, f"max relative error {err:.2e}"


def test_polar_distances_near_coincident_at_large_radius():
    # Same-angle-*ish* pairs at r=40, in the resolvable regime. The chord form
    # keeps the angular contribution where the (1 - <v,v>) form would lose it.
    #
    # Fundamental float64 limit (verified against the oracle, not a bug): at
    # radius r the angular term m ~ sinh^2(r)*sin^2(dtheta/2) must stand above
    # the ~eps*sinh^2(r) noise from v not being *exactly* unit-norm, i.e.
    # dtheta must exceed ~sqrt(eps) ~ 1e-8 (independent of r) AND, at large r,
    # comfortably more. So we test dtheta >= 1e-4 at r=40; below that the
    # geometry itself is not float64-resolvable in any chart.
    for dtheta in [1e-2, 1e-3, 1e-4]:
        vi = np.array([np.cos(0.7), np.sin(0.7)])
        vj = np.array([np.cos(0.7 + dtheta), np.sin(0.7 + dtheta)])
        r = np.array([40.0, 40.0])
        d = polar_distances(r, np.stack([vi, vj]))[0, 1]
        o = _oracle_distance(40.0, 40.0, vi, vj)
        assert abs(d - o) / max(1.0, o) < 1e-6, f"dtheta={dtheta}: {d} vs {o}"


def test_polar_distance_same_angle_is_delta_r():
    # the exact pairs that broke geoopt.Lorentz.dist: same angle, radii differ
    # by 1 -> true distance 1.0 at every radius (the fix regression guard).
    v = np.array([np.cos(0.7), np.sin(0.7)])
    for ri, rj in [(5, 6), (18, 19), (19, 20), (25, 26), (40, 41)]:
        d = polar_distances(np.array([float(ri), float(rj)]), np.stack([v, v]))[0, 1]
        assert abs(d - 1.0) < 1e-9, f"({ri},{rj}): {d}"


# ---------------------------------------------------------------------------
# torch: agreement with numpy + autograd
# ---------------------------------------------------------------------------

def test_polar_distances_torch_matches_numpy_and_is_differentiable():
    r_np, V_np = _random_polar(30, 3, 5.0, 40.0, seed=2)
    r = torch.tensor(r_np, requires_grad=True)
    V = torch.tensor(V_np, requires_grad=True)
    D = polar_distances_torch(r, V)
    assert np.allclose(D.detach().numpy(), polar_distances(r_np, V_np), atol=1e-10)
    D.sum().backward()
    assert torch.isfinite(r.grad).all() and torch.isfinite(V.grad).all()
