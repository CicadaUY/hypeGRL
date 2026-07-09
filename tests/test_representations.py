# -*- coding: utf-8 -*-
"""
Stage-B Representation layer: the three charts round-trip through canonical polar
coordinates, agree on pairwise distance (they are the same geometry), ingest from
any chart, and expose optimisable parameters.

Radii are kept moderate (r ≤ 6) so all three charts are numerically valid — the
large-radius divergence between charts is the subject of the Stage-D comparison,
not these correctness tests.
"""

import geoopt
import numpy as np
import pytest
import torch

from hypegrl.manifolds.polar import polar_distances
from hypegrl.representations import (
    BallRepresentation,
    HyperboloidRepresentation,
    PolarRepresentation,
)

torch.set_default_dtype(torch.float64)

ALL_REPS = [PolarRepresentation, BallRepresentation, HyperboloidRepresentation]


def _random_polar(n, d, r_lo, r_hi, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.uniform(r_lo, r_hi, size=n)
    V = rng.standard_normal((n, d))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return r, V


# ---------------------------------------------------------------------------
# Round-trip: from_polar(r, v) -> to_polar() recovers (r, v)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rep_cls", ALL_REPS)
@pytest.mark.parametrize("d", [2, 3])
def test_from_polar_to_polar_roundtrip(rep_cls, d):
    r, V = _random_polar(60, d, 0.5, 6.0, seed=d)
    rep = rep_cls.from_polar(r, V)
    r2, V2 = rep.to_polar()
    r2, V2 = r2.numpy(), V2.numpy()
    assert np.allclose(r, r2, atol=1e-8)
    # v is a direction: compare up to sign-free unit-vector equality
    assert np.allclose(np.abs((V * V2).sum(1)), 1.0, atol=1e-8)


# ---------------------------------------------------------------------------
# All three charts are the same geometry: dist() agrees at moderate radius
# ---------------------------------------------------------------------------

def test_all_reps_agree_on_distance():
    # Compare off-diagonal only: self-distance is 0 by definition, and the
    # charts differ there by convention (geoopt's guarded arccosh returns
    # sqrt(1e-15) ~ 3e-8 on the diagonal, polar/ball return ~0).
    r, V = _random_polar(40, 3, 0.5, 6.0, seed=7)
    ref = polar_distances(r, V)  # oracle-checked in test_charts.py
    mask = ~np.eye(len(r), dtype=bool)
    for rep_cls in ALL_REPS:
        D = rep_cls.from_polar(r, V).dist().detach().numpy()
        assert np.allclose(D[mask], ref[mask], atol=1e-8), \
            f"{rep_cls.__name__} disagrees"


# ---------------------------------------------------------------------------
# Ingestion from other charts (base composition via converters)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rep_cls", ALL_REPS)
def test_from_ball_and_from_hyperboloid(rep_cls):
    from hypegrl.manifolds.conversions import polar_to_ball, polar_to_hyperboloid

    r, V = _random_polar(40, 3, 0.5, 6.0, seed=3)
    ref = polar_distances(r, V)
    mask = ~np.eye(len(r), dtype=bool)
    for build in (
        lambda: rep_cls.from_ball(polar_to_ball(r, V)),
        lambda: rep_cls.from_hyperboloid(polar_to_hyperboloid(r, V)),
    ):
        D = build().dist().detach().numpy()
        assert np.allclose(D[mask], ref[mask], atol=1e-7)


# ---------------------------------------------------------------------------
# to_ball / to_hyperboloid readouts (base composition)
# ---------------------------------------------------------------------------

def test_readouts_to_other_charts():
    from hypegrl.manifolds.conversions import polar_to_ball, polar_to_hyperboloid

    r, V = _random_polar(30, 3, 0.5, 6.0, seed=11)
    rep = PolarRepresentation.from_polar(r, V)
    assert np.allclose(rep.to_ball().numpy(), polar_to_ball(r, V), atol=1e-8)
    assert np.allclose(rep.to_hyperboloid().numpy(),
                       polar_to_hyperboloid(r, V), atol=1e-8)


# ---------------------------------------------------------------------------
# parameters() are optimisable (a few inline RiemannianAdam steps, no driver)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rep_cls", ALL_REPS)
def test_parameters_are_optimisable(rep_cls):
    # toy target: pull all pairwise distances toward 0.5 (a well-posed, smooth
    # objective). Just check the loss drops and gradients stay finite.
    r, V = _random_polar(20, 3, 0.5, 4.0, seed=5)
    rep = rep_cls.from_polar(r, V)
    params = rep.parameters()
    opt = geoopt.optim.RiemannianAdam(params, lr=1e-2, stabilize=5)

    def loss_fn():
        D = rep.dist()
        mask = ~torch.eye(D.shape[0], dtype=torch.bool)
        return ((D[mask] - 0.5) ** 2).mean()

    first = loss_fn().item()
    for _ in range(50):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        for p in params:
            assert torch.isfinite(p.grad).all()
        opt.step()
    last = loss_fn().item()
    assert last < first, f"{rep_cls.__name__}: loss {first:.4f} -> {last:.4f}"
