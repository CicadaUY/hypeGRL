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
    ExactPolarRepresentation,
    HyperboloidRepresentation,
    PolarRepresentation,
    TangentRepresentation,
)

torch.set_default_dtype(torch.float64)

ALL_REPS = [PolarRepresentation, BallRepresentation, HyperboloidRepresentation]
DIST_BETWEEN_REPS = [
    PolarRepresentation, BallRepresentation, HyperboloidRepresentation,
    ExactPolarRepresentation, TangentRepresentation,
]


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


# ---------------------------------------------------------------------------
# Origin guard: a node at the centre (r = 0) has no defined direction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rep_cls", ALL_REPS)
def test_dist_finite_with_node_at_origin(rep_cls):
    """A node at the origin (``r = 0``) has an undefined direction; ``dist()``
    must stay finite. Regression: HyperMap places its root at the centre, whose
    zero direction made ``Sphere.projx`` return NaN and poisoned the whole
    distance matrix. The direction is irrelevant there (``sinh 0 = 0``), so the
    conversions assign the canonical ``e_0``.
    """
    X = np.array([
        [0.0, 0.0, 0.0],      # origin — direction undefined
        [0.30, 0.0, 0.0],
        [0.0, -0.40, 0.10],
        [0.20, 0.20, 0.20],
    ])
    rep = rep_cls.from_ball(X)
    D = rep.dist().detach().numpy()
    assert np.isfinite(D).all(), \
        f"{rep_cls.__name__}.dist() not finite with a node at the origin"
    # the origin's readout direction is a genuine unit vector (not zero/NaN)
    r, v = rep.to_polar()
    assert torch.isfinite(v).all()
    assert torch.isclose(v.norm(dim=1), torch.ones(len(X))).all()


# ---------------------------------------------------------------------------
# The polar chart's radial reparametrisation r = softplus(u)
# ---------------------------------------------------------------------------

def _one_radial_step(rep_cls, r0, lr):
    """One RiemannianAdam step on an outward-pulling, purely radial objective.

    Two antipodal nodes at radius ``r0`` are exactly ``2·r0`` apart, so a target
    on their distance moves the radii and nothing else. Returns ``Δr``.
    """
    v = np.array([[1.0, 0.0], [-1.0, 0.0]])
    rep = rep_cls.from_polar(np.array([r0, r0]), v)
    opt = geoopt.optim.RiemannianAdam(rep.parameters(), lr=lr, stabilize=10)
    opt.zero_grad()
    ((rep.dist()[0, 1] - (2.0 * r0 + 10.0)) ** 2).backward()
    opt.step()
    return float(rep.to_polar()[0][0]) - r0


@pytest.mark.parametrize("r0", [0.01, 0.1, 0.5, 1.5, 3.0, 6.0])
def test_polar_radial_step_is_tapered_by_the_softplus(r0):
    """``r = softplus(u)`` makes the radial step ``lr·(1 − e^{−r})``, not ``lr``.

    RiemannianAdam normalises the step in the *parameter* ``u`` to ``≈ lr``, so
    the chain rule ``dr/du = σ(u) = 1 − e^{−r}`` sets the radial displacement.
    The taper is deliberate (it keeps the optimiser from stepping across the
    origin, where the direction is undefined); this pins its exact size.
    """
    lr = 1e-2
    expected = lr * (1.0 - np.exp(-r0))
    got = _one_radial_step(PolarRepresentation, r0, lr)
    assert got == pytest.approx(expected, rel=0.02), (
        f"r={r0}: radial step {got:.4e}, expected lr·(1−e^-r) = {expected:.4e}")


def test_polar_radius_stays_positive_under_sustained_inward_pressure():
    """The origin is asymptotic: no radius reaches 0, so none loses its gradient.

    A radius parametrised directly and clamped at 0 would be pinned there for the
    rest of the run once a step crossed it; the softplus makes that unreachable,
    so an inward-driven node can still be driven back out.
    """
    v = np.array([[1.0, 0.0], [-1.0, 0.0]])
    rep = PolarRepresentation.from_polar(np.array([2.0, 2.0]), v)
    opt = geoopt.optim.RiemannianAdam(rep.parameters(), lr=0.1, stabilize=10)

    def step(target):
        opt.zero_grad()
        ((rep.dist()[0, 1] - target) ** 2).backward()
        opt.step()
        return float(rep.to_polar()[0][0])

    for _ in range(400):                      # collapse toward the origin
        r = step(0.0)
        assert r > 0.0, "radius reached the origin, where the gradient dies"
    # a large collapse, but asymptotic: the taper slows the last stretch down,
    # which is precisely the mechanism keeping the origin out of reach.
    assert r < 0.1, f"inward pressure did not take the node inward (r={r:.3f})"

    for _ in range(400):                      # and back out again
        r_out = step(8.0)
    assert r_out > 1.0, (
        f"node could not be driven back out from r={r:.2e} (reached {r_out:.3f})")


# ---------------------------------------------------------------------------
# O(d) equivariance: rotating a configuration is a no-op for the model
# ---------------------------------------------------------------------------

def _random_orthogonal(d, seed):
    """A random O(d) matrix, via QR of a Gaussian (no scipy dependency)."""
    Q, R = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, d)))
    return Q * np.sign(np.diag(R))            # fix QR's sign ambiguity


def _fit_toy(rep_cls, r, V, target, rot=None, lr=1e-2, n_steps=150):
    """Pull all pairwise distances toward ``target``, from a (rotated) start."""
    if rot is not None:
        V = V @ rot.T
    rep = rep_cls.from_polar(r, V)
    opt = geoopt.optim.RiemannianAdam(rep.parameters(), lr=lr, stabilize=10)
    T = torch.tensor(target)
    mask = ~torch.eye(len(r), dtype=torch.bool)
    for _ in range(n_steps):
        opt.zero_grad()
        loss = ((rep.dist()[mask] - T[mask]) ** 2).mean()
        loss.backward()
        opt.step()
    return loss.item()


@pytest.mark.parametrize("rep_cls", ALL_REPS + [ExactPolarRepresentation])
def test_optimisation_is_rotation_equivariant(rep_cls):
    """Rotating the whole configuration must not change the loss trajectory.

    Every loss in the library is a function of the pairwise distances alone, so
    ``(r, v) -> (r, v Rᵀ)`` for ``R ∈ O(d)`` is the *same problem*. These charts
    stay equivariant because ``RiemannianAdam`` accumulates its second moment
    through ``component_inner``, which for their manifolds is one scalar per
    point — a uniform rescaling, which commutes with ``R``. A per-coordinate
    second moment would not (see
    :class:`~hypegrl.representations.tangent.TangentRepresentation`), so this
    guards against a ``geoopt`` change that silently made these charts behave
    like ``Euclidean``.
    """
    r, V = _random_polar(30, 4, 0.5, 3.0, seed=11)
    rng = np.random.default_rng(12)
    target = np.triu(rng.uniform(0.5, 4.0, size=(30, 30)), 1)
    target = target + target.T

    base = _fit_toy(rep_cls, r, V, target)
    for seed in (1, 2, 3):
        rotated = _fit_toy(rep_cls, r, V, target, rot=_random_orthogonal(4, seed))
        assert rotated == pytest.approx(base, rel=1e-9, abs=1e-12), (
            f"{rep_cls.__name__}: rotating the configuration changed the loss "
            f"{base:.12f} -> {rotated:.12f}")


# ---------------------------------------------------------------------------
# dist_between: the gather-based sibling of dist(), for negative-sampling
# losses that must avoid materialising the full (N, N) matrix.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rep_cls", DIST_BETWEEN_REPS)
def test_dist_between_matches_dist_flat_pairs(rep_cls):
    """dist_between(i, j) with 1-D i/j must equal dist()[i, j] elementwise."""
    r, V = _random_polar(12, 3, 0.5, 6.0, seed=2)
    rep = rep_cls.from_polar(r, V)
    D = rep.dist().detach()

    i_idx = torch.tensor([0, 1, 2, 5, 7])
    j_idx = torch.tensor([1, 3, 4, 6, 0])
    got = rep.dist_between(i_idx, j_idx).detach()
    ref = D[i_idx, j_idx]
    assert torch.allclose(got, ref, atol=1e-8), f"{rep_cls.__name__} disagrees"


@pytest.mark.parametrize("rep_cls", DIST_BETWEEN_REPS)
def test_dist_between_matches_dist_broadcast_negatives(rep_cls):
    """dist_between((P,1), (P,K)) -- the negative-sampling call shape -- must
    equal dist()[i_idx, j_idx]'s broadcast-indexed result."""
    r, V = _random_polar(15, 3, 0.5, 6.0, seed=4)
    rep = rep_cls.from_polar(r, V)
    D = rep.dist().detach()

    i_idx = torch.tensor([0, 2, 5, 9]).unsqueeze(1)          # (P, 1)
    j_idx = torch.tensor([[1, 3, 4], [0, 1, 6], [1, 2, 3], [0, 4, 8]])  # (P, K)
    got = rep.dist_between(i_idx, j_idx).detach()
    ref = D[i_idx, j_idx]
    assert got.shape == (4, 3)
    assert torch.allclose(got, ref, atol=1e-8), f"{rep_cls.__name__} disagrees"


@pytest.mark.parametrize("rep_cls", DIST_BETWEEN_REPS)
def test_dist_between_gradient_flows_to_parameters(rep_cls):
    """Gradients through dist_between must reach every optimisable parameter,
    same as through dist() -- it is the accessor the ranking loss decodes."""
    r, V = _random_polar(10, 3, 0.5, 4.0, seed=6)
    rep = rep_cls.from_polar(r, V)
    i_idx = torch.tensor([0, 1, 2])
    j_idx = torch.tensor([3, 4, 5])
    rep.dist_between(i_idx, j_idx).sum().backward()
    for p in rep.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


@pytest.mark.parametrize("rep_cls", DIST_BETWEEN_REPS)
def test_dist_between_does_not_allocate_full_matrix(rep_cls):
    """Regression guard: dist_between must not internally call dist() (which
    would defeat the whole point -- materialising the (N, N) matrix it exists
    to avoid). Patches dist() on the instance to raise if invoked."""
    r, V = _random_polar(8, 3, 0.5, 4.0, seed=9)
    rep = rep_cls.from_polar(r, V)

    def _forbidden():
        raise AssertionError("dist_between must not call the full dist()")

    rep.dist = _forbidden
    i_idx = torch.tensor([0, 1])
    j_idx = torch.tensor([2, 3])
    out = rep.dist_between(i_idx, j_idx)
    assert torch.isfinite(out).all()
