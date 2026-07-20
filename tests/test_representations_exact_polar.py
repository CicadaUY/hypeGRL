# -*- coding: utf-8 -*-
"""
The exact warped-polar manifold: metric ``dr² + sinh²(r)·g_{S^D}``, its natural
gradient, and its geodesic.

The reference for every geometric claim is the hyperboloid model, computed here
independently of the implementation: a polar point ``(r, v)`` pushes forward to
``Φ(r,v) = (cosh r, sinh r·v)``, where the geodesic and the metric are textbook.
Those reference computations are only valid at moderate radius (``cosh r``
overflows and cancels catastrophically past ``r ≈ 18``), so the exactness tests
stay there; large-radius behaviour is tested for stability instead.
"""

import numpy as np
import pytest
import torch

from hypegrl.manifolds.polar import WarpedPolarHyperboloid, polar_distances
from hypegrl.representations import ExactPolarRepresentation

torch.set_default_dtype(torch.float64)

MANIFOLD = WarpedPolarHyperboloid()


def _pack(r, v):
    return torch.cat([torch.as_tensor([r], dtype=torch.float64),
                      torch.as_tensor(v, dtype=torch.float64)])


def _frame(theta):
    """A unit direction on S^1 and a unit vector orthogonal to it."""
    v = torch.tensor([np.cos(theta), np.sin(theta)])
    e = torch.tensor([-np.sin(theta), np.cos(theta)])
    return v, e


def _reference_expmap(r0, v0, u_r, u_v):
    """Geodesic computed on the hyperboloid, independently of the manifold code."""
    X0 = torch.cat([torch.cosh(r0), torch.sinh(r0) * v0])
    xi = torch.cat([u_r * torch.sinh(r0),
                    u_r * torch.cosh(r0) * v0 + torch.sinh(r0) * u_v])
    s = torch.sqrt(u_r ** 2 + (torch.sinh(r0) * u_v.norm()) ** 2)   # = ‖xi‖_L = ‖u‖_g
    g = torch.cosh(s) * X0 + torch.sinh(s) * xi / s
    return torch.arccosh(g[0]), g[1:] / g[1:].norm()


# ---------------------------------------------------------------- geodesic


@pytest.mark.parametrize("r0", [0.3, 1.0, 2.0, 5.0, 8.0])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_expmap_matches_hyperboloid_geodesic(r0, seed):
    """The exponential map is the exact H^{D+1} geodesic.

    The tangent's angular part is scaled by ``1/sinh r0`` so the step has a
    realistic *geodesic* length (~1e-2) at every radius — which is what the
    natural gradient produces (see ``test_step_length_is_lr_at_every_radius``).
    """
    rng = np.random.default_rng(seed)
    r0_t = torch.tensor([r0])
    v0, e = _frame(rng.uniform(0, 2 * np.pi))
    u_r = torch.tensor([rng.normal() * 1e-2])
    u_v = e * (rng.normal() * 1e-2 / np.sinh(r0))

    r1_ref, v1_ref = _reference_expmap(r0_t, v0, u_r, u_v)
    y = MANIFOLD.expmap(torch.cat([r0_t, v0]), torch.cat([u_r, u_v]))

    assert y[0].item() == pytest.approx(r1_ref.item(), abs=1e-12)
    assert torch.allclose(y[1:], v1_ref, atol=1e-12)


def test_expmap_pure_radial_is_exact():
    """A purely radial tangent moves the radius by exactly ``u_r``, fixing ``v``."""
    y = MANIFOLD.expmap(_pack(3.0, [1.0, 0.0]), torch.tensor([0.25, 0.0, 0.0]))
    assert y[0].item() == pytest.approx(3.25, abs=1e-12)
    assert torch.allclose(y[1:], torch.tensor([1.0, 0.0]), atol=1e-14)


def test_expmap_through_origin_flips_direction():
    """A radial step past the origin comes out the antipodal side, ``r ≥ 0``."""
    y = MANIFOLD.expmap(_pack(0.5, [1.0, 0.0]), torch.tensor([-2.0, 0.0, 0.0]))
    assert y[0].item() == pytest.approx(1.5, abs=1e-12)
    assert torch.allclose(y[1:], torch.tensor([-1.0, 0.0]), atol=1e-12)


@pytest.mark.parametrize("r0", [20.0, 40.0, 80.0])
def test_expmap_stable_at_large_radius(r0):
    """No ``e^r`` forms, so the geodesic stays exact where ambient charts die.

    The hyperboloid chart cannot even represent these points usefully
    (``cosh 40 ≈ 1e17`` cancels to noise); here the radial step is still exact
    to 1e-9.
    """
    y = MANIFOLD.expmap(_pack(r0, [1.0, 0.0]), torch.tensor([0.05, 0.0, 0.0]))
    assert torch.isfinite(y).all()
    assert y[0].item() == pytest.approx(r0 + 0.05, abs=1e-9)


def test_expmap_keeps_point_on_non_finite_step():
    """An already-overflowed tangent leaves the point put (as StableLorentz does)."""
    x = _pack(2.0, [1.0, 0.0])
    y = MANIFOLD.expmap(x, torch.tensor([float("inf"), 0.0, 0.0]))
    assert torch.allclose(y, x)


def test_max_step_clamps_runaway_step():
    """A step of absurd geodesic length is capped, not allowed to overflow.

    At ``r=8`` an angular coordinate step of 0.07 is a geodesic displacement of
    ``sinh(8)·0.07 ≈ 107``; the clamp caps it at ``max_step``.
    """
    manifold = WarpedPolarHyperboloid(max_step=5.0)
    v0, e = _frame(0.0)
    x = torch.cat([torch.tensor([8.0]), v0])
    y = manifold.expmap(x, torch.cat([torch.tensor([0.0]), e * 0.07]))
    assert torch.isfinite(y).all()
    assert manifold.dist(x, y).item() == pytest.approx(5.0, rel=1e-6)


# ------------------------------------------------------------------ metric


@pytest.mark.parametrize("r0", [0.5, 2.0, 6.0])
def test_inner_matches_pushforward_minkowski(r0):
    """``⟨u,w⟩_x`` equals the Minkowski inner product of the pushed-forward pair."""
    rng = np.random.default_rng(0)
    r0_t = torch.tensor([r0])
    v0, e = _frame(rng.uniform(0, 2 * np.pi))
    x = torch.cat([r0_t, v0])

    def push(u_r, u_v):                       # dΦ(u), see the derivation §3
        return torch.cat([u_r * torch.sinh(r0_t),
                          u_r * torch.cosh(r0_t) * v0 + torch.sinh(r0_t) * u_v])

    def minkowski(a, b):
        return (-a[0] * b[0] + (a[1:] * b[1:]).sum()).item()

    u_r, w_r = torch.tensor([0.3]), torch.tensor([-0.7])
    u_v, w_v = e * 0.2, e * 0.5
    got = MANIFOLD.inner(x, torch.cat([u_r, u_v]), torch.cat([w_r, w_v])).item()
    assert got == pytest.approx(minkowski(push(u_r, u_v), push(w_r, w_v)), rel=1e-10)


@pytest.mark.parametrize("r0", [0.5, 2.0, 6.0, 20.0])
def test_egrad2rgrad_satisfies_its_defining_property(r0):
    """``⟨∇g L, w⟩_x = dL(w)`` for every tangent ``w`` — the definition of the
    Riemannian gradient."""
    rng = np.random.default_rng(1)
    v0, e = _frame(rng.uniform(0, 2 * np.pi))
    x = torch.cat([torch.tensor([r0]), v0])
    egrad = torch.tensor([rng.normal(), rng.normal(), rng.normal()])
    rgrad = MANIFOLD.egrad2rgrad(x, egrad)

    for _ in range(3):
        w = torch.cat([torch.tensor([rng.normal()]), e * rng.normal()])   # tangent
        dL_w = (egrad[0] * w[0] + (egrad[1:] * w[1:]).sum()).item()
        assert MANIFOLD.inner(x, rgrad, w).item() == pytest.approx(dL_w, rel=1e-9)


def test_egrad2rgrad_divides_angular_part_by_the_warp():
    """The angular gradient is scaled by ``1/sinh²r`` — the product metric's
    omission, and the whole point of this manifold."""
    v0, e = _frame(0.0)
    r0 = 3.0
    x = torch.cat([torch.tensor([r0]), v0])
    egrad = torch.cat([torch.tensor([1.0]), e * 1.0])
    rgrad = MANIFOLD.egrad2rgrad(x, egrad)

    assert rgrad[0].item() == pytest.approx(1.0)                    # radial untouched
    assert rgrad[1:].norm().item() == pytest.approx(1.0 / np.sinh(r0) ** 2, rel=1e-10)


def test_egrad2rgrad_is_finite_at_the_origin():
    """A node at ``r = 0`` (a tree root) must not produce ``NaN``: polar
    coordinates are genuinely singular there, so the warp is floored."""
    v0, e = _frame(0.0)
    x = torch.cat([torch.tensor([0.0]), v0])
    rgrad = MANIFOLD.egrad2rgrad(x, torch.cat([torch.tensor([1.0]), e * 1.0]))
    assert torch.isfinite(rgrad).all()


def test_proju_makes_angular_part_tangent():
    v0, _ = _frame(0.7)
    x = torch.cat([torch.tensor([2.0]), v0])
    u = MANIFOLD.proju(x, torch.tensor([0.5, 1.0, -2.0]))
    assert (u[1:] * v0).sum().item() == pytest.approx(0.0, abs=1e-14)
    assert u[0].item() == pytest.approx(0.5)          # radial part passes through


@pytest.mark.parametrize("r0", [1.0, 5.0, 15.0])
def test_step_length_is_lr_at_every_radius(r0):
    """The exact metric self-regulates the step; the product metric does not.

    Normalising the natural gradient by its **metric** norm (what
    ``RiemannianAdam``'s second moment does) gives a geodesic step of exactly
    ``lr`` at any radius. Doing the same under the product metric — no ``1/sinh²r``
    in the gradient, Euclidean normalisation — yields a geodesic step that grows
    like ``sinh r``: the documented large-radius overshoot.
    """
    lr = 1e-2
    v0, e = _frame(0.3)
    x = torch.cat([torch.tensor([r0]), v0])
    egrad = torch.cat([torch.tensor([1.0]), e * 1.0])

    rgrad = MANIFOLD.egrad2rgrad(x, egrad)
    step = rgrad * (lr / MANIFOLD.norm(x, rgrad).item())
    assert MANIFOLD.norm(x, step).item() == pytest.approx(lr, rel=1e-9)

    # product metric: gradient unscaled by the warp, direction normalised in ℝ^{D+2}
    step_prod = egrad * (lr / egrad.norm().item())
    geodesic_len = MANIFOLD.norm(x, step_prod).item()
    assert geodesic_len == pytest.approx(
        lr * np.sqrt(0.5 + 0.5 * np.sinh(r0) ** 2), rel=1e-9)
    if r0 >= 5.0:
        assert geodesic_len > 20 * lr        # already an order of magnitude adrift


def test_dist_matches_polar_distances():
    """The manifold's point distance agrees with the library's pairwise helper."""
    r = np.array([0.5, 4.0])
    V = np.array([[1.0, 0.0], [np.cos(1.1), np.sin(1.1)]])
    expected = polar_distances(r, V)[0, 1]
    got = MANIFOLD.dist(_pack(r[0], V[0]), _pack(r[1], V[1])).item()
    assert got == pytest.approx(expected, rel=1e-10)


def test_projx_normalises_direction_and_floors_radius():
    x = MANIFOLD.projx(torch.tensor([-1.0, 3.0, 4.0]))
    assert x[0].item() == 0.0
    assert x[1:].norm().item() == pytest.approx(1.0)


# ---------------------------------------------------- representation layer


def test_representation_roundtrips_through_polar():
    rng = np.random.default_rng(3)
    r = rng.uniform(0.1, 25.0, size=12)
    V = rng.normal(size=(12, 3))
    V /= np.linalg.norm(V, axis=1, keepdims=True)

    rep = ExactPolarRepresentation.from_polar(r, V)
    r_out, v_out = rep.to_polar()
    assert np.allclose(r_out.numpy(), r, atol=1e-12)
    assert np.allclose(v_out.numpy(), V, atol=1e-12)


def test_representation_exposes_one_manifold_parameter():
    """The warp couples ``r`` and ``v``, so they must share a single parameter —
    unlike ``PolarRepresentation``'s Euclidean+Sphere pair."""
    rep = ExactPolarRepresentation.from_polar(np.array([1.0, 2.0]),
                                              np.array([[1.0, 0.0], [0.0, 1.0]]))
    params = rep.parameters()
    assert len(params) == 1
    assert isinstance(params[0].manifold, WarpedPolarHyperboloid)
    assert params[0].shape == (2, 3)          # [r, v] packed


def test_representation_distance_matches_the_plain_polar_chart():
    """Same chart, same geometry — only the optimiser step differs."""
    from hypegrl.representations import PolarRepresentation

    rng = np.random.default_rng(4)
    r = rng.uniform(0.1, 20.0, size=8)
    V = rng.normal(size=(8, 2))
    V /= np.linalg.norm(V, axis=1, keepdims=True)

    exact = ExactPolarRepresentation.from_polar(r, V).dist().detach().numpy()
    plain = PolarRepresentation.from_polar(r, V).dist().detach().numpy()
    assert np.allclose(exact, plain, atol=1e-8)


def test_riemannian_adam_optimises_on_the_exact_manifold():
    """End-to-end: geoopt drives the manifold and the loss falls.

    Pulls 6 points to a target radius through the packed parameter, exercising
    ``projx``/``egrad2rgrad``/``retr_transp`` for real.
    """
    import geoopt

    rng = np.random.default_rng(5)
    V = rng.normal(size=(6, 2))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    rep = ExactPolarRepresentation.from_polar(rng.uniform(1.0, 3.0, size=6), V)
    opt = geoopt.optim.RiemannianAdam(rep.parameters(), lr=1e-2)

    def loss_fn():
        r, _ = rep._unpack()
        return ((r - 5.0) ** 2).sum()

    first = loss_fn().item()
    for _ in range(200):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
    last = loss_fn().item()

    assert last < first * 0.5
    assert torch.isfinite(rep.parameters()[0]).all()


def test_embedder_accepts_the_exact_polar_chart():
    """It is selectable as ``representation="exact_polar"`` and fits."""
    import networkx as nx

    from hypegrl.embedders import PoincareMapsEmbedder

    G = nx.balanced_tree(2, 3)
    emb = PoincareMapsEmbedder(d=2, representation="exact_polar", n_steps=60,
                               lr_X=3e-2, random_state=0, log_every=0).fit(G)
    X = emb.embeddings()
    assert X.shape == (G.number_of_nodes(), 2)
    assert np.isfinite(X).all()
    assert emb._loss_history[-1] < emb._loss_history[0]
