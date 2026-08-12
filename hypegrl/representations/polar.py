# -*- coding: utf-8 -*-
"""
Polar representation: optimise on ``S^D × ℝ₊``.

The numerically robust chart — no ``e^r`` in the coordinates (unlike the
hyperboloid's ``cosh r``) and no boundary saturation (unlike the ball's
``tanh(r/2) → 1``), so it stays exact at the large radii real graphs need.

Parameters: the direction ``v`` on ``geoopt.Sphere`` (S^D), and the radius via
``r = softplus(u)`` with ``u`` an unconstrained Euclidean parameter. The
softplus keeps ``r > 0`` smoothly under RiemannianAdam — a raw clamped ``r``
would stick at a zero gradient once it hit 0, and a raw ``r`` could step
negative and make ``sinh r`` invalid. It is not only a positivity device: it
also tapers the radial step near the origin, which is documented on
:class:`PolarRepresentation`. Model-agnostic: **no κ** (that is a D-Mercator
readout computed from ``r`` with the model's global params).
"""

from __future__ import annotations

import geoopt
import torch
import torch.nn.functional as F

from hypegrl.manifolds.polar import WarpedPolarHyperboloid, polar_distances_torch
from hypegrl.representations.base import Representation, as_tensor, zero_diagonal

_SPHERE = geoopt.Sphere()
_EUCLIDEAN = geoopt.Euclidean()
_TINY = 1e-15


class PolarRepresentation(Representation):
    """
    Embedding stored as ``(u, v)`` with ``r = softplus(u)``, ``v ∈ S^D``.

    The radius lives on ``geoopt.Euclidean`` and the direction on
    ``geoopt.Sphere``, so the pair carries the *product* metric
    ``dr² + ⟨dv, dv⟩`` rather than the true ``dr² + sinh²(r)·g_{S^D}``. This is
    the default chart: it is exact at every radius and, unlike the exact metric,
    it keeps moving the angles when the radius is large.

    **Its one structural weakness is a spread of radii, and no ``lr`` fixes it.**
    ``RiemannianAdam`` normalises the step by the gradient's own magnitude, and
    the direction is pooled into a single scalar per node, so *every* node turns
    by ``‖δv‖ ≈ lr`` per step whatever its radius. The distance a node actually
    travels is ``sinh(r)·‖δv‖``, so two nodes at radii ``r_max`` and ``r_min``
    move by amounts in the ratio ``sinh(r_max)/sinh(r_min) ≈ e^(r_max − r_min)``
    from the same ``lr``. Pick ``lr`` for the outer nodes and the inner ones
    barely move; pick it for the inner ones and the outer ones are flung across
    the space. Tuning cannot resolve this, because it is a property of the
    *spread* rather than of the magnitude — a graph whose nodes all sit at one
    radius is unaffected however large that radius is.

    In practice the relevant spread is over the radii that carry the loss, not
    the extremes: the outermost nodes are typically low-degree and contribute
    little, so displacing them costs little. The usable ``lr`` therefore depends
    on the graph, and a large-radius graph generally needs a much smaller one
    than the default.

    A second, milder consequence of the same fact: since a node turns by ``≈ lr``
    per step, any node whose required angular correction is *below* ``lr``
    overshoots it immediately. The chart's angular resolution is ``lr``.

    **The radius is reparametrised, and that rescales the radial step.** Storing
    ``r = softplus(u)`` and stepping ``u`` means a step ``δu`` moves the radius by
    ``σ(u)·δu = (1 − e^{−r})·δu``. Since ``RiemannianAdam`` makes the step in
    ``u`` about ``lr`` whatever the gradient, the radial displacement per step is
    ``≈ lr·(1 − e^{−r})``: indistinguishable from ``lr`` beyond ``r ≈ 3`` (5%
    short), then damped progressively below it — 22% short at ``r = 1.5``, and an
    order of magnitude at ``r = 0.1``. Only the innermost nodes are affected, and
    the taper dies exponentially, so this does not compound with the angular
    spread described above.

    That taper is a reason to prefer this parametrisation, not a cost of it. Both
    obvious alternatives take full-size radial steps near the origin, which is
    where the direction ``v`` is worst determined and where a step large enough to
    cross zero effectively moves the node to the antipode. A radius clamped at 0
    has no gradient below the clamp, so a node that once steps past the origin
    stays pinned there for the rest of the run; ``r = |u|`` removes the dead zone
    but reflects through the origin instead. The softplus makes the approach to
    the origin asymptotic, so neither can happen.
    """

    def __init__(self, u: torch.Tensor, v: torch.Tensor, device: str = "cpu"):
        dev = torch.device(device)
        self._u = geoopt.ManifoldParameter(u.to(dev), manifold=_EUCLIDEAN)
        self._v = geoopt.ManifoldParameter(
            _SPHERE.projx(v.to(dev)), manifold=_SPHERE)

    @classmethod
    def from_polar(cls, r, v, device: str = "cpu", **_) -> "PolarRepresentation":
        r = as_tensor(r, device)
        v = as_tensor(v, device)
        # inverse softplus: u = log(expm1(r)); clamp keeps log(expm1(0)) finite.
        u = torch.log(torch.expm1(r.clamp_min(1e-6)))
        return cls(u, v, device=device)

    def _radius(self) -> torch.Tensor:
        return F.softplus(self._u)

    def to_polar(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._radius().detach(), self._v.detach()

    def parameters(self) -> list[torch.Tensor]:
        return [self._u, self._v]

    def dist(self) -> torch.Tensor:
        return zero_diagonal(polar_distances_torch(self._radius(), self._v))


class ExactPolarRepresentation(Representation):
    """
    Polar chart optimised under the **exact** hyperbolic metric.

    Same chart and same ``dist()`` as :class:`PolarRepresentation` — the
    difference is purely *how the parameters move*. Here ``(r, v)`` is a single
    packed ``[r, v]`` point on :class:`~hypegrl.manifolds.polar.WarpedPolarHyperboloid`,
    whose metric ``dr² + sinh²(r)·g_{S^D}`` is the true one, so ``RiemannianAdam``
    takes genuine Riemannian steps: the gradient is rescaled by the warp
    (``G⁻¹ = diag(1, sinh⁻²r·I)``) and the retraction is the exact geodesic.

    ``PolarRepresentation`` instead splits ``(r, v)`` across a ``Euclidean`` and a
    ``Sphere`` parameter, which imposes the *product* metric ``dr² + ⟨dv,dv⟩``:
    same minima, wrong gradient and wrong step scale. The practical consequence
    is that its effective step size drifts with radius under a shared ``lr``
    (under-driving at small ``r``, overshooting at large ``r``), whereas here the
    geometric step length is ``≈ lr`` at every radius.

    **That step length is a total budget, and at large radius the angle cannot
    afford it.** Turning by ``δθ`` at radius ``r`` costs an arc of ``sinh(r)·δθ``,
    so holding the step at ``≈ lr`` forces the angular coordinate to change by
    ``≈ lr/sinh(r)`` — a factor that vanishes exponentially. The radial direction
    carries no such penalty, so a run here moves almost purely radially once
    ``r`` is large. It follows that this chart is a poor choice for *refining* an
    embedding that is already near-optimal in the radial direction: ``∂f/∂r`` is
    then small by construction and the angular gradient is divided by
    ``sinh²(r)``, so both components of the step are negligible and the
    embedding barely moves. :class:`PolarRepresentation` and
    :class:`~hypegrl.representations.tangent.TangentRepresentation` have no such
    ceiling on angular motion and do move it. Prefer this chart at moderate
    radius, where its interpretable ``lr`` is a real advantage, and not as a
    large-radius refiner.

    The radius is stored **directly** (no softplus): the manifold keeps ``r ≥ 0``
    itself — its exponential map yields ``r ≥ 0`` by construction and flips ``v``
    to the antipode for a geodesic through the origin.

    Parameters
    ----------
    max_step:
        Forwarded to the manifold; a non-binding safety cap on the geodesic
        length of one step (see that class).
    """

    def __init__(self, r, v, device: str = "cpu", max_step: float = 30.0):
        r = as_tensor(r, device).reshape(-1, 1).clamp_min(0.0)
        v = as_tensor(v, device)
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(_TINY)
        self._manifold = WarpedPolarHyperboloid(max_step=max_step)
        self._x = geoopt.ManifoldParameter(
            torch.cat([r, v], dim=-1), manifold=self._manifold)

    @classmethod
    def from_polar(cls, r, v, device: str = "cpu", max_step: float = 30.0,
                   **_) -> "ExactPolarRepresentation":
        return cls(r, v, device=device, max_step=max_step)

    def _unpack(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(r, v)`` off the packed parameter, differentiably."""
        r = self._x[..., 0].clamp_min(0.0)
        v = self._x[..., 1:]
        return r, v / v.norm(dim=-1, keepdim=True).clamp_min(_TINY)

    def to_polar(self) -> tuple[torch.Tensor, torch.Tensor]:
        r, v = self._unpack()
        return r.detach(), v.detach()

    def parameters(self) -> list[torch.Tensor]:
        return [self._x]

    def dist(self) -> torch.Tensor:
        r, v = self._unpack()
        return zero_diagonal(polar_distances_torch(r, v))


__all__ = ["PolarRepresentation", "ExactPolarRepresentation"]
