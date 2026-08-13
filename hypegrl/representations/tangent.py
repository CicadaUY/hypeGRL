# -*- coding: utf-8 -*-
"""
Tangent representation: optimise the plain Euclidean vector ``z = r·v ∈ ℝ^d``.

Mishne, Wan, Wang & Yang (2023, arXiv:2211.00181, §4) parametrise a hyperbolic
point by its image in the tangent space at the origin, ``z = exp_0^{-1}(x) ∈ ℝ^d``,
and optimise the *plain Euclidean* function ``h(z) = f(exp_0(z))`` — no Riemannian
metric in the optimiser, ``z`` unconstrained. Here ``‖z‖`` is the hyperbolic radius
and ``z/‖z‖`` the unit direction, so ``z`` is the polar ``(r, v)`` packed into one
Cartesian vector.

We never form ``exp_0(z)``. Every loss in the library is a function of the pairwise
distance, and that distance is ``polar_distances(‖z‖, z/‖z‖)`` exactly (the distance
between the ``exp_0`` images), so we read ``(r, v)`` off ``z`` and route through the
same exact :func:`~hypegrl.manifolds.polar.polar_distances_torch` as the polar
charts. The autograd path through ``r = ‖z‖`` supplies Mishne's exp-map Jacobian, so
the optimiser inherits the tangent parametrisation's gradient behaviour without the
ball/hyperboloid coordinates it was defined on.

Same ``dist()`` as the two polar charts, different parametrisation:
:class:`~hypegrl.representations.polar.PolarRepresentation` splits ``(r, v)`` across
``Euclidean × Sphere`` (product metric — angular step radius-independent),
:class:`~hypegrl.representations.polar.ExactPolarRepresentation` uses the warped
metric (angular gradient ÷ ``sinh²r``). The Cartesian ``z`` sits between them: its
angular gradient carries a single ``1/‖z‖`` from ``∂v/∂z``, which is Mishne's point —
the large-radius gradient decays as ``δ¹`` rather than the ball's ``δ²``, and ``z`` is
unbounded so there is no ball saturation or hyperboloid overflow.

**Not rotation-invariant.** ``RiemannianAdam`` accumulates its second moment
through ``geoopt`` 0.5.1's ``component_inner``, which ``Euclidean`` — alone among
the manifolds used here — defines *per coordinate*, so the update is a diagonal
rescaling in the standard basis and does not commute with a rotation of ``z``.
Since the loss sees only distances, a rotated copy of a configuration is the same
problem, yet it follows a different path: the result carries an arbitrary
dependence on the frame the warm start arrived in, which the charts whose second
moment is one scalar per point do not. The dependence is the optimiser's, not the
parametrisation's — plain ``RiemannianSGD`` on this chart is exactly equivariant.
"""

from __future__ import annotations

import geoopt
import torch

from hypegrl.manifolds.polar import polar_distances_torch
from hypegrl.representations.base import Representation, as_tensor, zero_diagonal

_EUCLIDEAN = geoopt.Euclidean()
_TINY = 1e-15


class TangentRepresentation(Representation):
    """Embedding stored as the Cartesian tangent vector ``z = r·v`` (``r = ‖z‖``)."""

    def __init__(self, z: torch.Tensor, device: str = "cpu"):
        dev = torch.device(device)
        self._z = geoopt.ManifoldParameter(z.to(dev), manifold=_EUCLIDEAN)

    @classmethod
    def from_polar(cls, r, v, device: str = "cpu", **_) -> "TangentRepresentation":
        r = as_tensor(r, device).reshape(-1)
        v = as_tensor(v, device)
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(_TINY)
        return cls(r[:, None] * v, device=device)

    def _polar(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        ``(r, v) = (‖z‖, z/‖z‖)`` off ``z``, differentiably. The ``1e-30`` floor
        inside the sqrt keeps the gradient finite for a node at the origin (``z = 0``,
        where ``∂r/∂z = z/r`` would otherwise be ``0/0``).
        """
        r = torch.sqrt((self._z ** 2).sum(-1) + 1e-30)
        return r, self._z / r[:, None]

    def to_polar(self) -> tuple[torch.Tensor, torch.Tensor]:
        r, v = self._polar()
        return r.detach(), v.detach()

    def parameters(self) -> list[torch.Tensor]:
        return [self._z]

    def dist(self) -> torch.Tensor:
        r, v = self._polar()
        return zero_diagonal(polar_distances_torch(r, v))


__all__ = ["TangentRepresentation"]
