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
negative and make ``sinh r`` invalid. Model-agnostic: **no κ** (that is a
D-Mercator readout computed from ``r`` with the model's global params).
"""

from __future__ import annotations

import geoopt
import torch
import torch.nn.functional as F

from hypegrl.manifolds.polar import polar_distances_torch
from hypegrl.representations.base import Representation, as_tensor

_SPHERE = geoopt.Sphere()
_EUCLIDEAN = geoopt.Euclidean()


class PolarRepresentation(Representation):
    """Embedding stored as ``(u, v)`` with ``r = softplus(u)``, ``v ∈ S^D``."""

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
        return polar_distances_torch(self._radius(), self._v)


__all__ = ["PolarRepresentation"]
