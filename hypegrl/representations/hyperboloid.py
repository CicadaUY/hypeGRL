# -*- coding: utf-8 -*-
"""
Hyperboloid representation: optimise on ``StableLorentz`` (a spatial-norm-clamped
``geoopt.Lorentz``).

A baseline chart for the comparison. Distances use the manifold's
``dist`` = ``arccosh(−⟨x,y⟩_L)``, whose Minkowski inner product is destroyed by
catastrophic cancellation past ``r ≈ 18`` (returning negative / NaN / −∞ via the
guarded ``arccosh``).

``max_norm`` clamps the spatial norm inside ``projx``/retraction, which also caps
the radius: the default ``1e3`` gives ``r ≤ arcsinh(1e3) ≈ 7.6``, so a warm start
at larger radius is squashed on construction. To expose the *distance* failure
(rather than the clamp) in the comparison, build with a large ``max_norm``
(e.g. ``1e18``, ``r ≤ ~42``).

This is the **only** representation that owns its manifold (``self._manifold``),
because ``StableLorentz`` is the only one with per-instance state: ``max_norm`` is
graph-dependent (tuned per fit) and drives ``RiemannianAdam``'s retraction, so a
fresh instance is built per representation rather than sharing or mutating a
global. ``BallRepresentation`` / ``PolarRepresentation`` instead reference the
fixed, parameterless module-level manifolds (``POINCARE_BALL`` / ``Sphere`` +
``Euclidean``). Curvature is *not* a per-instance manifold parameter here: every
chart fixes ``k = 1``, and the metric scale (curvature) is a global quantity
absorbed into the loss, never stored in a chart.
"""

from __future__ import annotations

import geoopt
import torch

from hypegrl.manifolds.conversions import (
    hyperboloid_to_polar_torch,
    polar_to_hyperboloid_torch,
)
from hypegrl.manifolds.lorentz import StableLorentz
from hypegrl.representations.base import Representation, as_tensor, zero_diagonal


class HyperboloidRepresentation(Representation):
    """Embedding stored as hyperboloid coordinates ``H`` ``(N, D+2)``."""

    def __init__(self, H: torch.Tensor, manifold: StableLorentz, device: str = "cpu"):
        self._manifold = manifold
        H = manifold.projx(as_tensor(H, device))
        self._H = geoopt.ManifoldParameter(H, manifold=manifold)

    @classmethod
    def from_polar(
        cls, r, v, device: str = "cpu", max_norm: float = 1e3, **_,
    ) -> "HyperboloidRepresentation":
        manifold = StableLorentz(max_norm=max_norm)
        H = polar_to_hyperboloid_torch(as_tensor(r, device), as_tensor(v, device))
        return cls(H, manifold=manifold, device=device)

    def to_polar(self) -> tuple[torch.Tensor, torch.Tensor]:
        return hyperboloid_to_polar_torch(self._H.detach())

    def parameters(self) -> list[torch.Tensor]:
        return [self._H]

    def dist(self) -> torch.Tensor:
        return zero_diagonal(
            self._manifold.dist(self._H.unsqueeze(1), self._H.unsqueeze(0)))


__all__ = ["HyperboloidRepresentation"]
