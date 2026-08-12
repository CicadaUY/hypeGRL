# -*- coding: utf-8 -*-
"""
Poincaré-ball representation: optimise on ``geoopt.PoincareBall``.

A baseline chart, included for the representation comparison. Faithful at
moderate radius, and bounded above by two ``geoopt`` safety constants — *not*
by ``float64``:

- **The stored coordinate is clamped.** ``geoopt``'s ``project`` caps the norm at
  ``1 − eps`` (``eps = 1e-5`` in double precision), i.e. a hyperbolic radius
  ``r ≤ 2·artanh(1 − 1e-5) ≈ 12.21``. ``BallRepresentation`` applies it on
  construction and ``RiemannianAdam`` re-applies it at every step, so a warm
  start beyond that shell is squashed onto it and the radial coordinate of
  leaf-heavy / large graphs collapses.
- **The distance has a hard ceiling.** ``POINCARE_BALL.dist`` is
  ``2·artanh(‖−x ⊕ y‖)`` and ``geoopt``'s ``artanh`` clamps its argument to
  ``1 − 1e-7``, so any pair further apart than ``2·artanh(1 − 1e-7) =
  16.811243`` returns *exactly* that value, with an exactly zero gradient.
  This is a ceiling on the **pair separation**, so it binds well inside the
  coordinate clamp: two nodes at ``r = 8.41`` in opposite directions are already
  past it. It is the limit that bites first in practice.

Away from that ceiling the chart stays accurate — a pair one unit apart is
recovered to a relative error of ``2e-11`` at ``r = 8`` and ``8e-8`` at
``r = 12`` — so the failure mode is the two clamps, not gradual precision loss.

Both constants are *stricter* than double precision itself: in raw ``float64``
the ball coordinate ``tanh(r/2)`` stays distinct from ``1`` until ``r ≈ 38``
(Mishne, Wan, Wang & Yang 2023, arXiv:2211.00181, §3). The exact converters in
:mod:`hypegrl.manifolds.conversions` carry their own, looser limit, so a
``to_ball()`` readout is not bounded by the two constants above.

How the steps are taken
-----------------------

Range is only half of what a chart settles; the other half is the metric the
optimiser steps under, and there this chart is *not* a neutral baseline.
``RiemannianAdam`` calls ``PoincareBall.egrad2rgrad``, which divides by the
conformal factor — the exact hyperbolic metric. That is the same choice as
:class:`~hypegrl.representations.polar.ExactPolarRepresentation` and the
hyperboloid, and a different one from
:class:`~hypegrl.representations.polar.PolarRepresentation`, whose product
metric drops the ``sinh²r`` warp. A ball-versus-polar comparison therefore
varies the metric as well as the range.

One detail comes from ``geoopt`` 0.5.1 rather than from this library:
``PoincareBall.retr`` is the first-order projected addition ``project(x + u)``,
not the exponential map, so the step lands slightly off the geodesic. ``geoopt``
ships ``PoincareBallExact`` (``retr = expmap``) if that difference ever needs to
be removed; this chart uses the plain ``PoincareBall``.
"""

from __future__ import annotations

import geoopt
import torch

from hypegrl.manifolds.conversions import ball_to_polar_torch, polar_to_ball_torch
from hypegrl.manifolds.poincare import POINCARE_BALL
from hypegrl.representations.base import Representation, as_tensor, zero_diagonal


class BallRepresentation(Representation):
    """Embedding stored as Poincaré-ball coordinates ``X`` ``(N, D+1)``."""

    def __init__(self, X: torch.Tensor, device: str = "cpu"):
        X = POINCARE_BALL.projx(as_tensor(X, device))
        self._X = geoopt.ManifoldParameter(X, manifold=POINCARE_BALL)

    @classmethod
    def from_polar(cls, r, v, device: str = "cpu", **_) -> "BallRepresentation":
        X = polar_to_ball_torch(as_tensor(r, device), as_tensor(v, device))
        return cls(X, device=device)

    def to_polar(self) -> tuple[torch.Tensor, torch.Tensor]:
        return ball_to_polar_torch(self._X.detach())

    def parameters(self) -> list[torch.Tensor]:
        return [self._X]

    def dist(self) -> torch.Tensor:
        return zero_diagonal(
            POINCARE_BALL.dist(self._X.unsqueeze(1), self._X.unsqueeze(0)))


__all__ = ["BallRepresentation"]
