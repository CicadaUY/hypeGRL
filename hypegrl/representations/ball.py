# -*- coding: utf-8 -*-
"""
Poincaré-ball representation: optimise on ``geoopt.PoincareBall``.

A baseline chart, included for the representation comparison. Faithful at
moderate radius, but its coordinates saturate past ``r ≈ 12`` (``tanh(r/2) → 1``
maps every large radius to the boundary), so it collapses the radial coordinate
of leaf-heavy / large graphs. Distances use ``POINCARE_BALL.dist`` (the exact
ball geodesic distance).
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
