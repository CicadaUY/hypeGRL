# -*- coding: utf-8 -*-
"""
The ``Representation`` abstraction: a swappable chart in which an embedding is
stored and optimised.

Every gradient embedder's loss is a function of the pairwise hyperbolic distance
matrix, and ball / hyperboloid / polar are isometric charts of the *same*
H^{D+1} — they differ only in numerical conditioning. So the chart is an
orthogonal, swappable axis. A ``Representation`` owns the optimisable parameters
(``geoopt.ManifoldParameter`` s), computes ``dist()`` from them, and converts
to/from any chart.

Canonical interchange coordinate: **polar** ``(r, v)`` — ``r ≥ 0`` hyperbolic
radius, ``v ∈ S^D`` unit vector. It is the lossless hub (see
:mod:`hypegrl.manifolds.conversions`). Each subclass implements only the two
polar hooks plus ``parameters``/``dist``; the base composes them with the exact
converters to provide ``from_ball``/``from_hyperboloid`` and
``to_ball``/``to_hyperboloid`` once, for every subclass.

The Representation is *consumed by* the existing optimisers
(``riemannian_optimize`` / ``joint_optimize``); it deliberately does not carry
its own optimisation loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch

from hypegrl.manifolds.conversions import (
    ball_to_polar_torch,
    hyperboloid_to_polar_torch,
    polar_to_ball_torch,
    polar_to_hyperboloid_torch,
)


def as_tensor(x, device: str = "cpu") -> torch.Tensor:
    """Coerce ``x`` (array or tensor) to a float64 tensor on ``device``."""
    return torch.as_tensor(x, dtype=torch.float64, device=torch.device(device))


def build_representation(
    rep_cls: "type[Representation]",
    X_init,
    input_chart: str = "ball",
    device: str = "cpu",
    **cfg,
) -> "Representation":
    """
    Build a ``rep_cls`` warm start from an embedder's ``X_init`` argument.

    ``X_init`` is either a :class:`Representation` (any chart) — re-charted into
    ``rep_cls`` via ``from_polar`` (its ``to_polar`` readout is the lossless hub),
    so an exact large-radius warm start survives without ball saturation — or a
    coordinate array in ``input_chart`` (the embedder's native input chart,
    ``"ball"`` or ``"hyperboloid"``), ingested via the matching ``from_*``
    constructor. ``cfg`` is forwarded to the constructor (e.g. ``max_norm`` for
    the hyperboloid).
    """
    if hasattr(X_init, "to_polar"):
        r, v = X_init.to_polar()
        return rep_cls.from_polar(r, v, device=device, **cfg)
    ingest = {
        "ball": rep_cls.from_ball,
        "hyperboloid": rep_cls.from_hyperboloid,
    }[input_chart]
    return ingest(np.asarray(X_init, dtype=np.float64), device=device, **cfg)


class Representation(ABC):
    """
    Base class for a chart-specific embedding representation.

    Subclasses implement four methods:

    - ``from_polar(r, v, **cfg)`` — build a representation from canonical polar
      coordinates (the primary constructor; ``from_ball``/``from_hyperboloid``
      delegate to it after converting).
    - ``to_polar()`` — read the current point set back as ``(r, v)``.
    - ``parameters()`` — the ``geoopt`` parameters ``RiemannianAdam`` steps.
    - ``dist()`` — the ``(N, N)`` pairwise hyperbolic distance from those
      parameters (autograd-differentiable).
    """

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def from_polar(cls, r, v, **cfg) -> "Representation":
        ...

    @abstractmethod
    def to_polar(self) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    @abstractmethod
    def parameters(self) -> list[torch.Tensor]:
        ...

    @abstractmethod
    def dist(self) -> torch.Tensor:
        ...

    # ------------------------------------------------------------------
    # Ingestion from other charts (convert → polar → subclass)
    # ------------------------------------------------------------------

    @classmethod
    def from_ball(cls, X, **cfg) -> "Representation":
        """Build from Poincaré-ball coordinates ``X`` ``(N, D+1)``."""
        r, v = ball_to_polar_torch(as_tensor(X, cfg.get("device", "cpu")))
        return cls.from_polar(r, v, **cfg)

    @classmethod
    def from_hyperboloid(cls, H, **cfg) -> "Representation":
        """Build from hyperboloid coordinates ``H`` ``(N, D+2)``."""
        r, v = hyperboloid_to_polar_torch(as_tensor(H, cfg.get("device", "cpu")))
        return cls.from_polar(r, v, **cfg)

    # ------------------------------------------------------------------
    # Readout to other charts (subclass → polar → convert)
    # ------------------------------------------------------------------

    def to_ball(self) -> torch.Tensor:
        """Poincaré-ball coordinates (saturates past ``r ≈ 12`` — for plots)."""
        return polar_to_ball_torch(*self.to_polar())

    def to_hyperboloid(self) -> torch.Tensor:
        """Hyperboloid coordinates (exact to ``r ≈ 350``)."""
        return polar_to_hyperboloid_torch(*self.to_polar())


__all__ = ["Representation", "as_tensor", "build_representation"]
