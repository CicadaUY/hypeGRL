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


def zero_diagonal(D: torch.Tensor) -> torch.Tensor:
    """
    Return the ``(N, N)`` distance matrix ``D`` with an exact-zero diagonal,
    differentiably (``masked_fill`` passes gradients to the off-diagonal and
    zeros them on the diagonal, whose self-distance has no meaningful gradient).

    Each chart's underlying distance leaves a *different* residue on the
    diagonal — ``0`` on the ball, ``≈3e-8`` on the hyperboloid (geoopt's guarded
    ``arccosh``), ``≈1e-15`` on polar (the ``1e-30`` floor inside the sqrt). Every
    :meth:`Representation.dist` routes through this so the three charts agree
    *exactly* on the diagonal, and any decoder that touches the full matrix
    (e.g. a row ``softmax``) sees a clean self-distance of zero.
    """
    n = D.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=D.device)
    return D.masked_fill(eye, 0.0)


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
      parameters (autograd-differentiable, zero diagonal).

    **Differentiable accessor vs. readouts (important):** ``dist()`` is the
    *live* geometric accessor — it is built from the parameters and carries the
    gradient, so it is what a loss decodes. The ``to_polar`` / ``to_ball`` /
    ``to_hyperboloid`` methods are **detached readouts** for inspection and
    interop (plotting, `embeddings()`, warm-starting a later fit); a decoder
    built on them would silently receive no gradient. If a future decoder needs a
    *differentiable* geometric quantity other than distance (e.g. the
    inner-product an RDPG decoder wants), add a purpose-built live accessor
    alongside ``dist()`` (``inner()``) rather than differentiating a ``to_*``.
    """

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def from_polar(cls, r, v, **cfg) -> "Representation":
        """Build from canonical polar coordinates ``(r, v)`` (primary constructor)."""
        ...

    @abstractmethod
    def to_polar(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Detached readout of the current points as ``(r, v)`` (not for gradients)."""
        ...

    @abstractmethod
    def parameters(self) -> list[torch.Tensor]:
        """The ``geoopt`` parameters ``RiemannianAdam`` steps (may span manifolds)."""
        ...

    @abstractmethod
    def dist(self) -> torch.Tensor:
        """
        The ``(N, N)`` pairwise hyperbolic distance — the **live, differentiable**
        geometric accessor a loss decodes. Built from :meth:`parameters`, with an
        exact-zero diagonal (see :func:`zero_diagonal`).
        """
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
        """
        Detached Poincaré-ball coordinates (saturates past ``r ≈ 12`` — for
        plotting / interop, not gradients; see the class differentiability note).
        """
        return polar_to_ball_torch(*self.to_polar())

    def to_hyperboloid(self) -> torch.Tensor:
        """
        Detached hyperboloid coordinates (exact to ``r ≈ 350`` — for interop, not
        gradients; see the class differentiability note).
        """
        return polar_to_hyperboloid_torch(*self.to_polar())


__all__ = ["Representation", "as_tensor", "build_representation", "zero_diagonal"]
