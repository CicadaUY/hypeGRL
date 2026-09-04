# -*- coding: utf-8 -*-
"""
The ``Representation`` abstraction: a swappable chart in which an embedding is
stored and optimised.

Every gradient embedder's loss is a function of the pairwise hyperbolic distance
matrix, and ball / hyperboloid / polar / tangent are all charts of the *same*
H^{D+1}: each one's ``dist()`` is the exact hyperbolic distance, so they share
their minima and the chart is an orthogonal, swappable axis. A
``Representation`` owns the optimisable parameters (``geoopt.ManifoldParameter``
s), computes ``dist()`` from them, and converts to/from any chart.

Choosing a chart is not only a choice of numerical conditioning
---------------------------------------------------------------

It settles two independent questions, and only the first is about precision:

1. **What the coordinates can represent** — the radius at which the stored
   numbers stop resolving distinct points. The ball saturates, the hyperboloid
   overflows, polar and tangent stay exact far past both. Each class documents
   its own range.
2. **How the optimiser moves them** — the metric under which the gradient is
   converted into a step. This never appears in the loss, so it is invisible in
   the objective value, yet it decides *where* the run ends up.

The second is easiest to see in the angular direction. Separating two points at
radius ``r`` by an angle ``δθ`` moves them along an arc of length
``sinh(r)·δθ``, which grows without bound in ``r``. A chart optimised under the
exact metric spends its step budget on that arc, so it makes very slow angular
progress at large radius; the polar chart's product metric instead moves the
angular *coordinate* at a rate set by ``lr`` alone, regardless of radius. Same
space, same minima, very different paths — and the effect is unbounded, not a
constant factor.

The effect is strong enough to stall a run outright. A chart under the exact
metric changes the angular coordinate by only ``≈ lr/sinh(r)``; if the starting
point is also near-optimal radially — as a warm start from a method's own
initialisation typically is — then ``∂f/∂r`` is small too, and *neither*
component of the step moves the embedding appreciably.

Consequences: a shared ``lr`` does **not** mean a shared step size across
charts, so comparing charts at one ``lr`` compares nothing. Every comparison
needs a per-chart ``lr`` sweep, read at each chart's own optimum. And no ``lr``
rescues the stall above, since it scales both components alike.

What each chart does, in this library
-------------------------------------

The metric column is the substantive choice; the last two columns are how
``geoopt`` 0.5.1 happens to implement the manifolds we build on (see below)::

    chart         parameters                        metric on the angle  exact range
    polar         Euclidean(u), r = softplus(u)     product (no warp)    r ≈ 350
                  + Sphere(v)
    tangent       Euclidean(z = r·v)                z-norm scaling       r ≈ 350
    exact_polar   WarpedPolarHyperboloid            exact, sinh²r        r ≈ 350
    curved_polar  WarpedPolarHyperboloid(c)         warped, w_c(r)²      r ≈ 350
    ball          PoincareBall                      exact (conformal)    r ≈ 8.4
    hyperboloid   StableLorentz                     exact (Minkowski)    r ≈ 7.6

``curved_polar`` makes the metric column a continuum rather than a set of
cases: its ``chart_curvature`` runs from ``exact_polar``'s exact warp at
``c = 1`` down towards ``tangent``'s flat one, without changing the distance
the loss decodes (see
:class:`~hypegrl.representations.polar.CurvedPolarRepresentation`).

The polar chart's radius is the one parameter that is not the coordinate itself:
``r = softplus(u)`` tapers the radial step near the origin, deliberately (see
:class:`~hypegrl.representations.polar.PolarRepresentation`).

The two short ranges are set by constants rather than by ``float64``, and each
class documents its own: the ball's is a ceiling on the *pair separation*
(``16.811243``, reached by two points at ``r = 8.4`` in opposite directions),
and the hyperboloid's is the ``max_norm`` clamp (default ``1e3``, i.e.
``r ≤ arcsinh(1e3) ≈ 7.6``, raisable up to the arithmetic limit ``r ≈ 18`` where
the Minkowski inner product cancels).

So ``ball``, ``hyperboloid`` and ``exact_polar`` share one metric and differ in
representable range, while ``polar``, ``tangent`` and ``exact_polar`` share an
effectively unlimited range and differ in metric. A comparison that varies both
at once cannot attribute its result to either.

Behaviour inherited from geoopt, not chosen here
------------------------------------------------

Two further details differ between charts. Both come from ``geoopt`` 0.5.1 as
installed — they are not decisions of this library, and a future ``geoopt`` may
change them, so a chart comparison is reproducible only against a pinned
version.

- **The retraction** — how a tangent step becomes a new point.
  ``RiemannianAdam`` moves through ``retr_transp → retr``. For ``PoincareBall``
  and ``Sphere``, ``geoopt``'s ``retr`` is the first-order projected addition
  ``project(x + u)``, not the exponential map; ``Lorentz`` sets
  ``retr = expmap``, and so does
  :class:`~hypegrl.manifolds.polar.WarpedPolarHyperboloid`, which is ours.
  ``geoopt`` also ships ``PoincareBallExact`` and ``SphereExact``, which set
  ``retr = expmap`` — swapping those in would remove this difference and is
  untried here.
- **Adam's second moment** — over what unit the adaptive scale is accumulated.
  ``RiemannianAdam`` accumulates ``manifold.component_inner``, which
  ``Euclidean`` overrides to be *per coordinate* while every other manifold
  used here keeps the base implementation, *one scalar per point*. So the polar
  chart's radius is scaled per coordinate and its direction per point, whereas
  the ball, hyperboloid and exact-polar charts get a single shared scale per
  point.

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
    constructor.

    ``cfg`` is forwarded to the constructor — the chart-specific options an
    embedder's ``representation_kwargs`` carries, e.g. ``max_norm`` for the
    hyperboloid or ``chart_curvature`` for the curved polar chart. No
    ``from_polar`` takes ``**kwargs``, so an option the chosen chart does not
    name raises ``TypeError`` from the call itself rather than being silently
    dropped into a fit that then runs with the default the caller meant to
    replace.
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

    Subclasses implement five methods:

    - ``from_polar(r, v, **cfg)`` — build a representation from canonical polar
      coordinates (the primary constructor; ``from_ball``/``from_hyperboloid``
      delegate to it after converting).
    - ``to_polar()`` — read the current point set back as ``(r, v)``.
    - ``parameters()`` — the ``geoopt`` parameters ``RiemannianAdam`` steps.
    - ``dist()`` — the ``(N, N)`` pairwise hyperbolic distance from those
      parameters (autograd-differentiable, zero diagonal).
    - ``dist_between(i_idx, j_idx)`` — the same distance, gathered for only
      the given index pairs, without materialising the full matrix (what a
      negative-sampling loss decodes; see its own docstring).

    **Differentiable accessor vs. readouts (important):** ``dist()`` and
    ``dist_between()`` are the *live* geometric accessors — built from the
    parameters and carrying the gradient, so they're what a loss decodes. The
    ``to_polar`` / ``to_ball`` / ``to_hyperboloid`` methods are **detached
    readouts** for inspection and interop (plotting, `embeddings()`,
    warm-starting a later fit); a decoder built on them would silently
    receive no gradient. If a future decoder needs a *differentiable*
    geometric quantity other than distance (e.g. the inner-product an RDPG
    decoder wants), add a purpose-built live accessor alongside ``dist()``
    (``inner()``) rather than differentiating a ``to_*``.
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

    @abstractmethod
    def dist_between(self, i_idx: torch.Tensor, j_idx: torch.Tensor) -> torch.Tensor:
        """
        Differentiable hyperbolic distance between indexed pairs of points,
        without materialising the full ``(N, N)`` matrix ``dist()`` builds.

        ``i_idx``/``j_idx`` are integer index tensors into this
        representation's node order, broadcastable against each other exactly
        as plain fancy indexing would be: both ``(P,)`` gives ``P`` pairwise
        distances (one per row), and ``(P, 1)`` against ``(P, K)`` gives ``K``
        distances per row (e.g. a positive index paired with its ``K`` sampled
        negatives). Equivalent to ``self.dist()[i_idx, j_idx]``, but
        ``O(len(i_idx)·len(j_idx))`` memory/compute instead of ``O(N²)`` — the
        gather-based accessor a negative-sampling loss decodes, so the full
        matrix is never built. No diagonal handling: callers never pass
        ``i_idx == j_idx`` (real edges have ``i≠j``; sampled negatives exclude
        self).
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
        Detached Poincaré-ball coordinates — for plotting / interop, not
        gradients (see the class differentiability note).

        Exact to ``r ≈ 28.32``, the clamp in
        :mod:`hypegrl.manifolds.conversions`; every larger radius reads back as
        ``28.32``. Well before that the ball radius crowds against ``1``
        (``1 − 4e-9`` at ``r = 20``), so large radii are visually
        indistinguishable on a disk and any distance recomputed from these
        coordinates through ``geoopt`` hits its ``16.811243`` ceiling
        (:class:`~hypegrl.representations.ball.BallRepresentation`). Prefer
        :meth:`to_polar` wherever the radius carries meaning.
        """
        return polar_to_ball_torch(*self.to_polar())

    def to_hyperboloid(self) -> torch.Tensor:
        """
        Detached hyperboloid coordinates (exact to ``r ≈ 350`` — for interop, not
        gradients; see the class differentiability note).
        """
        return polar_to_hyperboloid_torch(*self.to_polar())


__all__ = ["Representation", "as_tensor", "build_representation", "zero_diagonal"]
