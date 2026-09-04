# -*- coding: utf-8 -*-
"""
Polar (geodesic-polar) chart of hyperbolic space H^{D+1}.

A point is ``(r, v)``: the hyperbolic radius ``r ≥ 0`` from the origin and a
unit direction ``v ∈ S^D`` (a ``(D+1)``-vector, ``‖v‖ = 1``). This is the
numerically robust chart at large radius — ``r`` is a plain moderate number, so
no ``e^r`` ever forms in the coordinates (unlike the hyperboloid's ``cosh r``,
which overflows) and there is no boundary saturation (unlike the ball's
``tanh(r/2) → 1``). It is the lossless canonical hub of the Representation
abstraction; see :mod:`hypegrl.manifolds.conversions` for exact maps to/from the
ball and hyperboloid.

Curvature is fixed to ``k = 1``, matching ``POINCARE_BALL`` and ``LORENTZ``.
"""

from __future__ import annotations

import geoopt
import numpy as np
import torch


def polar_distances(r: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Pairwise geodesic distances ``(N, N)`` in H^{D+1} from polar coords.

    Stable form of the hyperbolic law of cosines. With ``Δr = r_i − r_j`` and
    the chord ``‖v_i − v_j‖²`` (computed by explicit differencing, so it stays
    accurate for near-coincident directions), ``cosh d − 1`` is written as a
    **sum of non-negative terms** — never a difference of huge ones:

    .. math::

        m = \\cosh d - 1
          = 2\\sinh^2(\\Delta r/2)
          + 2\\sinh r_i \\sinh r_j \\cdot \\tfrac{1}{4}\\lVert v_i - v_j\\rVert^2 ,

    then ``d = arccosh(1 + m) = log1p(m + √(m(m+2)))``, stable for all
    ``m ≥ 0``. No ``e^{2r}`` intermediate ever forms, so this is exact up to
    ``r ≈ 350`` (where ``sinh r_i · sinh r_j`` finally overflows ``float64``) —
    versus the ambient hyperboloid distance, which is destroyed by catastrophic
    cancellation already at ``r ≈ 18``.

    Parameters
    ----------
    r:
        ``(N,)`` hyperbolic radii ``≥ 0``.
    V:
        ``(N, D+1)`` unit vectors on ``S^D``.

    Returns
    -------
    ``(N, N)`` symmetric distance matrix with zero diagonal.

    Notes
    -----
    Forms an ``(N, N, D+1)`` difference tensor — ``O(N² D)`` memory. Fine for
    the moderate ``N`` used here; a chunked/fast path is a later concern.
    """
    dr = r[:, None] - r[None, :]
    chord2 = ((V[:, None, :] - V[None, :, :]) ** 2).sum(-1)
    m = (2.0 * np.sinh(0.5 * dr) ** 2
         + 0.5 * np.sinh(r)[:, None] * np.sinh(r)[None, :] * chord2)
    m = np.maximum(m, 0.0)
    # Floor inside the sqrt: arccosh(1+m) has d/dm → ∞ as m → 0 (coincident
    # points), so the bare sqrt gives inf·0 = NaN gradients on the diagonal and
    # at near-coincident pairs. 1e-30 bounds the derivative while leaving every
    # real distance unchanged.
    D = np.log1p(m + np.sqrt(m * (m + 2.0) + 1e-30))
    np.fill_diagonal(D, 0.0)
    return D


def polar_distances_torch(r: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Autograd-compatible version of :func:`polar_distances`.

    Differentiable in both ``r`` and ``V`` — the gradient path for a
    ``PolarRepresentation`` optimising on ``S^D × ℝ₊``.
    """
    dr = r[:, None] - r[None, :]
    chord2 = ((V[:, None, :] - V[None, :, :]) ** 2).sum(-1)
    m = (2.0 * torch.sinh(0.5 * dr) ** 2
         + 0.5 * torch.sinh(r)[:, None] * torch.sinh(r)[None, :] * chord2)
    m = m.clamp_min(0.0)
    # 1e-30 floor inside the sqrt keeps the gradient finite at m = 0 (coincident
    # points / diagonal), where arccosh(1+m) is otherwise inf·0 = NaN.
    return torch.log1p(m + torch.sqrt(m * (m + 2.0) + 1e-30))


def polar_distances_between_torch(
    r_i: torch.Tensor, v_i: torch.Tensor, r_j: torch.Tensor, v_j: torch.Tensor,
) -> torch.Tensor:
    """
    Elementwise sibling of :func:`polar_distances_torch`: distance between
    *paired* points rather than all pairs — the same stable law-of-cosines
    formula with the ``[:, None]``/``[None, :]`` all-pairs broadcasting
    dropped, so it costs ``O(len(r_i))`` instead of ``O(N²)``. Used by
    gather-based losses (negative-sampling ranking) via
    ``Representation.dist_between`` for the ``polar``/``tangent`` charts,
    whose ``dist()`` has no other elementwise primitive to reuse (unlike
    ball/hyperboloid/exact_polar, whose underlying ``geoopt`` manifold
    ``dist(x, y)`` is already elementwise/broadcastable).

    ``r_i``/``r_j`` and ``v_i``/``v_j`` must be broadcastable against each
    other (e.g. both ``(P,)``/``(P, D)`` for ``P`` distinct pairs, or
    ``(P, 1)``/``(P, 1, D)`` against ``(P, K)``/``(P, K, D)`` for a
    one-to-many batch, mirroring how ``dist_between`` is called for negatives).
    """
    dr = r_i - r_j
    chord2 = ((v_i - v_j) ** 2).sum(-1)
    m = (2.0 * torch.sinh(0.5 * dr) ** 2
         + 0.5 * torch.sinh(r_i) * torch.sinh(r_j) * chord2)
    m = m.clamp_min(0.0)
    return torch.log1p(m + torch.sqrt(m * (m + 2.0) + 1e-30))


_TINY = 1e-15


class WarpedPolarHyperboloid(geoopt.Manifold):
    """
    Geodesic-polar coordinates on H^{D+1}, carrying a warped-product metric.

    A point is the packed tensor ``x = [r, v]`` of shape ``(..., D+2)``: the
    radius ``r ≥ 0`` followed by the unit direction ``v ∈ S^D``. The metric is
    the warped product

    .. math::

        g_c = dr^2 + w_c(r)^2 g_{S^D},
        \\qquad w_c(r) = \\frac{\\sinh(\\sqrt{c}\\,r)}{\\sqrt{c}},

    i.e. ``ℝ₊ ×_{w_c} S^D``, where ``c`` is ``chart_curvature``. At the default
    ``c = 1`` the warp is ``sinh r`` and this is the **exact** metric of
    H^{D+1}. The warp couples the two factors at every ``c``, so they must live
    on **one** manifold (a single ``ManifoldParameter``) rather than on a
    ``Euclidean`` × ``Sphere`` pair — optimising that pair applies the *product*
    metric ``dr² + ⟨dv,dv⟩``, which drops the warp and drives the angular
    coordinate as if it were flat (under-driving at small radius, overshooting
    at large radius under a shared learning rate).

    With the warp restored, every ``RiemannianAdam`` step is a true Riemannian
    one *for this metric*: :meth:`egrad2rgrad` raises the index with
    ``G⁻¹ = diag(1, w_c⁻²·I)`` and :meth:`expmap` follows the exact geodesic.
    That makes the step **self-regulating** — the geometric length of an update
    is ``≈ lr`` at every radius, because the second moment normalises by the
    metric norm — which is what removes the large-radius blow-up the ambient
    charts suffer.

    **``chart_curvature`` sets the metric the optimiser steps under, not the
    geometry of the embedding.** ``g_c`` is a genuine constant-curvature ``−c``
    metric, but it is used here as a preconditioner: the *loss* keeps decoding
    the curvature ``−1`` distance of :func:`polar_distances_torch`, which is a
    function of ``(r, v)`` alone and does not involve ``c``. Lowering ``c``
    shrinks the warp, so a given angular gradient buys a larger change in the
    angular *coordinate* at large radius; ``c → 0`` approaches the flat warp
    ``w = r`` (the tangent chart's scaling), and ``c = 1`` is the exact metric,
    which at large radius can barely turn a point at all.

    Numerics: the exponential map is evaluated through ``tanh`` and a log-space
    radius increment, so no ``e^r`` or ``cosh r`` ever forms (the same
    stability trick as :func:`polar_distances`).

    Parameters
    ----------
    max_step:
        Cap on the length of a single retraction, measured in the variable
        ``ρ = √c·r`` the exponential map is evaluated in. A safety valve
        against optimiser transients: it bounds ``cosh s``, which would
        otherwise overflow float64 at ``s ≈ 710``. With this metric the natural
        step is ``≈ √c·lr`` in that variable, so it does not bind in normal use
        — it is not a tuning knob. A step whose length is non-finite leaves the
        point unchanged. (hypeGRL's engineering choice, not from a reference.)
    eps_warp:
        Floor on ``w_c(r)²`` in :meth:`egrad2rgrad`. Polar coordinates are
        genuinely singular at the origin (``v`` is undefined there); the floor
        keeps a node sitting at ``r = 0`` — a tree root, say — at a bounded,
        near-zero angular gradient so it drifts radially instead of producing
        ``NaN``.
    chart_curvature:
        The ``c`` above, ``> 0``. Default ``1.0``: the exact metric of the space
        the distances are measured in. ``c ≤ 0`` is rejected rather than
        silently treated as flat — the ``c → 0`` member is the tangent chart
        (:class:`~hypegrl.representations.tangent.TangentRepresentation`), which
        is a different parametrisation rather than a limit of this one.
    """

    name = "WarpedPolarHyperboloid"
    ndim = 1
    reversible = False

    def __init__(self, max_step: float = 30.0, eps_warp: float = 1e-12,
                 chart_curvature: float = 1.0):
        super().__init__()
        if chart_curvature <= 0:
            raise ValueError(
                f"chart_curvature must be positive; got {chart_curvature!r}. "
                "The c → 0 member of the family is the tangent chart "
                "(representation='tangent'), not this manifold.")
        self.max_step = max_step
        self.eps_warp = eps_warp
        self.chart_curvature = float(chart_curvature)
        self._sqrt_c = self.chart_curvature ** 0.5

    @staticmethod
    def _split(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Unpack ``[r, v]``; ``r`` keeps its trailing dim so it broadcasts."""
        return x[..., :1], x[..., 1:]

    def _warp(self, r: torch.Tensor) -> torch.Tensor:
        """``w_c(r) = sinh(√c·r)/√c`` — the metric's angular scale factor."""
        return torch.sinh(self._sqrt_c * r) / self._sqrt_c

    # ``ρ = √c·r`` rescales the metric by the constant ``1/c``:
    # ``g_c = (1/c)·[dρ² + sinh²(ρ)·g_{S^D}]``. A constant factor leaves
    # geodesics and Christoffel symbols alone, so the exponential map and the
    # geodesic distance can be evaluated by the ``c = 1`` formulas in ``ρ`` and
    # read back — only lengths carry the ``1/√c``. The angular coordinate is
    # untouched by the substitution; the radial component of a *tangent* vector
    # transforms with it.
    def _to_rho(self, x: torch.Tensor) -> torch.Tensor:
        r, v = self._split(x)
        return torch.cat([self._sqrt_c * r, v], dim=-1)

    def _from_rho(self, y: torch.Tensor) -> torch.Tensor:
        rho, v = self._split(y)
        return torch.cat([rho / self._sqrt_c, v], dim=-1)

    def _u_to_rho(self, u: torch.Tensor) -> torch.Tensor:
        u_r, u_v = self._split(u)
        return torch.cat([self._sqrt_c * u_r, u_v], dim=-1)

    def inner(self, x, u, v=None, *, keepdim=False):
        """``⟨u, w⟩_x = u_r w_r + w_c(r)² ⟨u_v, w_v⟩`` — the warped metric."""
        if v is None:
            v = u
        r, _ = self._split(x)
        u_r, u_v = self._split(u)
        w_r, w_v = self._split(v)
        warp = self._warp(r) ** 2
        out = u_r * w_r + warp * (u_v * w_v).sum(dim=-1, keepdim=True)
        return out if keepdim else out.squeeze(-1)

    def proju(self, x, u):
        """Keep the radial part; make the angular part tangent to ``S^D``."""
        _, v = self._split(x)
        u_r, u_v = self._split(u)
        u_v = u_v - (u_v * v).sum(dim=-1, keepdim=True) * v
        return torch.cat([u_r, u_v], dim=-1)

    def egrad2rgrad(self, x, u):
        """
        The natural gradient ``G⁻¹ P(e)``: project the angular part onto
        ``T_v S^D``, then divide it by the warp ``w_c(r)²``.

        That division is the whole difference from the product metric. It shrinks
        the angular gradient at large radius, compensating exactly for the fact
        that a unit angular coordinate move there is a huge geodesic
        displacement — by ``sinh²r`` at the default ``chart_curvature = 1``, and
        by a milder factor for smaller ``c``.
        """
        r, v = self._split(x)
        e_r, e_v = self._split(u)
        e_v = e_v - (e_v * v).sum(dim=-1, keepdim=True) * v
        warp = (self._warp(r) ** 2).clamp_min(self.eps_warp)
        return torch.cat([e_r, e_v / warp], dim=-1)

    def projx(self, x):
        r, v = self._split(x)
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(_TINY)
        return torch.cat([r.clamp_min(0.0), v], dim=-1)

    def expmap(self, x, u):
        """
        The exact geodesic of the metric ``g_c``, read off the unit-curvature one.

        Since ``g_c = (1/c)·g₁`` in ``ρ = √c·r`` and a constant rescale of a
        metric leaves its geodesics unchanged, the curve is
        :meth:`_expmap_unit` run in ``ρ`` and converted back. The tangent
        transforms with the coordinate, so its radial component enters as
        ``√c·u_r`` while the angular part is unchanged.
        """
        return self._from_rho(
            self._expmap_unit(self._to_rho(x), self._u_to_rho(u)))

    def _expmap_unit(self, x, u):
        """
        The unit-curvature geodesic in ``(ρ, v)``, in a form that never builds
        ``e^ρ``.

        The hyperboloid geodesic through ``Φ(ρ,v) = (cosh ρ, sinh ρ·v)`` read
        back in polar coordinates gives ``cosh ρ₁`` and ``sinh ρ₁·v₁`` as
        combinations of ``cosh ρ₀``/``sinh ρ₀``. Dividing both by ``cosh ρ₀``
        leaves only ``tanh ρ₀``-weighted ``O(1)`` quantities::

            C = cosh s + sinhc(s)·u_ρ·tanh ρ₀              ( = cosh ρ₁ / cosh ρ₀ )
            n = (cosh s·tanh ρ₀ + sinhc(s)·u_ρ)·v₀
                + sinhc(s)·tanh ρ₀·u_v                     ( = sinh ρ₁ / cosh ρ₀ · v₁ )

        with ``s = ‖u‖_{g₁}``, so ``v₁ = n/‖n‖`` and the radius comes from the
        ratio ``e^{ρ₁}/e^{ρ₀} = (C + ‖n‖)/(1 + tanh ρ₀)`` as a log-space
        increment. A geodesic crossing the origin is handled for free:
        ``‖n‖ ≥ 0`` and the ``v₀`` coefficient flips sign, so ``v₁`` flips to
        the antipode.
        """
        r0, v0 = self._split(x)
        u_r, u_v = self._split(u)
        u_v = u_v - (u_v * v0).sum(dim=-1, keepdim=True) * v0
        sinh_r0 = torch.sinh(r0)
        tanh_r0 = torch.tanh(r0)

        # geodesic step length ‖u‖_{g₁}, capped at max_step (bounds cosh s below)
        s_raw = torch.sqrt(u_r ** 2 + (sinh_r0 * u_v.norm(dim=-1, keepdim=True)) ** 2)
        scale = (self.max_step / s_raw.clamp_min(_TINY)).clamp(max=1.0)
        u_r, u_v = u_r * scale, u_v * scale
        s = torch.sqrt(u_r ** 2 + (sinh_r0 * u_v.norm(dim=-1, keepdim=True)) ** 2)

        sinhc = torch.where(                       # sinh(s)/s, → 1 as s → 0
            s > _TINY, torch.sinh(s) / s.clamp_min(_TINY), torch.ones_like(s))
        cosh_s = torch.cosh(s)

        C = cosh_s + sinhc * u_r * tanh_r0
        n = (cosh_s * tanh_r0 + sinhc * u_r) * v0 + sinhc * tanh_r0 * u_v
        n_norm = n.norm(dim=-1, keepdim=True)

        v1 = n / n_norm.clamp_min(_TINY)
        dr = torch.log(C + n_norm) - torch.log1p(tanh_r0)
        r1 = (r0 + dr).clamp_min(0.0)

        y = torch.cat([r1, v1], dim=-1)
        # An already-overflowed tangent keeps the previous point, as StableLorentz does.
        ok = torch.isfinite(y).all(dim=-1, keepdim=True) & torch.isfinite(s_raw)
        return torch.where(ok, y, x)

    # RiemannianAdam moves points through retr_transp -> retr; the retraction is
    # the true geodesic, so the optimiser takes exact Riemannian steps.
    retr = expmap

    def transp(self, x, y, v):
        """
        Projection vector transport — re-project the tangent onto the tangent
        space at ``y``, as ``geoopt.Sphere`` does.

        Only Adam's momentum is transported, and a single step moves the point by
        ``O(lr)``, so the distortion against exact parallel transport is ``O(lr)``
        per step. Exact transport would additionally rescale the angular part by
        ``sinh r₀/sinh r₁`` (unbounded near the origin) and rotate the in-plane
        frame for oblique geodesics; neither touches the gradient or the step.
        """
        return self.proju(y, v)

    def dist(self, x, y, *, keepdim=False):
        """
        Geodesic distance **of this manifold's own metric** ``g_c``, via the same
        stable law of cosines as :func:`polar_distances` (``cosh d − 1`` as a sum
        of non-negative terms) evaluated in ``ρ = √c·r`` and scaled by
        ``1/√c``.

        At the default ``chart_curvature = 1`` this is the curvature ``−1``
        hyperbolic distance, so it is what a loss should decode. **For any other
        ``chart_curvature`` it is not**: ``g_c`` is a preconditioner, and the
        embedding still lives in the curvature ``−1`` space, whose distance is
        :func:`polar_distances_torch` / :func:`polar_distances_between_torch` as
        a function of ``(r, v)``. Those are what
        :class:`~hypegrl.representations.polar.CurvedPolarRepresentation`
        decodes; this method stays consistent with :meth:`inner` and
        :meth:`expmap` instead, as ``geoopt`` expects of a manifold.
        """
        rho0, v0 = self._split(self._to_rho(x))
        rho1, v1 = self._split(self._to_rho(y))
        chord2 = ((v0 - v1) ** 2).sum(dim=-1, keepdim=True)
        m = (2.0 * torch.sinh(0.5 * (rho0 - rho1)) ** 2
             + 0.5 * torch.sinh(rho0) * torch.sinh(rho1) * chord2).clamp_min(0.0)
        d = torch.log1p(m + torch.sqrt(m * (m + 2.0) + 1e-30)) / self._sqrt_c
        return d if keepdim else d.squeeze(-1)

    def _check_point_on_manifold(self, x, *, atol=1e-5, rtol=1e-5):
        r, v = self._split(x)
        if bool((r < -atol).any()):
            return False, "radius r must be non-negative"
        norm = v.norm(dim=-1)
        if not torch.allclose(norm, torch.ones_like(norm), atol=atol, rtol=rtol):
            return False, "direction v must have unit norm"
        return True, None

    def _check_vector_on_tangent(self, x, u, *, atol=1e-5, rtol=1e-5):
        _, v = self._split(x)
        _, u_v = self._split(u)
        radial = (u_v * v).sum(dim=-1)
        if not torch.allclose(radial, torch.zeros_like(radial), atol=atol, rtol=rtol):
            return False, "angular part of u must be orthogonal to v"
        return True, None


__all__ = [
    "polar_distances", "polar_distances_torch", "polar_distances_between_torch",
    "WarpedPolarHyperboloid",
]
