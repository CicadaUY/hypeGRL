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


_TINY = 1e-15


class WarpedPolarHyperboloid(geoopt.Manifold):
    """
    H^{D+1} in geodesic-polar coordinates, with its **exact** warped metric.

    A point is the packed tensor ``x = [r, v]`` of shape ``(..., D+2)``: the
    radius ``r ≥ 0`` followed by the unit direction ``v ∈ S^D``. The metric is
    the warped product

    .. math::

        g = dr^2 + \\sinh^2(r)\\, g_{S^D},

    i.e. ``ℝ₊ ×_{sinh} S^D``. The ``sinh²r`` warp couples the two factors, so
    they must live on **one** manifold (a single ``ManifoldParameter``) rather
    than on a ``Euclidean`` × ``Sphere`` pair — optimising that pair applies the
    *product* metric ``dr² + ⟨dv,dv⟩``, which drops the warp and drives the
    angular coordinate as if it were flat (under-driving at small radius,
    overshooting at large radius under a shared learning rate).

    With the warp restored, every ``RiemannianAdam`` step is a true Riemannian
    one: :meth:`egrad2rgrad` raises the index with ``G⁻¹ = diag(1, sinh⁻²r·I)``
    and :meth:`expmap` follows the exact geodesic. That also makes the step
    **self-regulating** — the geometric length of an update is ``≈ lr`` at every
    radius, because the second moment normalises by the metric norm — which is
    what removes the large-radius blow-up the ambient charts suffer.

    Numerics: the exponential map is evaluated through ``tanh r`` and a
    log-space radius increment, so no ``e^r`` or ``cosh r`` ever forms (the same
    stability trick as :func:`polar_distances`). Curvature is fixed at ``k = 1``.

    Parameters
    ----------
    max_step:
        Cap on the geodesic length ``‖u‖_g`` of a single retraction. A safety
        valve against optimiser transients: it bounds ``cosh s``, which would
        otherwise overflow float64 at ``s ≈ 710``. With the exact metric the
        natural step is ``≈ lr``, so this does not bind in normal use — it is
        not a tuning knob. A step whose length is non-finite leaves the point
        unchanged. (hypeGRL's engineering choice, not from a reference.)
    eps_warp:
        Floor on ``sinh²r`` in :meth:`egrad2rgrad`. Polar coordinates are
        genuinely singular at the origin (``v`` is undefined there); the floor
        keeps a node sitting at ``r = 0`` — a tree root, say — at a bounded,
        near-zero angular gradient so it drifts radially instead of producing
        ``NaN``.
    """

    name = "WarpedPolarHyperboloid"
    ndim = 1
    reversible = False

    def __init__(self, max_step: float = 30.0, eps_warp: float = 1e-12):
        super().__init__()
        self.max_step = max_step
        self.eps_warp = eps_warp

    @staticmethod
    def _split(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Unpack ``[r, v]``; ``r`` keeps its trailing dim so it broadcasts."""
        return x[..., :1], x[..., 1:]

    def inner(self, x, u, v=None, *, keepdim=False):
        """``⟨u, w⟩_x = u_r w_r + sinh²(r) ⟨u_v, w_v⟩`` — the warped metric."""
        if v is None:
            v = u
        r, _ = self._split(x)
        u_r, u_v = self._split(u)
        w_r, w_v = self._split(v)
        warp = torch.sinh(r) ** 2
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
        ``T_v S^D``, then divide it by the warp ``sinh²r``.

        That division is the whole difference from the product metric. It shrinks
        the angular gradient at large radius, compensating exactly for the fact
        that a unit angular coordinate move there is a huge geodesic
        displacement.
        """
        r, v = self._split(x)
        e_r, e_v = self._split(u)
        e_v = e_v - (e_v * v).sum(dim=-1, keepdim=True) * v
        warp = (torch.sinh(r) ** 2).clamp_min(self.eps_warp)
        return torch.cat([e_r, e_v / warp], dim=-1)

    def projx(self, x):
        r, v = self._split(x)
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(_TINY)
        return torch.cat([r.clamp_min(0.0), v], dim=-1)

    def expmap(self, x, u):
        """
        The exact geodesic of H^{D+1}, in a form that never builds ``e^r``.

        The hyperboloid geodesic through ``Φ(r,v) = (cosh r, sinh r·v)`` read
        back in polar coordinates gives ``cosh r₁`` and ``sinh r₁·v₁`` as
        combinations of ``cosh r₀``/``sinh r₀``. Dividing both by ``cosh r₀``
        leaves only ``tanh r₀``-weighted ``O(1)`` quantities,

            C = cosh s + sinhc(s)·u_r·tanh r₀              ( = cosh r₁ / cosh r₀ )
            n = (cosh s·tanh r₀ + sinhc(s)·u_r)·v₀
                + sinhc(s)·tanh r₀·u_v                     ( = sinh r₁ / cosh r₀ · v₁ )

        with ``s = ‖u‖_g``, so ``v₁ = n/‖n‖`` and the radius comes from the ratio
        ``e^{r₁}/e^{r₀} = (C + ‖n‖)/(1 + tanh r₀)`` as a log-space increment.
        A geodesic crossing the origin is handled for free: ``‖n‖ ≥ 0`` and the
        ``v₀`` coefficient flips sign, so ``v₁`` flips to the antipode.
        """
        r0, v0 = self._split(x)
        u_r, u_v = self._split(u)
        u_v = u_v - (u_v * v0).sum(dim=-1, keepdim=True) * v0
        sinh_r0 = torch.sinh(r0)
        tanh_r0 = torch.tanh(r0)

        # geodesic step length ‖u‖_g, capped at max_step (bounds cosh s below)
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
        """Geodesic distance, via the same stable law of cosines as
        :func:`polar_distances` (``cosh d − 1`` as a sum of non-negative terms)."""
        r0, v0 = self._split(x)
        r1, v1 = self._split(y)
        chord2 = ((v0 - v1) ** 2).sum(dim=-1, keepdim=True)
        m = (2.0 * torch.sinh(0.5 * (r0 - r1)) ** 2
             + 0.5 * torch.sinh(r0) * torch.sinh(r1) * chord2).clamp_min(0.0)
        d = torch.log1p(m + torch.sqrt(m * (m + 2.0) + 1e-30))
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


__all__ = ["polar_distances", "polar_distances_torch", "WarpedPolarHyperboloid"]
