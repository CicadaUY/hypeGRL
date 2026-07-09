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


__all__ = ["polar_distances", "polar_distances_torch"]
