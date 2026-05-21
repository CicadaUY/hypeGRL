# -*- coding: utf-8 -*-
"""
Poincaré ball manifold helpers.

Wraps ``geoopt.PoincareBall`` and provides coordinate conversion
utilities between the Poincaré ball and other common representations
used in hyperbolic graph embedding:

- H^2 polar coordinates (r, theta)  [HyperMap, Mercator]
- Hyperspherical coordinates         [d-dimensional generalization]
- Lorentz / hyperboloid model        [Nickel & Kiela 2018]
"""

from __future__ import annotations

import numpy as np
import torch
import geoopt

# Shared manifold instance (curvature c=1 by default)
POINCARE_BALL = geoopt.PoincareBall()


# ---------------------------------------------------------------------------
# H^2 polar  <-->  Poincaré disk  (2D)
# ---------------------------------------------------------------------------

def polar_to_poincare(
    thetas: np.ndarray,
    r: np.ndarray,
    zeta: float = 1.0,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Convert H^2 polar coordinates ``(r, theta)`` to Poincaré disk Cartesian.

    The isometric map gives:

    .. math::

        x_i = \\tanh\\!\\left(\\frac{\\zeta r_i}{2}\\right)
               \\begin{pmatrix} \\cos\\theta_i \\\\ \\sin\\theta_i \\end{pmatrix}

    Parameters
    ----------
    thetas:
        ``(N,)`` angular coordinates in ``[0, 2*pi)``.
    r:
        ``(N,)`` radial coordinates in H^2 (non-negative).
    zeta:
        Curvature parameter (default 1.0).
    eps:
        Safety margin: norms are clipped to ``1 - eps``.

    Returns
    -------
    ``(N, 2)`` Poincaré disk coordinates with norms strictly in ``(0, 1)``.
    """
    rho = np.tanh(np.clip(zeta * r / 2.0, 0.0, None))
    X   = np.stack([rho * np.cos(thetas), rho * np.sin(thetas)], axis=1)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    mask  = norms[:, 0] >= 1.0
    if mask.any():
        X[mask] = X[mask] / (norms[mask] + eps) * (1.0 - eps)
    return X


def poincare_to_polar(
    X: np.ndarray,
    zeta: float = 1.0,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert Poincaré disk Cartesian coordinates to H^2 polar.

    Parameters
    ----------
    X:
        ``(N, 2)`` Poincaré disk coordinates.
    zeta:
        Curvature parameter.
    eps:
        Numerical floor for norm computation.

    Returns
    -------
    thetas : ``(N,)`` angular coordinates in ``[0, 2*pi)``.
    r      : ``(N,)`` radial coordinates in H^2.
    """
    assert X.shape[1] == 2, "poincare_to_polar requires 2D embeddings."
    norms  = np.clip(np.linalg.norm(X, axis=1), eps, 1.0 - eps)
    r      = (2.0 / zeta) * np.arctanh(norms)
    thetas = np.arctan2(X[:, 1], X[:, 0]) % (2.0 * np.pi)
    return thetas, r


# ---------------------------------------------------------------------------
# Hyperspherical  <-->  Poincaré ball  (d-dimensional)
# ---------------------------------------------------------------------------

def hyperspherical_to_poincare(
    angles: np.ndarray,
    r: np.ndarray,
    zeta: float = 1.0,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Convert hyperspherical coordinates to Poincaré ball Cartesian (d-dim).

    Generalises :func:`polar_to_poincare` to arbitrary dimension ``d``.
    The ``d-1`` hyperspherical angles parametrise a unit vector on
    ``S^{d-1}``, and the radial coordinate is encoded in the Poincaré
    ball norm via :math:`\\rho = \\tanh(\\zeta r / 2)`.

    Hyperspherical convention (ISO physics, ``d-1`` angles):

    .. math::

        u_1 &= \\cos\\phi_1 \\\\
        u_2 &= \\sin\\phi_1 \\cos\\phi_2 \\\\
        &\\vdots \\\\
        u_{d-1} &= \\sin\\phi_1 \\cdots \\sin\\phi_{d-2} \\cos\\phi_{d-1} \\\\
        u_d     &= \\sin\\phi_1 \\cdots \\sin\\phi_{d-2} \\sin\\phi_{d-1}

    where :math:`\\phi_k \\in [0, \\pi)` for :math:`k < d-1` and
    :math:`\\phi_{d-1} \\in [0, 2\\pi)`.

    Parameters
    ----------
    angles:
        ``(N, d-1)`` array of hyperspherical angles.
    r:
        ``(N,)`` radial coordinates in H^d.
    zeta:
        Curvature parameter.
    eps:
        Safety margin from the disk boundary.

    Returns
    -------
    ``(N, d)`` Poincaré ball coordinates with row norms in ``(0, 1)``.
    """
    N, dm1 = angles.shape
    d      = dm1 + 1

    # Build unit vectors on S^{d-1} from hyperspherical angles
    U = np.ones((N, d))
    for k in range(dm1):
        U[:, k]  *= np.cos(angles[:, k])
        U[:, k+1:] = (U[:, k+1:].T * np.sin(angles[:, k])).T

    # Scale by tanh(zeta * r / 2)
    rho = np.tanh(np.clip(zeta * r / 2.0, 0.0, None))
    X   = U * rho[:, np.newaxis]

    # Clip to strictly inside ball
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    mask  = norms[:, 0] >= 1.0
    if mask.any():
        X[mask] = X[mask] / (norms[mask] + eps) * (1.0 - eps)
    return X


def poincare_to_hyperspherical(
    X: np.ndarray,
    zeta: float = 1.0,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert Poincaré ball Cartesian coordinates to hyperspherical.

    Parameters
    ----------
    X:
        ``(N, d)`` Poincaré ball coordinates.
    zeta:
        Curvature parameter.
    eps:
        Numerical floor.

    Returns
    -------
    angles : ``(N, d-1)`` hyperspherical angles.
    r      : ``(N,)`` radial coordinates in H^d.
    """
    N, d   = X.shape
    norms  = np.clip(np.linalg.norm(X, axis=1), eps, 1.0 - eps)
    r      = (2.0 / zeta) * np.arctanh(norms)
    U      = X / norms[:, np.newaxis]   # unit vectors on S^{d-1}

    # Recover angles from unit vectors
    angles = np.zeros((N, d - 1))
    for k in range(d - 2):
        nk = np.linalg.norm(U[:, k:], axis=1).clip(min=eps)
        angles[:, k] = np.arccos(np.clip(U[:, k] / nk, -1.0, 1.0))
    # Last angle: use atan2 for full [0, 2*pi) range
    angles[:, d - 2] = np.arctan2(U[:, d - 1], U[:, d - 2]) % (2.0 * np.pi)

    return angles, r


# ---------------------------------------------------------------------------
# Lorentz (hyperboloid)  <-->  Poincaré ball
# ---------------------------------------------------------------------------

def lorentz_to_poincare(
    H: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Map points from the Lorentz / hyperboloid model to the Poincaré ball.

    The stereographic projection from the hyperboloid
    :math:`\\mathcal{H}^d = \\{x \\in \\mathbb{R}^{d+1} : -x_0^2 + \\|x_{1:}\\|^2 = -1,\\, x_0 > 0\\}`
    to the Poincaré ball is:

    .. math::

        \\pi(x_0, x_{1:}) = \\frac{x_{1:}}{1 + x_0}

    Parameters
    ----------
    H:
        ``(N, d+1)`` points on the hyperboloid (first coordinate is timelike).
    eps:
        Safety margin from the ball boundary.

    Returns
    -------
    ``(N, d)`` Poincaré ball coordinates.
    """
    x0  = H[:, 0:1]
    xr  = H[:, 1:]
    X   = xr / (1.0 + x0)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    mask  = norms[:, 0] >= 1.0
    if mask.any():
        X[mask] = X[mask] / (norms[mask] + eps) * (1.0 - eps)
    return X


def poincare_to_lorentz(
    X: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Map points from the Poincaré ball to the Lorentz / hyperboloid model.

    The inverse stereographic projection is:

    .. math::

        \\pi^{-1}(x) = \\frac{1}{1 - \\|x\\|^2}
                       \\begin{pmatrix} 1 + \\|x\\|^2 \\\\ 2x \\end{pmatrix}

    Parameters
    ----------
    X:
        ``(N, d)`` Poincaré ball coordinates with row norms ``< 1``.
    eps:
        Numerical floor for norm computation.

    Returns
    -------
    ``(N, d+1)`` points on the unit hyperboloid.
    """
    norm2 = np.clip((X ** 2).sum(axis=1, keepdims=True), 0.0, 1.0 - eps)
    denom = 1.0 - norm2
    x0    = (1.0 + norm2) / denom          # (N, 1)
    xr    = 2.0 * X / denom                # (N, d)
    return np.concatenate([x0, xr], axis=1)


# ---------------------------------------------------------------------------
# Torch versions (autograd-compatible)
# ---------------------------------------------------------------------------

def polar_to_poincare_torch(
    thetas: torch.Tensor,
    r: torch.Tensor,
    zeta: float = 1.0,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Torch (autograd-compatible) version of :func:`polar_to_poincare`.

    Parameters
    ----------
    thetas:
        ``(N,)`` angular coordinates.
    r:
        ``(N,)`` radial coordinates.
    zeta:
        Curvature.
    eps:
        Safety margin.

    Returns
    -------
    ``(N, 2)`` Poincaré disk tensor.
    """
    rho = torch.tanh(zeta * r.clamp(min=0.0) / 2.0)
    X   = torch.stack([rho * torch.cos(thetas),
                       rho * torch.sin(thetas)], dim=1)
    norms = X.norm(dim=1, keepdim=True)
    mask  = (norms >= 1.0).squeeze(1)
    if mask.any():
        X[mask] = X[mask] / (norms[mask] + eps) * (1.0 - eps)
    return X


def hyperspherical_to_poincare_torch(
    angles: torch.Tensor,
    r: torch.Tensor,
    zeta: float = 1.0,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Torch (autograd-compatible) version of
    :func:`hyperspherical_to_poincare`.

    Parameters
    ----------
    angles:
        ``(N, d-1)`` hyperspherical angles.
    r:
        ``(N,)`` radial coordinates.
    zeta:
        Curvature.
    eps:
        Safety margin.

    Returns
    -------
    ``(N, d)`` Poincaré ball tensor.
    """
    N, dm1 = angles.shape
    d      = dm1 + 1

    U = torch.ones(N, d, dtype=angles.dtype, device=angles.device)
    for k in range(dm1):
        U[:, k]   = U[:, k] * torch.cos(angles[:, k])
        U[:, k+1:] = U[:, k+1:] * torch.sin(angles[:, k]).unsqueeze(1)

    rho = torch.tanh(zeta * r.clamp(min=0.0) / 2.0)
    X   = U * rho.unsqueeze(1)

    norms = X.norm(dim=1, keepdim=True)
    mask  = (norms >= 1.0).squeeze(1)
    if mask.any():
        X[mask] = X[mask] / (norms[mask] + eps) * (1.0 - eps)
    return X


def lorentz_to_poincare_torch(
    H: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Torch (autograd-compatible) version of :func:`lorentz_to_poincare`.

    Parameters
    ----------
    H:
        ``(N, d+1)`` hyperboloid points.
    eps:
        Safety margin.

    Returns
    -------
    ``(N, d)`` Poincaré ball tensor.
    """
    x0 = H[:, 0:1]
    xr = H[:, 1:]
    X  = xr / (1.0 + x0)
    norms = X.norm(dim=1, keepdim=True)
    mask  = (norms >= 1.0).squeeze(1)
    if mask.any():
        X[mask] = X[mask] / (norms[mask] + eps) * (1.0 - eps)
    return X


def poincare_to_lorentz_torch(
    X: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Torch (autograd-compatible) version of :func:`poincare_to_lorentz`.

    Parameters
    ----------
    X:
        ``(N, d)`` Poincaré ball coordinates.
    eps:
        Numerical floor.

    Returns
    -------
    ``(N, d+1)`` hyperboloid tensor.
    """
    norm2 = (X ** 2).sum(dim=1, keepdim=True).clamp(0.0, 1.0 - eps)
    denom = 1.0 - norm2
    x0    = (1.0 + norm2) / denom
    xr    = 2.0 * X / denom
    return torch.cat([x0, xr], dim=1)


__all__ = [
    "POINCARE_BALL",
    # NumPy
    "polar_to_poincare",
    "poincare_to_polar",
    "hyperspherical_to_poincare",
    "poincare_to_hyperspherical",
    "lorentz_to_poincare",
    "poincare_to_lorentz",
    # Torch
    "polar_to_poincare_torch",
    "hyperspherical_to_poincare_torch",
    "lorentz_to_poincare_torch",
    "poincare_to_lorentz_torch",
]
