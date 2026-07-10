# -*- coding: utf-8 -*-
"""
Exact maps between hyperbolic charts, with polar ``(r, v)`` as the canonical hub.

Every map is **direct**: ``polar ↔ ball`` and ``polar ↔ hyperboloid`` never
route through the third chart, so a large-radius point converted
``polar → hyperboloid`` does not first saturate in the ball. Fidelity is
chart-limited by construction:

- **polar** is lossless (``r`` a plain number, ``v`` a unit vector);
- **ball** saturates past ``r ≈ 12`` (``tanh(r/2) → 1``);
- **hyperboloid** is exact to ``r ≈ 350`` (``cosh r`` overflow).

A point of H^{D+1} is: ball ``X ∈ ℝ^{D+1}``; hyperboloid ``H ∈ ℝ^{D+2}``
(``H[0]`` timelike); polar ``(r, v)`` with ``r ≥ 0`` and ``v ∈ S^D ⊂ ℝ^{D+1}``.
NumPy and torch (autograd) versions; curvature ``k = 1``.

``ball ↔ hyperboloid`` reuses the stereographic maps in
:mod:`hypegrl.manifolds.poincare`.
"""

from __future__ import annotations

import numpy as np
import torch

from hypegrl.manifolds.poincare import (
    lorentz_to_poincare,
    lorentz_to_poincare_torch,
    poincare_to_lorentz,
    poincare_to_lorentz_torch,
)

# ball <-> hyperboloid: stereographic maps already live in poincare.py; alias
# them under uniform names so the Representation layer has one import surface.
ball_to_hyperboloid = poincare_to_lorentz
hyperboloid_to_ball = lorentz_to_poincare
ball_to_hyperboloid_torch = poincare_to_lorentz_torch
hyperboloid_to_ball_torch = lorentz_to_poincare_torch


# ---------------------------------------------------------------------------
# Direction extraction with an origin guard
# ---------------------------------------------------------------------------
# The origin (``‖space‖ = 0``, i.e. ``r = 0``) has no defined direction. Left as
# the zero vector it makes a downstream ``Sphere`` projection divide by zero
# (``0/0 → NaN``), which poisons the whole distance matrix. Assign it the
# canonical ``e_0 = (1, 0, …)``: harmless, because the angular term vanishes at
# ``r = 0`` (``sinh r = 0``), so a point at the origin has the same distances to
# every other point whatever its nominal direction. Methods that place a node at
# the centre (e.g. HyperMap's root) rely on this.

def _origin_safe_units(space: np.ndarray, norms: np.ndarray, eps: float) -> np.ndarray:
    V = space / np.clip(norms[:, None], eps, None)
    zero = norms == 0.0
    if np.any(zero):
        V[zero] = 0.0
        V[zero, 0] = 1.0
    return V


def _origin_safe_units_torch(
    space: torch.Tensor, norms: torch.Tensor, eps: float,
) -> torch.Tensor:
    V = space / norms.clamp_min(eps).unsqueeze(1)
    zero = norms == 0.0
    if zero.any():
        V = V.clone()
        V[zero] = 0.0
        V[zero, 0] = 1.0
    return V


# ---------------------------------------------------------------------------
# polar  <-->  Poincaré ball   (unit-vector form; direct)
# ---------------------------------------------------------------------------

def polar_to_ball(r: np.ndarray, V: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """``(r, v) → X = tanh(r/2)·v``. Saturates for ``r ≳ 12`` (ball limit)."""
    rho = np.tanh(np.clip(r / 2.0, 0.0, None))
    X = rho[:, None] * V
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    mask = norms[:, 0] >= 1.0
    if mask.any():
        X[mask] = X[mask] / (norms[mask] + eps) * (1.0 - eps)
    return X


def ball_to_polar(X: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """``X → (r, v)`` with ``r = 2·artanh‖X‖``, ``v = X/‖X‖``."""
    norms = np.linalg.norm(X, axis=1)
    r = 2.0 * np.arctanh(np.clip(norms, 0.0, 1.0 - eps))
    V = _origin_safe_units(X, norms, eps)
    return r, V


def polar_to_ball_torch(
    r: torch.Tensor, V: torch.Tensor, eps: float = 1e-15,
) -> torch.Tensor:
    """Autograd-compatible :func:`polar_to_ball`."""
    rho = torch.tanh(r.clamp(min=0.0) / 2.0)
    X = rho.unsqueeze(1) * V
    norms = X.norm(dim=1, keepdim=True)
    mask = (norms >= 1.0).squeeze(1)
    if mask.any():
        X = X.clone()
        X[mask] = X[mask] / (norms[mask] + eps) * (1.0 - eps)
    return X


def ball_to_polar_torch(
    X: torch.Tensor, eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Autograd-compatible :func:`ball_to_polar`."""
    norms = X.norm(dim=1)
    r = 2.0 * torch.arctanh(norms.clamp(0.0, 1.0 - eps))
    V = _origin_safe_units_torch(X, norms, eps)
    return r, V


# ---------------------------------------------------------------------------
# polar  <-->  hyperboloid   (direct; no ball intermediate)
# ---------------------------------------------------------------------------

def polar_to_hyperboloid(r: np.ndarray, V: np.ndarray) -> np.ndarray:
    """``(r, v) → H = (cosh r, sinh r · v)``. Exact to ``r ≈ 350``."""
    time = np.cosh(r)[:, None]
    space = np.sinh(r)[:, None] * V
    return np.concatenate([time, space], axis=1)


def hyperboloid_to_polar(
    H: np.ndarray, eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    ``H → (r, v)`` via the spatial part: ``r = arcsinh‖x'‖``, ``v = x'/‖x'‖``.

    Reading ``r`` off ``‖x'‖ = sinh r`` (not ``x_0 = cosh r``) avoids leaning on
    the timelike coordinate, which is the one corrupted by cancellation.
    """
    xs = H[:, 1:]
    norms = np.linalg.norm(xs, axis=1)
    r = np.arcsinh(norms)
    V = _origin_safe_units(xs, norms, eps)
    return r, V


def polar_to_hyperboloid_torch(
    r: torch.Tensor, V: torch.Tensor,
) -> torch.Tensor:
    """Autograd-compatible :func:`polar_to_hyperboloid`."""
    time = torch.cosh(r).unsqueeze(1)
    space = torch.sinh(r).unsqueeze(1) * V
    return torch.cat([time, space], dim=1)


def hyperboloid_to_polar_torch(
    H: torch.Tensor, eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Autograd-compatible :func:`hyperboloid_to_polar`."""
    xs = H[:, 1:]
    norms = xs.norm(dim=1)
    r = torch.arcsinh(norms)
    V = _origin_safe_units_torch(xs, norms, eps)
    return r, V


__all__ = [
    "ball_to_hyperboloid", "hyperboloid_to_ball",
    "ball_to_hyperboloid_torch", "hyperboloid_to_ball_torch",
    "polar_to_ball", "ball_to_polar",
    "polar_to_ball_torch", "ball_to_polar_torch",
    "polar_to_hyperboloid", "hyperboloid_to_polar",
    "polar_to_hyperboloid_torch", "hyperboloid_to_polar_torch",
]
