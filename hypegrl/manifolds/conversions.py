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
    V = X / np.clip(norms[:, None], eps, None)
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
    V = X / norms.clamp_min(eps).unsqueeze(1)
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
    V = xs / np.clip(norms[:, None], eps, None)
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
    V = xs / norms.clamp_min(eps).unsqueeze(1)
    return r, V


__all__ = [
    "ball_to_hyperboloid", "hyperboloid_to_ball",
    "ball_to_hyperboloid_torch", "hyperboloid_to_ball_torch",
    "polar_to_ball", "ball_to_polar",
    "polar_to_ball_torch", "ball_to_polar_torch",
    "polar_to_hyperboloid", "hyperboloid_to_polar",
    "polar_to_hyperboloid_torch", "hyperboloid_to_polar_torch",
]
