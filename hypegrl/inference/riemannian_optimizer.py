"""
Vanilla Riemannian gradient descent over node embeddings X.

Solves:
    min_{X in M^d} d( s_A, dec(X) )

where M^d is the embedding manifold (Poincaré ball, hyperboloid, or R^d),
s_A is the structural similarity matrix computed once from the fixed graph,
and dec(X) is the decoder output.

Use this when the structural similarity is fixed throughout training (no
unknown edges). The caller is responsible for computing s_A = s(A) before
calling this function and passing it as ``s_A``. For the joint case where
unknown edge weights are optimised simultaneously with X, see
``joint_optimizer.py``.
"""

from __future__ import annotations

from typing import Callable

import geoopt
import numpy as np
import torch


def riemannian_optimize(
    X_init: np.ndarray,
    s_A: np.ndarray,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    manifold: geoopt.Manifold,
    lr: float = 1e-2,
    n_steps: int = 500,
    grad_clip: float = 10.0,
    log_every: int = 50,
    stabilize: int = 10,
    device: str = "cpu",
) -> dict:
    """
    Optimise node embeddings ``X`` on a Riemannian manifold by minimising
    ``loss_fn(X, s_A)`` via RiemannianAdam, with ``s_A`` held fixed.

    Parameters
    ----------
    X_init:
        ``(N, d)`` initial embeddings. Must already lie on ``manifold``
        (e.g. inside the Poincaré disk).
    s_A:
        ``(N, N)`` structural similarity matrix, precomputed once from the
        fixed graph (e.g. forest matrix ``Q = (I+L)^{-1}``, shortest-path
        distance matrix ``D``). Converted to a float64 tensor on ``device``
        before the loop; never modified during optimisation.
    loss_fn:
        Callable ``(X: torch.Tensor, s_A: torch.Tensor) -> scalar tensor``
        computing the encoder-decoder loss. Matches the signature used by
        ``joint_optimize``, but here ``s_A`` is the precomputed structural
        similarity rather than the raw adjacency.
    manifold:
        A ``geoopt.Manifold`` instance defining the embedding geometry.
        Used to wrap ``X`` as a ``ManifoldParameter`` for ``RiemannianAdam``.
    lr:
        Learning rate for RiemannianAdam.
    n_steps:
        Number of gradient steps.
    grad_clip:
        Maximum gradient norm for clipping. Set to ``0`` to disable.
    log_every:
        Print loss every this many steps. Set to ``0`` to suppress output.
    stabilize:
        Re-project ``X`` onto the manifold every this many steps.
        Passed directly to ``geoopt.optim.RiemannianAdam``.
    device:
        Torch device string (``"cpu"``, ``"cuda"``, etc.).

    Returns
    -------
    dict with keys:

    - ``X``: ``(N, d)`` NumPy array of final embeddings.
    - ``loss_history``: list of scalar loss values, one per step.
    """
    device_ = torch.device(device)

    # ── Structural similarity (fixed, converted once) ─────────────────────
    s_A_t = torch.tensor(s_A, dtype=torch.float64, device=device_)

    # ── Embeddings (ManifoldParameter for RiemannianAdam) ────────────────
    X = geoopt.ManifoldParameter(
        torch.tensor(X_init, dtype=torch.float64, device=device_),
        manifold=manifold,
    )

    # ── Optimiser ────────────────────────────────────────────────────────
    optimizer = geoopt.optim.RiemannianAdam([X], lr=lr, stabilize=stabilize)

    # ── Training loop ────────────────────────────────────────────────────
    loss_history: list[float] = []

    for step in range(n_steps):
        optimizer.zero_grad()

        loss = loss_fn(X, s_A_t)
        loss.backward()

        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_([X], max_norm=grad_clip)

        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if log_every > 0 and (step % log_every == 0 or step == n_steps - 1):
            print(f"Step {step:4d} | Loss: {loss_val:.6f}")

    # ── Extract results ───────────────────────────────────────────────────
    return {
        "X":            X.detach().cpu().numpy(),
        "loss_history": loss_history,
    }
