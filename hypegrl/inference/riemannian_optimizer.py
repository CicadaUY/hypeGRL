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
    X_init: np.ndarray = None,
    s_A: np.ndarray = None,
    loss_fn: Callable = None,
    manifold: geoopt.Manifold = None,
    lr: float = 1e-2,
    n_steps: int = 500,
    grad_clip: float = 10.0,
    log_every: int = 50,
    stabilize: int = 10,
    device: str = "cpu",
    representation=None,
) -> dict:
    """
    Minimise an encoder-decoder loss by RiemannianAdam, with ``s_A`` held fixed.

    Two modes, one loop:

    - **Legacy (single embedding on one manifold):** pass ``X_init`` and
      ``manifold``. ``X`` is wrapped as a ``ManifoldParameter`` and the loss is
      ``loss_fn(X, s_A)``.
    - **Representation:** pass ``representation`` (a
      :class:`hypegrl.representations.Representation`). Its ``parameters()``
      (which may span several manifolds) are optimised and the loss is
      ``loss_fn(representation, s_A)`` — the loss receives the encoder's output
      and lets the decoder pull whatever it needs from it (``rep.dist()`` for a
      distance decoder; a future inner-product accessor for an RDPG-style one),
      so the encoder-decoder contract ``L(s(A), decode(X))`` stays general and
      chart-agnostic.

    Exactly one mode must be selected.

    Parameters
    ----------
    X_init:
        ``(N, d)`` initial embeddings on ``manifold`` (legacy mode).
    s_A:
        ``(N, N)`` structural similarity ``s(A)``, precomputed once from the
        fixed graph. Converted to a float64 tensor on ``device``; never modified.
    loss_fn:
        Legacy: ``(X: Tensor, s_A: Tensor) -> scalar``.
        Representation: ``(rep: Representation, s_A: Tensor) -> scalar``.
    manifold:
        ``geoopt.Manifold`` for the legacy mode.
    representation:
        A ``Representation`` for the representation mode (mutually exclusive with
        ``X_init``/``manifold``).
    lr, n_steps, grad_clip, log_every, stabilize, device:
        RiemannianAdam settings; ``grad_clip=0`` disables clipping,
        ``log_every=0`` suppresses output.

    Returns
    -------
    dict with ``loss_history`` and, depending on mode, ``X`` (legacy, ``(N, d)``
    NumPy) or ``representation`` (optimised in place).
    """
    if representation is not None:
        if X_init is not None or manifold is not None:
            raise ValueError(
                "Pass either representation, or X_init+manifold — not both.")
    elif X_init is None or manifold is None:
        raise ValueError("Legacy mode requires both X_init and manifold.")

    device_ = torch.device(device)
    s_A_t = torch.tensor(s_A, dtype=torch.float64, device=device_)

    # ── Parameters + loss closure (the only per-mode difference) ──────────
    if representation is not None:
        params = representation.parameters()

        def closure() -> torch.Tensor:
            return loss_fn(representation, s_A_t)
    else:
        X = geoopt.ManifoldParameter(
            torch.tensor(X_init, dtype=torch.float64, device=device_),
            manifold=manifold,
        )
        params = [X]

        def closure() -> torch.Tensor:
            return loss_fn(X, s_A_t)

    # ── Training loop ────────────────────────────────────────────────────
    optimizer = geoopt.optim.RiemannianAdam(params, lr=lr, stabilize=stabilize)
    loss_history: list[float] = []

    for step in range(n_steps):
        optimizer.zero_grad()
        loss = closure()
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)
        if log_every > 0 and (step % log_every == 0 or step == n_steps - 1):
            print(f"Step {step:4d} | Loss: {loss_val:.6f}")

    # ── Results ───────────────────────────────────────────────────────────
    if representation is not None:
        return {"representation": representation, "loss_history": loss_history}
    return {"X": X.detach().cpu().numpy(), "loss_history": loss_history}
