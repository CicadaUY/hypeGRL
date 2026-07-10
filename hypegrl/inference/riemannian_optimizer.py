"""
Vanilla Riemannian gradient descent over an embedding representation.

Solves:
    min_{rep} d( s_A, dec(rep) )

where ``rep`` is a :class:`hypegrl.representations.Representation` (the chart —
polar / ball / hyperboloid — in which the embedding is stored and optimised),
``s_A = s(A)`` is the structural similarity computed once from the fixed graph,
and ``dec(rep)`` is the decoder output (which pulls what it needs from the
representation, e.g. ``rep.dist()``).

Use this when the structural similarity is fixed throughout training (no unknown
edges). The caller computes ``s_A = s(A)`` before calling and passes it as
``s_A``. For the joint case where unknown edge weights are optimised
simultaneously, see ``joint_optimizer.py``.
"""

from __future__ import annotations

from typing import Callable

import geoopt
import numpy as np
import torch


def riemannian_optimize(
    representation,
    s_A: np.ndarray,
    loss_fn: Callable,
    lr: float = 1e-2,
    n_steps: int = 500,
    grad_clip: float = 10.0,
    log_every: int = 50,
    stabilize: int = 10,
    device: str = "cpu",
) -> dict:
    """
    Minimise an encoder-decoder loss by RiemannianAdam, with ``s_A`` held fixed.

    The representation's ``parameters()`` (which may span several manifolds — e.g.
    the polar chart optimises a radius on ``Euclidean`` and a direction on
    ``Sphere``) are optimised, and the loss is ``loss_fn(representation, s_A)``.
    The loss receives the **representation** (not a bare distance matrix) so the
    decoder pulls whatever it needs from it (``rep.dist()`` for a distance
    decoder; a future inner-product accessor for an RDPG-style one), keeping the
    encoder-decoder contract ``L(s(A), decode(X))`` general and chart-agnostic.

    Parameters
    ----------
    representation:
        A :class:`hypegrl.representations.Representation`. Optimised in place.
    s_A:
        ``(N, N)`` structural similarity ``s(A)``, precomputed once from the
        fixed graph. Converted to a float64 tensor on ``device``; never modified.
    loss_fn:
        ``(rep: Representation, s_A: Tensor) -> scalar``.
    lr, n_steps, grad_clip, log_every, stabilize, device:
        RiemannianAdam settings; ``grad_clip=0`` disables clipping,
        ``log_every=0`` suppresses output.

    Returns
    -------
    dict with ``representation`` (the same object, optimised in place) and
    ``loss_history``.
    """
    device_ = torch.device(device)
    s_A_t = torch.tensor(s_A, dtype=torch.float64, device=device_)

    params = representation.parameters()
    optimizer = geoopt.optim.RiemannianAdam(params, lr=lr, stabilize=stabilize)
    loss_history: list[float] = []

    for step in range(n_steps):
        optimizer.zero_grad()
        loss = loss_fn(representation, s_A_t)
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)
        if log_every > 0 and (step % log_every == 0 or step == n_steps - 1):
            print(f"Step {step:4d} | Loss: {loss_val:.6f}")

    return {"representation": representation, "loss_history": loss_history}
