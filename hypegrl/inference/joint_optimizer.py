"""
Joint optimization over embeddings X and unknown adjacency entries a_Omega.

Solves:
    min_{X in M^d, a_Omega} L(A, X)

where M^d is the embedding manifold (Poincaré disk, hyperboloid, or R^d),
L is the encoder-decoder loss, and a_Omega are the unknown edge weights,
reparametrized as sigmoid(a_omega_raw) to keep them in (0, 1).

The optimization is performed via RiemannianAdam (geoopt) for X and
standard Adam for a_omega_raw, with optional L2 regularization on a_Omega
for well-posedness when ``|Omega|`` is large or the graph is tree-like.
"""

from __future__ import annotations

from typing import Callable, Optional

import geoopt
import networkx as nx
import numpy as np
import torch

import hypegrl.inference.imputation as imputation


# ---------------------------------------------------------------------------
# Adjacency helpers
# ---------------------------------------------------------------------------

def build_adjacency(
    A_known: torch.Tensor,
    unknown_edges: list[tuple[int, int]],
    a_omega: torch.Tensor,
) -> torch.Tensor:
    """
    Assemble the full symmetric adjacency matrix from known entries and
    the current estimate of unknown edge weights.

    Parameters
    ----------
    A_known:
        ``(N, N)`` adjacency tensor with zeros at positions in ``unknown_edges``.
    unknown_edges:
        List of ``(m, n)`` pairs identifying unknown entries.
    a_omega:
        ``(|Omega|,)`` tensor of current unknown edge weight estimates,
        already passed through sigmoid (values in ``(0, 1)``).

    Returns
    -------
    ``(N, N)`` symmetric adjacency tensor with unknown entries filled in.
    """
    A = A_known.clone()
    for k, (m, n) in enumerate(unknown_edges):
        A[m, n] = a_omega[k]
        A[n, m] = a_omega[k]
    return A


def graph_to_tensor(
    G: nx.Graph,
    unknown_edges: list[tuple[int, int]],
    device: torch.device,
) -> torch.Tensor:
    """
    Convert a NetworkX graph to a torch adjacency tensor with unknown
    entries zeroed out.

    Parameters
    ----------
    G:
        Input graph.
    unknown_edges:
        Edges whose weights are treated as unknown (zeroed in output).
    device:
        Target torch device.

    Returns
    -------
    ``(N, N)`` float64 tensor.
    """
    A_np = nx.to_numpy_array(G, dtype=np.float64)
    for (m, n) in unknown_edges:
        A_np[m, n] = 0.0
        A_np[n, m] = 0.0
    return torch.tensor(A_np, dtype=torch.float64, device=device)


# ---------------------------------------------------------------------------
# Sigmoid reparametrisation
# ---------------------------------------------------------------------------

def logit_init(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Map ``values`` in ``(0, 1)`` to the logit domain so that
    ``sigmoid(logit_init(values)) == values`` at initialisation.

    Parameters
    ----------
    values:
        Initial edge weight estimates, clipped to ``[eps, 1-eps]``.
    eps:
        Numerical safety margin away from the boundary.

    Returns
    -------
    Array of logit-transformed values.
    """
    v = np.clip(values, eps, 1.0 - eps)
    return np.log(v / (1.0 - v))


# ---------------------------------------------------------------------------
# Core optimisation loop
# ---------------------------------------------------------------------------

def joint_optimize(
    G: nx.Graph,
    representation,
    loss_fn: Callable,
    unknown_edges: Optional[list[tuple[int, int]]] = None,
    a_omega_init: Optional[np.ndarray] = None,
    lr_X: float = 1e-2,
    lr_a: float = 1e-2,
    n_steps: int = 500,
    regularize_a: float = 0.0,
    grad_clip: float = 10.0,
    log_every: int = 50,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    """
    Jointly optimise an embedding ``representation`` and unknown adjacency
    entries ``a_Omega`` by minimising ``loss_fn(representation, A)`` over both.

    Like :func:`~hypegrl.inference.riemannian_optimizer.riemannian_optimize` but
    the target ``A`` is *rebuilt every step* from the current ``a_Omega`` estimate
    rather than held fixed — this is what lets gradients reach the unknown edge
    weights. The loss receives the **representation** (the decoder pulls
    ``rep.dist()`` or whatever it needs) and the current adjacency ``A``.

    Parameters
    ----------
    G:
        Input graph. Edge weights are used when present; unknown edges
        are zeroed out before optimisation.
    representation:
        A :class:`hypegrl.representations.Representation`. Its ``parameters()``
        are optimised jointly with ``a_Omega``; optimised in place.
    loss_fn:
        Callable ``(rep: Representation, A: Tensor) -> scalar`` computing the
        encoder-decoder loss. ``A`` is the ``(N, N)`` adjacency with the current
        ``a_Omega`` estimate filled in. The embedder provides this function.
    unknown_edges:
        List of ``(m, n)`` tuples for edges whose weights are unknown.
        If ``None`` or empty, no unknown entries are optimised and the
        problem reduces to standard embedding (``A`` is constant).
    a_omega_init:
        Initial values for unknown edge weights, in ``(0, 1)``.
        Defaults to the row/column-mean heuristic.
    lr_X:
        Learning rate for the Riemannian Adam update on the representation's
        parameters.
    lr_a:
        Learning rate for the Euclidean Adam update on ``a_omega_raw``.
    n_steps:
        Number of gradient steps.
    regularize_a:
        L2 regularisation coefficient on ``a_Omega``. Positive values
        improve well-posedness for tree-like graphs and reduce overfitting
        when ``|Omega|`` is large, at the cost of biasing imputed weights
        toward zero.
    grad_clip:
        Maximum gradient norm for clipping. Set to ``0`` to disable.
    log_every:
        Print loss every this many steps. Set to ``0`` to suppress output.
    device:
        Torch device string (``"cpu"``, ``"cuda"``, etc.).
    verbose:
        If ``False``, suppress all printed output regardless of
        ``log_every``.

    Returns
    -------
    dict with keys:

    - ``representation``: the same object, optimised in place.
    - ``a_omega``: ``(|Omega|,)`` NumPy array of imputed edge weights.
    - ``loss_history``: list of scalar loss values, one per step.
    - ``unknown_edges``: the input ``unknown_edges`` list (for reference).
    """
    device_ = torch.device(device)
    unknown_edges = unknown_edges or []

    # ── Adjacency ────────────────────────────────────────────────────────
    A_known = graph_to_tensor(G, unknown_edges, device_)

    # ── Unknown edge weights (sigmoid reparametrisation) ─────────────────
    if len(unknown_edges) > 0:
        if a_omega_init is None:
            a_omega_init = imputation.compute_a_omega_init(G, unknown_edges)
        a_omega_raw = torch.tensor(
            logit_init(a_omega_init),
            dtype=torch.float64, device=device_, requires_grad=True,
        )
    else:
        a_omega_raw = None

    # ── Optimiser (representation parameters + unknown weights) ───────────
    embed_params = representation.parameters()
    param_groups = [{"params": embed_params, "lr": lr_X}]
    if a_omega_raw is not None:
        param_groups.append({"params": [a_omega_raw], "lr": lr_a})
    optimizer = geoopt.optim.RiemannianAdam(param_groups)

    clip_params = list(embed_params) + (
        [a_omega_raw] if a_omega_raw is not None else []
    )

    # ── Training loop ────────────────────────────────────────────────────
    loss_history: list[float] = []

    for step in range(n_steps):
        optimizer.zero_grad()

        # Build adjacency with current a_Omega estimate
        if a_omega_raw is not None:
            a_omega = torch.sigmoid(a_omega_raw)
            A = build_adjacency(A_known, unknown_edges, a_omega)
        else:
            a_omega = torch.zeros(0, dtype=torch.float64, device=device_)
            A = A_known

        # Evaluate loss (embedder-provided; pulls what it needs from the rep)
        loss = loss_fn(representation, A)

        # Optional L2 regularisation on unknown weights
        if regularize_a > 0.0 and a_omega_raw is not None:
            loss = loss + regularize_a * (a_omega ** 2).sum()

        loss.backward()

        # Gradient clipping
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(clip_params, max_norm=grad_clip)

        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if verbose and log_every > 0 and (
            step % log_every == 0 or step == n_steps - 1
        ):
            a_str = (
                np.array2string(
                    a_omega.detach().cpu().numpy(),
                    precision=4, suppress_small=True, threshold=10
                )
                if a_omega_raw is not None
                else "[]"
            )
            print(f"Step {step:4d} | Loss: {loss_val:.6f} | a_Omega: {a_str}")

    # ── Extract results ───────────────────────────────────────────────────
    a_final = (
        torch.sigmoid(a_omega_raw).detach().cpu().numpy()
        if a_omega_raw is not None
        else np.array([])
    )

    return {
        "representation": representation,
        "a_omega":        a_final,
        "loss_history":   loss_history,
        "unknown_edges":  unknown_edges,
    }
