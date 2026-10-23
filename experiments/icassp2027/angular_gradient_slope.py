"""How the stress's angular derivative scales with radius at the HYDRA warm start.

THE QUESTION. Under plain gradient descent with warp w, node i's angular step has arc
length dS_i = lr * |df/dtheta_i| * sinh(r_i) / w(r_i)^2. The ratio between the outermost
and innermost node then factors as

    (|df/dtheta|_out / |df/dtheta|_in) * (warp factor ~ e^{(1 - 2 sqrt c) s}),

with s = r_max - r_min. The second factor is the preconditioner's; the first belongs to
the loss and the configuration. If the first grows like e^{kappa s}, the warp that
balances the two moves from c = 1/4 to sqrt(c) = (1 + kappa) / 2. This script measures
kappa as the slope of log|df/dtheta_i| against r_i, at the warm start the paper refines.

WHAT TO EXPECT. Per pair, d_theta_i d_ij = sinh(r_i) * sin(alpha_ij), with alpha_ij the
angle at x_i of the geodesic triangle (o, x_i, x_j), so it is bounded by
min(sinh r_i, sinh r_j) and saturates for an outer node. That is a bound on each term,
not on the residual-weighted sum, so the slope is an empirical question.

Only H^2 (dim=2), where the direction is a single angle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from experiments.icassp2027.two_stage_chart_schedule import caterpillar, warm_start

RESULTS = Path(__file__).resolve().parents[1] / "results"


def pair_distances(r: torch.Tensor, theta: torch.Tensor, i, j) -> torch.Tensor:
    """Curvature -1 distance of the pairs (i, j) in H^2 polar coordinates.

    cosh d = cosh(r_i - r_j) + 2 sinh r_i sinh r_j sin^2((theta_i - theta_j) / 2), a sum
    of non-negative terms, so there is no cancellation at large radius.
    """
    half = (theta[i] - theta[j]) / 2
    x = torch.cosh(r[i] - r[j]) + 2 * torch.sinh(r[i]) * torch.sinh(r[j]) * torch.sin(half) ** 2
    return torch.arccosh(x)


def angular_gradient(r, theta, target) -> np.ndarray:
    """df/dtheta_i of the stress f = sum_{i<j} (d_ij - target_ij)^2, one entry per node."""
    r = torch.as_tensor(r, dtype=torch.float64)
    theta = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    target = torch.as_tensor(target, dtype=torch.float64)
    i, j = torch.triu_indices(len(r), len(r), offset=1)
    f = torch.sum((pair_distances(r, theta, i, j) - target[i, j]) ** 2)
    (grad,) = torch.autograd.grad(f, theta)
    return grad.numpy()


def fit_slope(r, grad) -> float:
    """Least-squares slope of log|grad| against r."""
    return float(np.polyfit(r, np.log(np.abs(grad)), 1)[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--spine", type=int, default=40)
    ap.add_argument("--leaves", type=int, default=4)
    args = ap.parse_args()

    G = caterpillar(args.spine, args.leaves)
    r, v, k, nodes = warm_start(G, curvature=1.0)
    D = nx.floyd_warshall_numpy(G, nodelist=nodes)
    theta = np.arctan2(v[:, 1], v[:, 0])
    grad = angular_gradient(r, theta, D * np.sqrt(k))

    on_spine = np.array([n < args.spine for n in nodes])
    outer = r >= np.median(r)
    result = dict(
        graph=f"caterpillar{args.spine}-{args.leaves}", curvature=k,
        r_min=float(r.min()), r_max=float(r.max()), span=float(r.max() - r.min()),
        kappa_all=fit_slope(r, grad),
        kappa_outer_half=fit_slope(r[outer], grad[outer]),
        kappa_inner_half=fit_slope(r[~outer], grad[~outer]),
        kappa_spine=fit_slope(r[on_spine], grad[on_spine]),
        kappa_leaves=fit_slope(r[~on_spine], grad[~on_spine]),
    )
    result["balanced_c"] = ((1 + result["kappa_all"]) / 2) ** 2
    stem = RESULTS / f"angular_gradient_slope_{result['graph']}"
    stem.with_suffix(".json").write_text(json.dumps(result, indent=2))
    np.savez(stem.with_suffix(".npz"), r=r, theta=theta, grad=grad, on_spine=on_spine)
    print(json.dumps(result, indent=2))

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.scatter(r[~on_spine], np.abs(grad[~on_spine]), s=10, c="tab:blue", label="hojas")
    ax.scatter(r[on_spine], np.abs(grad[on_spine]), s=14, c="tab:red", label="espina")
    rr = np.linspace(r.min(), r.max(), 50)
    b, a = np.polyfit(r, np.log(np.abs(grad)), 1)
    ax.plot(rr, np.exp(a + b * rr), "k--", lw=1, label=rf"ajuste: $\kappa={b:.2f}$")
    ax.set_yscale("log")
    ax.set_xlabel(r"$r_i$")
    ax.set_ylabel(r"$|\partial_{\theta_i} f|$")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"))


if __name__ == "__main__":
    main()
