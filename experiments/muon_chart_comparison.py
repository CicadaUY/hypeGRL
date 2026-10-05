"""Muon and a hyperbolic Muon on the tangent chart, against Adam, on the stress problem.

THE QUESTION. The two-stage experiment treats the metric an optimiser steps under as a
preconditioner, separate from the curvature -1 geometry the stress is measured in.
Muon is the other well-known "choose the norm" optimiser: steepest descent under the
spectral norm, whose step is the orthogonal (polar) factor ``U Vᵀ`` of the momentum,
approximated by a Newton-Schulz iteration.
This runs both ideas on the same problem, the tangent chart ``Z ∈ R^{N×d}`` from the
same HYDRA warm start, with every arm's learning rate swept.

THE ARMS.
  tangent        Adam on ``Z`` -- the chart as the paper runs it, and the control.
  muon           Muon on the full ``N×d`` gradient of ``Z``.
  hmuon_c=<c>    "Hyperbolic Muon": steepest descent under the spectral norm of the
                 update *measured in the chart metric* ``g_c = dr² + w_c(r)² dθ²``.
                 Each node's row is whitened by ``G_c(z_i)^{1/2}``, the matrix is
                 orthogonalised, and the result is mapped back by ``G_c^{-1/2}``. This
                 is the lift of a unitarily invariant norm through a Riemannian
                 metric used by Intrinsic Muon (Li et al., arXiv:2605.09238);
                 applying it to the warp family of the paper is ours. ``c → 0``
                 recovers ``muon`` exactly.
  normgd         The ablation: Muon's momentum and step scale, but the direction is
                 only divided by its spectral norm, not orthogonalised. ``muon`` minus
                 ``normgd`` is what orthogonalisation buys beyond normalisation.

THE PREDICTION THE DESIGN IS BUILT AROUND (our derivation). For a tall ``N×d`` matrix
the polar factor is ``D (DᵀD)^{-1/2}``, and the Newton-Schulz approximation of it is
``D·p(DᵀD)`` for a polynomial ``p``: either way, the gradient right-multiplied by one
global ``d×d`` matrix. Every node's step therefore keeps plain gradient descent's
proportions; Muon equalises the ``d`` global directions, not the nodes. The paper's
difficulty is a spread of per-node step sizes across radii, which per-coordinate Adam
does equalise, so Muon is not expected to fix it -- and ``hmuon`` is the natural
gradient times a global ``d×d`` matrix, with the same limitation. ``muon`` losing to
``tangent`` here is a result, not a bug. (Pinned by
``test_muon_step_keeps_plain_gradient_node_proportions`` in
``tests/test_experiments.py``.)

IMPLEMENTATION CHOICES (ours, not from a reference).
  * Orthogonalisation is the exact polar factor ``U Vᵀ`` from an SVD by default.
    ``--orthogonalizer newton_schulz5`` selects Muon's own quintic Newton-Schulz
    iteration instead (five steps, Jordan's coefficients), which returns ``U f(Σ) Vᵀ``
    with ``f(σ)`` only roughly 1 -- Muon as it is used in practice. It runs in float64
    like the rest of the pipeline, where Jordan's runs in bfloat16. It is not faster
    here: at ``N×d = 200×2`` a full training step measured 1.42-1.44 ms with the SVD
    and 1.49-1.52 ms with Newton-Schulz (CUDA), both dominated by the ``N×N``
    distance matrix.
  * The step is ``lr · √max(N, d) · O``, which gives the update an RMS near 1, so a rate
    means roughly what it means for Adam. Moonshot's Muon uses ``0.2 · √max(N, d)``;
    the rate is swept either way.
  * Momentum 0.95 with Nesterov, as in Jordan's Muon; accumulated on the raw gradient of
    ``Z``, which lives in a flat space and so needs no transport.
  * No gradient clipping, unlike ``riemannian_optimize`` (clip at 10): the step is
    normalised, so clipping would only reweight the momentum average.

Output uses the two-stage runner's layout, so its plotter reads it:
    python two_stage_chart_schedule_plot.py caterpillar40-4 cuda \
        --prefix muon_chart_comparison

Usage:
    python muon_chart_comparison.py                               # caterpillar(40,4)
    python muon_chart_comparison.py --arms tangent,muon,hmuon_c=0.1,hmuon_c=0.3
    python muon_chart_comparison.py --n-coarse 6000 --lr-coarse 0.03   # two-stage
"""
from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

import networkx as nx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.icassp2027 import two_stage_chart_schedule as two_stage  # noqa: E402
from hypegrl.embedders.hydra_plus import _stress_loss_from_dist  # noqa: E402
from hypegrl.representations import TangentRepresentation  # noqa: E402

RESULTS = two_stage.RESULTS
DEFAULT_DEVICE = two_stage.DEFAULT_DEVICE
RATES = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0]
ARMS = ["tangent", "muon", "hmuon_c=0.3", "normgd"]
N_STEPS = 30000
_TINY_SQ = 1e-30


def polar_factor(D: torch.Tensor) -> torch.Tensor:
    """The orthogonal factor ``U Vᵀ`` of ``D = U Σ Vᵀ`` -- Muon's ideal direction."""
    U, _, Vh = torch.linalg.svd(D, full_matrices=False)
    return U @ Vh


NS5_COEFFICIENTS = (3.4445, -4.7750, 2.0315)


def newton_schulz5(D: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate orthogonalisation of ``D`` -- Muon's update direction in practice.

    Jordan's quintic iteration ``X ← aX + (b·XXᵀ + c·(XXᵀ)²)X`` from ``X = D/‖D‖_F``
    (https://kellerjordan.github.io/posts/muon/). Each step maps every singular value
    ``σ`` to ``aσ + bσ³ + cσ⁵`` and leaves the singular vectors alone, so the result is
    ``U f(Σ) Vᵀ``; the coefficients maximise the slope at zero rather than converge,
    leaving the singular values in about ``[0.7, 1.2]``. A tall matrix is transposed
    first so that ``XXᵀ`` is the small ``d×d`` Gram matrix.
    """
    a, b, c = NS5_COEFFICIENTS
    tall = D.shape[-2] > D.shape[-1]
    X = D.mT if tall else D
    X = X / (torch.linalg.matrix_norm(X) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    return X.mT if tall else X


ORTHOGONALIZERS = {"svd": polar_factor, "newton_schulz5": newton_schulz5}


def _radius_direction(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(r, v) = (‖z‖, z/‖z‖)`` row-wise, floored as in ``TangentRepresentation``."""
    r = torch.sqrt((z ** 2).sum(-1) + _TINY_SQ)
    return r, z / r[:, None]


def angular_whitening(z: torch.Tensor, chart_curvature: float) -> torch.Tensor:
    """``r / w_c(r)`` per node, with ``w_c(r) = sinh(√c·r)/√c`` (→ 1 at the origin).

    In ``z`` coordinates the flat metric is ``dr² + r² dθ²`` and the chart metric is
    ``dr² + w_c(r)² dθ²``, so ``G_c(z) = P_r + (w_c/r)² P_⊥`` and this is the angular
    entry of ``G_c(z)^{-1/2}``.
    """
    r, _ = _radius_direction(z)
    x = np.sqrt(chart_curvature) * r
    return torch.where(x > 1e-12, x / torch.sinh(x.clamp_min(1e-12)),
                       torch.ones_like(x))


def apply_metric_root(z: torch.Tensor, X: torch.Tensor,
                      s: torch.Tensor) -> torch.Tensor:
    """``G_c(z_i)^{-1/2} X_i`` row-wise: keep the radial part, scale the angular by
    ``s``."""
    _, v = _radius_direction(z)
    radial = (X * v).sum(-1, keepdim=True) * v
    return radial + s[:, None] * (X - radial)


class TangentMuon(torch.optim.Optimizer):
    """Muon, hyperbolic Muon or spectrally normalised GD on an ``(N, d)`` parameter.

    ``mode`` is ``"muon"``, ``"hmuon"`` (needs ``chart_curvature``) or ``"normgd"``;
    see the module docstring for what each one is and for the choices made here.
    """

    def __init__(self, params, lr: float, momentum: float = 0.95, nesterov: bool = True,
                 mode: str = "muon", chart_curvature: float | None = None,
                 orthogonalizer: str = "svd"):
        if mode not in ("muon", "hmuon", "normgd"):
            raise ValueError(f"unknown mode {mode!r}")
        if mode == "hmuon" and chart_curvature is None:
            raise ValueError("mode='hmuon' needs a chart_curvature")
        if orthogonalizer not in ORTHOGONALIZERS:
            raise ValueError(f"unknown orthogonalizer {orthogonalizer!r}; "
                             f"expected one of {sorted(ORTHOGONALIZERS)}")
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov))
        self.orthogonalize = ORTHOGONALIZERS[orthogonalizer]
        self.mode = mode
        self.chart_curvature = chart_curvature

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            mu = group["momentum"]
            for z in group["params"]:
                if z.grad is None:
                    continue
                g = z.grad
                buf = self.state[z].setdefault("momentum_buffer", torch.zeros_like(g))
                buf.mul_(mu).add_(g)
                D = g + mu * buf if group["nesterov"] else buf.clone()

                if self.mode == "hmuon":
                    s = angular_whitening(z, self.chart_curvature)
                    direction = apply_metric_root(
                        z, self.orthogonalize(apply_metric_root(z, D, s)), s)
                elif self.mode == "muon":
                    direction = self.orthogonalize(D)
                else:
                    direction = D / torch.linalg.matrix_norm(D, ord=2).clamp_min(1e-300)

                z.add_(direction, alpha=-group["lr"] * max(z.shape) ** 0.5)


def build_optimizer(arm: str, params, lr: float,
                    orthogonalizer: str = "svd") -> TangentMuon:
    """``"muon"``, ``"normgd"`` or ``"hmuon_c=<c>"``."""
    if arm.startswith("hmuon_c="):
        return TangentMuon(params, lr=lr, mode="hmuon",
                           chart_curvature=float(arm.split("=")[1]),
                           orthogonalizer=orthogonalizer)
    return TangentMuon(params, lr=lr, mode=arm, orthogonalizer=orthogonalizer)


def refine_arm(arm, r, v, target, mask, lr, n_steps, device: str = DEFAULT_DEVICE,
               orthogonalizer: str = "svd"):
    """One optimisation of the tangent chart: ``(final stress, rep, loss history)``.

    ``"tangent"`` is the two-stage runner's own Adam refinement. The others run the loop
    of ``riemannian_optimize`` with a :class:`TangentMuon` in place of RiemannianAdam.
    The final stress and the ``inf`` on divergence follow ``two_stage.refine``; a
    non-finite loss ends the run there, with the rest of the history ``nan``.
    """
    if arm == "tangent":
        return two_stage.refine("tangent", r, v, target, mask, lr, n_steps, device)
    rep = TangentRepresentation.from_polar(torch.as_tensor(r), torch.as_tensor(v),
                                           device=device)
    target_t = torch.as_tensor(target, dtype=torch.float64, device=device)
    optimizer = build_optimizer(arm, rep.parameters(), lr, orthogonalizer)
    history = np.empty(n_steps)
    for step in range(n_steps):
        optimizer.zero_grad()
        loss = _stress_loss_from_dist(rep.dist(), target_t, mask)
        if not torch.isfinite(loss):
            # Diverged: stop rather than keep stepping on non-finite values (normgd's
            # spectral norm is an SVD, which raises on them on CPU).
            history[step:] = np.nan
            break
        loss.backward()
        optimizer.step()
        history[step] = loss.item()
    if not np.isfinite(history[-100:]).all():
        return float("inf"), rep, history
    return float(history[-100:].mean()), rep, history


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", default="caterpillar(40,4)")
    ap.add_argument("--curvature", type=float, default=1.0,
                    help="curvature of the embedding space, for HYDRA's target scaling")
    ap.add_argument("--arms", default=",".join(ARMS),
                    help="comma-separated: tangent, muon, normgd, hmuon_c=<c>. Keep "
                         "'tangent': it is the control the table is read against.")
    ap.add_argument("--rates", default=",".join(str(x) for x in RATES))
    ap.add_argument("--n-steps", type=int, default=N_STEPS)
    ap.add_argument("--n-coarse", type=int, default=0,
                    help="steps of a tangent-Adam coarse phase before the arms; 0 (the "
                         "default) starts every arm from the HYDRA warm start")
    ap.add_argument("--lr-coarse", type=float, default=two_stage.LR_COARSE)
    ap.add_argument("--orthogonalizer", default="svd",
                    choices=["svd", "newton_schulz5"],
                    help="how muon/hmuon orthogonalise: the exact SVD (default) or "
                         "Muon's Newton-Schulz iteration")
    ap.add_argument("--device", default=DEFAULT_DEVICE)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    device = args.device
    rates = [float(x) for x in args.rates.split(",")]
    arms = args.arms.split(",")

    G = two_stage.load_graph(args.graph)
    n = G.number_of_nodes()
    D = np.array(nx.floyd_warshall_numpy(G), dtype=np.float64)
    mask = torch.as_tensor(np.triu(np.ones((n, n), dtype=bool), k=1)).to(device)
    r0, v0, k, nodes = two_stage.warm_start(G, args.curvature)
    A = nx.to_numpy_array(G, nodelist=nodes, weight=None)
    target = D * np.sqrt(k)

    print(f"=== {args.graph}  N={n}  diameter={nx.diameter(G)}  k={k:g}  "
          f"device={device} ===")
    print(f"warm start: r in [{r0.min():.2f}, {r0.max():.2f}]  "
          f"span {r0.max() - r0.min():.2f}", flush=True)

    RESULTS.mkdir(exist_ok=True)
    tag = Path(args.graph).stem.replace("(", "").replace(")", "").replace(",", "-")
    coarse_tag = f"_coarse{args.n_coarse}" if args.n_coarse else ""
    # Only the non-default orthogonaliser is named, so SVD runs keep their filenames.
    if args.orthogonalizer != "svd":
        coarse_tag += f"_{args.orthogonalizer}"
    stem = RESULTS / (f"muon_chart_comparison_{tag}_{device}{coarse_tag}"
                      + (f"_{args.tag}" if args.tag else ""))
    json_path, npz_path = f"{stem}.json", f"{stem}.npz"
    coords_path = f"{stem}_coords.npz"

    curves, coords, rows = {}, {}, []
    coarse_stress = None

    def save():
        json.dump(dict(graph=args.graph, n_nodes=n, curvature=k, rates=rates,
                       device=device, n_coarse=args.n_coarse, lr_coarse=args.lr_coarse,
                       coarse_chart="tangent", coarse_stress=coarse_stress,
                       n_fine=args.n_steps, arms=arms,
                       orthogonalizer=args.orthogonalizer, best=rows),
                  open(json_path, "w"), indent=1)
        np.savez(npz_path, **curves)
        np.savez(coords_path, nodes=np.asarray(nodes), **coords)

    r1, v1 = torch.as_tensor(r0), torch.as_tensor(v0)
    curves["coarse"] = np.empty(0)
    if args.n_coarse:
        coarse_stress, coarse_rep, coarse_history = two_stage.refine(
            "tangent", r0, v0, target, mask, args.lr_coarse, args.n_coarse, device)
        curves["coarse"] = coarse_history
        r1, v1 = coarse_rep.to_polar()
        coords["coarse"] = two_stage.as_polar_array(r1, v1)
        print(f"\ncoarse: tangent, lr={args.lr_coarse:g}, {args.n_coarse} steps  "
              f"{coarse_history[0]:.0f} -> {coarse_stress:.0f}", flush=True)
    save()

    print(f"\n{args.n_steps} steps per arm, {len(rates)} rates", flush=True)
    for arm in arms:
        def checkpoint(best_so_far, arm=arm):
            rows[:] = [r for r in rows if r["chart"] != arm] + [best_so_far]
            save()
        best = two_stage.sweep(arm, r1, v1, target, mask, A, rates, args.n_steps,
                               curves, coords, device, checkpoint,
                               refine=partial(refine_arm,
                                              orthogonalizer=args.orthogonalizer))
        checkpoint(best)

    control = next((r for r in rows if r["chart"] == "tangent"),
                   min(rows, key=lambda r: r["stress"]))
    print(f"\n{'arm':>12}{'best lr':>10}{'stress':>12}{'AP':>8}"
          f"{'vs ' + control['chart']:>12}")
    for row in sorted(rows, key=lambda r: r["stress"]):
        rel = row["stress"] / control["stress"]
        print(f"{row['chart']:>12}{row['lr']:>10g}{row['stress']:>12.1f}"
              f"{row['average_precision']:>8.4f}{rel:>11.2f}x"
              f"{'   [grid edge]' if row['on_grid_edge'] else ''}")
    print(f"\nwrote {json_path}")


if __name__ == "__main__":
    main()
