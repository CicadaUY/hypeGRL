"""Convergence check for the imputation gap screen (Poincaré Maps, polar, d=2, 20% hidden).

All arms start from the SAME random init and get the SAME budget (no warm start):
  zeroed : fixed target Q(G minus hidden edges)   — riemannian_optimize semantics
  oracle : fixed target Q(G)                      — riemannian_optimize semantics
  joint  : target Q(A(a_Omega)) rebuilt per step   — joint_optimize semantics
over a grid of lr_x (and lr_a for joint). The loops copy the library's optimisers
line for line (RiemannianAdam, clip 10, stabilize=10 on the fixed-target path only);
they are re-written only to record the loss every step and the Omega scores at
checkpoints without resetting Adam's moments.

Usage: python imputation_convergence_check.py <cell>
       cell in {hk100, gsc100_b1.5, gsc100_b3}
"""
import json
import sys
import time

import geoopt
import networkx as nx
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from hypegrl.embedders.poincare_maps import forest_matrix, symkl_loss_from_dist
from hypegrl.inference import imputation
from hypegrl.inference.joint_optimizer import build_adjacency, graph_to_tensor, logit_init
from hypegrl.representations import PolarRepresentation, build_representation

from imputation_gap_screen_v2 import RESULTS, hide, make_graph

D_EMB, FRAC, SEED, GAMMA = 2, 0.2, 0, 1.0
T = 20000
CHECKPOINTS = [500, 1000, 2000, 3000, 5000, 10000, 20000]
LRS_X = [3e-3, 1e-2, 3e-2, 1e-1]
LRS_A = [1e-3, 1e-2, 1e-1]
CELLS = {"hk100": ("hk", 100, None),
         "gsc100_b1.5": ("gsc", 100, 1.5),
         "gsc100_b3": ("gsc", 100, 3.0)}


def run(G, unknown, labels, X_init, mode, lr_x, lr_a=None):
    """mode in {'zeroed', 'oracle', 'joint'}; returns loss curve + checkpoint scores."""
    rep = build_representation(PolarRepresentation, X_init, input_chart="ball", device="cpu")
    params = rep.parameters()
    if mode == "joint":
        A_known = graph_to_tensor(G, unknown, torch.device("cpu"))
        a_raw = torch.tensor(logit_init(imputation.compute_a_omega_init(G, unknown)),
                             dtype=torch.float64, requires_grad=True)
        opt = geoopt.optim.RiemannianAdam([{"params": params, "lr": lr_x},
                                           {"params": [a_raw], "lr": lr_a}])
        clip = list(params) + [a_raw]
    else:
        G_fit = G.copy()
        if mode == "zeroed":
            G_fit.remove_edges_from(unknown)
        Q = forest_matrix(torch.tensor(nx.to_numpy_array(G_fit), dtype=torch.float64))
        opt = geoopt.optim.RiemannianAdam(params, lr=lr_x, stabilize=10)
        clip = params
    loss_curve, checks = [], {}
    for step in range(1, T + 1):
        opt.zero_grad()
        if mode == "joint":
            a = torch.sigmoid(a_raw)
            loss = symkl_loss_from_dist(rep.dist(), forest_matrix(build_adjacency(A_known, unknown, a)), GAMMA)
        else:
            loss = symkl_loss_from_dist(rep.dist(), Q, GAMMA)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(clip, max_norm=10.0)
        opt.step()
        loss_curve.append(loss.item())
        if step in CHECKPOINTS:
            with torch.no_grad():
                D = rep.dist().numpy()
            d = np.array([D[m, n] for m, n in unknown])
            c = dict(auc=float(roc_auc_score(labels, -d)), loss=loss.item())
            if mode == "joint":
                av = torch.sigmoid(a_raw).detach().numpy()
                c.update(auc_a=float(roc_auc_score(labels, av)),
                         a_pos=float(av[labels == 1].mean()), a_neg=float(av[labels == 0].mean()))
            checks[step] = c
    return dict(loss_curve=loss_curve[::10], checkpoints=checks)


def main():
    cell = sys.argv[1]
    out = RESULTS / f"imputation_convergence_check_{cell}_cpu.json"
    G = make_graph(*CELLS[cell])
    unknown = hide(G, FRAC, SEED)
    labels = np.array([int(G.has_edge(m, n)) for m, n in unknown])
    X_init = np.random.default_rng(SEED).standard_normal((G.number_of_nodes(), D_EMB)) * 0.1
    rows = json.loads(out.read_text()) if out.exists() else []
    done = {(r["mode"], r["lr_x"], r["lr_a"]) for r in rows}
    arms = [(m, lx, None) for lx in LRS_X for m in ("zeroed", "oracle")]
    arms += [("joint", lx, la) for lx in LRS_X for la in LRS_A]
    for mode, lx, la in arms:
        if (mode, lx, la) in done:
            continue
        t0 = time.perf_counter()
        torch.manual_seed(SEED)
        r = run(G, unknown, labels, X_init, mode, lx, la)
        rows.append(dict(cell=cell, N=G.number_of_nodes(), E=G.number_of_edges(),
                         n_pos=int(labels.sum()), n_neg=int(len(labels) - labels.sum()),
                         mode=mode, lr_x=lx, lr_a=la, seconds=time.perf_counter() - t0, **r))
        out.write_text(json.dumps(rows))
        last = r["checkpoints"][T]
        print(f"{cell} {mode:>6} lr_x={lx:<6g} lr_a={la}  AUC@3k={r['checkpoints'][3000]['auc']:.3f} "
              f"AUC@20k={last['auc']:.3f}  loss@3k={r['checkpoints'][3000]['loss']:.2f} "
              f"loss@20k={last['loss']:.2f}  ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
