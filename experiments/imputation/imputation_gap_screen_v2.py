"""Does imputing unknown edges help Poincaré Maps (polar chart)? Gap screen.

Hide a fraction ``f`` of the edges and ``f/5`` of the non-edges (the 5:1 ratio of the
original unbalanced protocol), then score the hidden pairs with

  zeroed     : fixed target Q(G minus the hidden edges)
  oracle     : fixed target Q(G) — trained on the hidden edges, an upper reference
  prior      : fixed target Q(A), hidden pairs frozen at the row/column-mean init
  joint      : a_Omega optimised jointly with the embedding (joint_optimize semantics)
  joint_slow : joint at lr_a = 1e-3, which is NOT converged at 3000 steps
  cn, aa     : common neighbours / Adamic-Adar on the zeroed graph

Every embedding arm starts from the SAME random init with the SAME budget (no warm
start). Embedding arms are scored by -distance; joint arms also by a_Omega
(``<arm>_aomega``). The optimisation loops copy riemannian_optimize (fixed targets)
and joint_optimize line for line, without their logging.

Usage:
  python imputation_gap_screen_v2.py <family> <N,N,...> <steps> <lr_x> <lr_a>
         [--seeds 0,1,2] [--dims 2,8] [--fracs 0.05,0.2,0.5] [--device cpu]
  family in {gsc (gamma 2.5), gsc21 (gamma 2.1), hk (Holme-Kim), air (European airports)}
One json row per cell, written as it goes; finished cells are skipped on restart.
"""
import argparse
import json
import time
from itertools import combinations
from pathlib import Path

import geoopt
import networkx as nx
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from hypegrl.embedders.poincare_maps import forest_matrix, symkl_loss_from_dist
from hypegrl.inference import imputation
from hypegrl.inference.joint_optimizer import build_adjacency, graph_to_tensor, logit_init
from hypegrl.representations import PolarRepresentation, build_representation

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
AIRPORTS = HERE.parent / "data" / "hubs" / "airports_europe.edgelist"
GSC_GAMMA = {"gsc": 2.5, "gsc21": 2.1}
GSC_BETAS = [1.5, 3.0]
LR_A_SLOW = 1e-3
ARMS = ("zeroed", "oracle", "prior", "joint", "joint_slow")


def make_graph(family, N, beta=None):
    """The test graph, with node order equal to label order (labels index matrices)."""
    if family == "hk":
        return nx.powerlaw_cluster_graph(N, 4, 0.5, seed=0)
    if family == "air":  # N is ignored: the vendored struc2vec graph has 399 nodes
        E = nx.read_edgelist(AIRPORTS, nodetype=int)
        G = nx.Graph()
        G.add_nodes_from(sorted(E.nodes()))
        G.add_edges_from(E.edges())
        return G
    G = nx.Graph(nx.geometric_soft_configuration_graph(
        beta=beta, n=N, gamma=GSC_GAMMA[family], mean_degree=12, seed=0))
    G.remove_edges_from(nx.selfloop_edges(G))
    G = G.subgraph(max(nx.connected_components(G), key=len))
    return nx.convert_node_labels_to_integers(G)


def hide(G, f, seed):
    """Hidden pairs: f/5 of the non-edges followed by f of the edges, without replacement."""
    rng = np.random.default_rng(seed)
    edges = list(G.edges())
    non_edges = [(u, v) for u, v in combinations(G, 2) if not G.has_edge(u, v)]
    n_con, n_dis = max(int(len(edges) * f), 1), max(int(len(non_edges) * f / 5), 1)
    return ([tuple(map(int, non_edges[i])) for i in rng.choice(len(non_edges), n_dis, replace=False)]
            + [tuple(map(int, edges[i])) for i in rng.choice(len(edges), n_con, replace=False)])


def fit(G, unknown, X_init, mode, steps, lr_x, lr_a, device="cpu"):
    """One arm. Returns (distance matrix, a_Omega or None, final loss)."""
    dev = torch.device(device)
    joint = mode.startswith("joint")
    rep = build_representation(PolarRepresentation, X_init, input_chart="ball", device=device)
    params = rep.parameters()
    a_raw = None
    if joint:
        A_known = graph_to_tensor(G, unknown, dev)
        a_raw = torch.tensor(logit_init(imputation.compute_a_omega_init(G, unknown)),
                             dtype=torch.float64, device=dev, requires_grad=True)
        opt = geoopt.optim.RiemannianAdam(
            [{"params": params, "lr": lr_x},
             {"params": [a_raw], "lr": LR_A_SLOW if mode == "joint_slow" else lr_a}])
        clip = list(params) + [a_raw]
    else:
        if mode == "oracle":
            A = torch.tensor(nx.to_numpy_array(G), dtype=torch.float64, device=dev)
        else:
            A = graph_to_tensor(G, unknown, dev)
            if mode == "prior":
                a0 = torch.tensor(imputation.compute_a_omega_init(G, unknown),
                                  dtype=torch.float64, device=dev)
                A = build_adjacency(A, unknown, a0)
        Q = forest_matrix(A)
        opt = geoopt.optim.RiemannianAdam(params, lr=lr_x, stabilize=10)
        clip = params
    for _ in range(steps):
        opt.zero_grad()
        if joint:
            Q = forest_matrix(build_adjacency(A_known, unknown, torch.sigmoid(a_raw)))
        loss = symkl_loss_from_dist(rep.dist(), Q, 1.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(clip, max_norm=10.0)
        opt.step()
    with torch.no_grad():
        D = rep.dist().cpu().numpy()
    a = torch.sigmoid(a_raw).detach().cpu().numpy() if a_raw is not None else None
    return D, a, loss.item()


def metrics(labels, score):
    return dict(auc=float(roc_auc_score(labels, score)),
                ap=float(average_precision_score(labels, score)))


def run_cell(G, d, f, seed, steps, lr_x, lr_a, device="cpu"):
    unknown = hide(G, f, seed)
    labels = np.array([int(G.has_edge(m, n)) for m, n in unknown])
    G0 = G.copy()
    G0.remove_edges_from(unknown)
    X_init = np.random.default_rng(seed).standard_normal((G.number_of_nodes(), d)) * 0.1
    row = dict(n_nodes=G.number_of_nodes(), n_pos=int(labels.sum()),
               n_neg=int(len(labels) - labels.sum()),
               components_zeroed=nx.number_connected_components(G0), final_loss={})
    for mode in ARMS:
        torch.manual_seed(seed)
        D, a, loss = fit(G, unknown, X_init, mode, steps, lr_x, lr_a, device)
        row[mode] = metrics(labels, -np.array([D[m, n] for m, n in unknown]))
        row["final_loss"][mode] = loss
        if a is not None:
            row[mode + "_aomega"] = metrics(labels, a)
    row["cn"] = metrics(labels, np.array(
        [len(list(nx.common_neighbors(G0, m, n))) for m, n in unknown], float))
    row["aa"] = metrics(labels, np.array([s for _, _, s in nx.adamic_adar_index(G0, unknown)]))
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("family", choices=["gsc", "gsc21", "hk", "air"])
    p.add_argument("sizes")
    p.add_argument("steps", type=int)
    p.add_argument("lr_x", type=float)
    p.add_argument("lr_a", type=float)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--dims", default=None, help="default 2,8 (air: 2)")
    p.add_argument("--fracs", default=None, help="default 0.05,0.2,0.5 (air: 0.05,0.2)")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    sizes = [int(n) for n in args.sizes.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    dims = [int(x) for x in (args.dims or ("2" if args.family == "air" else "2,8")).split(",")]
    fracs = [float(x) for x in (args.fracs or ("0.05,0.2" if args.family == "air"
                                                else "0.05,0.2,0.5")).split(",")]
    betas = GSC_BETAS if args.family in GSC_GAMMA else [None]
    out = RESULTS / (f"imputation_gap_screen_v2_{args.family}_N{'-'.join(map(str, sizes))}"
                     f"_{args.device}.json")

    rows = json.loads(out.read_text()) if out.exists() else []
    done = {(r["N"], r["beta"], r["d"], r["f"], r["seed"]) for r in rows}
    for N in sizes:
        for beta in betas:
            G = make_graph(args.family, N, beta)
            for d in dims:
                for f in fracs:
                    for seed in seeds:
                        if (N, beta, d, f, seed) in done:
                            continue
                        t0 = time.perf_counter()
                        row = dict(family=args.family, N=N, beta=beta, d=d, f=f, seed=seed,
                                   steps=args.steps, lr_x=args.lr_x, lr_a=args.lr_a,
                                   lr_a_slow=LR_A_SLOW,
                                   **run_cell(G, d, f, seed, args.steps, args.lr_x, args.lr_a,
                                              args.device))
                        row["seconds"] = time.perf_counter() - t0
                        rows.append(row)
                        out.write_text(json.dumps(rows, indent=1))
                        print(f"{args.family} N={N} beta={beta} d={d} f={f} s={seed} AUC "
                              + " ".join(f"{k}={row[k]['auc']:.3f}" for k in
                                         ARMS + ("joint_aomega", "cn", "aa"))
                              + f" ({row['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
