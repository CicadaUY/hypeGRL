"""Does imputing unknown edges give Poincaré Maps a better *embedding*? Quality screen.

The gap screen scores the hidden pairs by ROC AUC, a per-edge question the method does
not claim to answer. Poincaré Maps (Klimovskaia et al., Nat. Commun. 11, 2966 (2020),
Methods, "Scale-independent quality measure") evaluates embeddings with the
scale-independent quality criteria of Lee & Verleysen (Neurocomputing 72, 1431 (2009)):
from the co-ranking matrix of a reference and an embedding distance,
Q_NX(K) is the mean fraction of K-nearest neighbours preserved,
K_max = argmax_K (Q_NX(K) − K/(N−1)), Q_local = mean_{K ≤ K_max} Q_NX(K) and
Q_global = mean_{K ≥ K_max} Q_NX(K). The functions below follow the official
implementation (``embedding_quality_score.py``: get_ranking, get_coRanking, get_score,
get_scalars), vectorised. K_max is chosen per embedding, so Q_local/Q_global of two
embeddings average over different K ranges; the split-free summary is the area under
R_NX(K) = ((N−1)·Q_NX(K) − K)/(N−1−K) on a log-K axis, AUC_lnK = Σ R_NX(K)/K ÷ Σ 1/K
over K = 1..N−2 (Lee et al., Neurocomputing 112 (2013) Eq. (44); Neurocomputing 169
(2015) Eq. (17)), also recorded.

Arms (hidden pairs = v2's ``hide``; each fit gets the same budget):
  zeroed     : fixed target Q(A), hidden pairs filled with 0
  oracle     : fixed target Q(A), hidden pairs filled with the truth (= the full graph)
  prior      : fixed target Q(A), hidden pairs at the row/column-mean init
  em_prior   : fixed target Q(A), hidden pairs at sigma(a − b·d_ij), a logistic fit of
               the known pairs' edge indicator on the previous fit's distances; starts
               from zeroed and repeats EM_ITERS times. Our heuristic, not from the
               paper.
  joint      : joint_optimize (a_Omega optimised with the embedding)
  joint_warm : joint_optimize started from the zeroed fit (the original protocol's
               order). The joint objective has a degenerate basin where a_Omega -> 1
               on hidden non-edges too; the random start can fall into it, the warm
               start from the zeroed geometry avoids it on Myeloid.
All arms but joint_warm start from the same random init; joint_warm continues from
zeroed, so it has seen twice the steps.

Per arm: Q_local / Q_global against each reference, the paper's loss (symmetric KL)
against the FULL graph's forest matrix, and the hidden-pair ROC AUC.
References:
  gsc, gsc21 : 'latent' — true H² distances of the generator's (radius, theta), taken as
               curvature −1 coordinates; 'graph' — hop distances on the full graph.
  myeloid    : 'features' — the paper's reference, weighted shortest paths of a k=20
               nearest-neighbour graph of the features; 'graph' — hop distances on the
               full k=15 graph that is embedded.
'graph' distances are integers with many ties; ties are broken by node index (stable
sort), as arbitrary as the official code's argsort, so read 'graph' scores as secondary.

Usage (from the repository root):
  python -m experiments.imputation.imputation_quality_screen <family> <N,N,...> <steps>
         <lr_x> <lr_a> [--seeds 0,1,2,3,4] [--dims 2,8] [--fracs 0.2,0.5] [--device cpu]
  family in {gsc, gsc21, myeloid} (N is ignored for myeloid)
Writes one json row per cell as it goes (finished cells are skipped on restart) and the
fitted polar coordinates of every arm to a matching .npz.
"""
import argparse
import json
import time

import networkx as nx
import numpy as np
import torch
from scipy.sparse import csgraph
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from experiments.datasets import single_cell_graph
from experiments.imputation.imputation_gap_screen_v2 import (
    GSC_BETAS,
    GSC_GAMMA,
    RESULTS,
    hide,
    make_graph,
)
from hypegrl.embedders.poincare_maps import forest_matrix, symkl_loss_from_dist
from hypegrl.inference import imputation
from hypegrl.inference.joint_optimizer import joint_optimize
from hypegrl.inference.riemannian_optimizer import riemannian_optimize
from hypegrl.manifolds.polar import polar_distances_torch
from hypegrl.representations import PolarRepresentation, build_representation

EM_ITERS = 3
ARMS = ("zeroed", "oracle", "prior", "em_prior", "joint", "joint_warm")


# ── Lee & Verleysen quality criteria ──────────────────────────────────────

def rank_matrix(D):
    """R[i, j] = rank of j among i's neighbours by D (self 0, others 1..N−1).

    Ties are broken by node index (stable sort)."""
    D = np.array(D, dtype=float)
    np.fill_diagonal(D, -np.inf)
    order = np.argsort(D, axis=1, kind="stable")
    R = np.empty_like(order)
    R[np.arange(len(D))[:, None], order] = np.arange(len(D))[None, :]
    return R


def coranking(R_ref, R_emb):
    """(N−1, N−1) co-ranking matrix: C[k−1, l−1] = #{(i, j): R_ref = k, R_emb = l}."""
    N = len(R_ref)
    off = ~np.eye(N, dtype=bool)
    k_ref, k_emb = R_ref[off] - 1, R_emb[off] - 1
    counts = np.bincount(k_ref * (N - 1) + k_emb, minlength=(N - 1) ** 2)
    return counts.reshape(N - 1, N - 1)


def q_nx(C):
    """Q_NX(K), K = 1..N−1: the K×K upper-left block of C, normalised by K·N."""
    N = len(C) + 1
    K = np.arange(1, N)
    return np.diagonal(C.cumsum(0).cumsum(1)) / (K * N)


def q_scalars(qnx):
    """(Q_local, Q_global, K_max) from Q_NX, as the official get_scalars."""
    n = len(qnx)
    k_max = int(np.argmax(qnx - np.arange(1, n + 1) / n))
    return float(qnx[: k_max + 1].mean()), float(qnx[k_max:].mean()), k_max


def r_nx(qnx):
    """R_NX(K), K = 1..N−2: Q_NX rescaled so random = 0 and perfect = 1.

    Lee, Renard, Bernard, Dupont & Verleysen, Neurocomputing 112, 92 (2013),
    Eq. (44)."""
    n = len(qnx)  # = N − 1
    K = np.arange(1, n)
    return (n * qnx[: n - 1] - K) / (n - K)


def auc_lnk(rnx):
    """Area under R_NX with K on a log scale (weights 1/K), K = 1..N−2.

    Lee, Peluffo-Ordóñez & Verleysen, Neurocomputing 169, 246 (2015), Eq. (17)."""
    w = 1.0 / np.arange(1, len(rnx) + 1)
    return float((rnx * w).sum() / w.sum())


def quality(R_ref, D_emb):
    qnx = q_nx(coranking(R_ref, rank_matrix(D_emb)))
    q_local, q_global, k_max = q_scalars(qnx)
    return dict(q_local=q_local, q_global=q_global, k_max=k_max,
                auc_lnk=auc_lnk(r_nx(qnx)))


# ── Graphs and their reference distances ──────────────────────────────────

def load(family, N, beta):
    """(graph to embed, {reference name: distance matrix})."""
    if family == "myeloid":
        knn = single_cell_graph("MyeloidProgenitors", k=15)
        # Unweighted topology; node order = label order (labels index matrices).
        G = nx.Graph()
        G.add_nodes_from(range(knn.number_of_nodes()))
        G.add_edges_from(knn.edges())
        knn20 = single_cell_graph("MyeloidProgenitors", k=20)
        W = nx.to_numpy_array(knn20, weight="weight")
        refs = {"features": csgraph.shortest_path(W, directed=False)}
    else:
        G = make_graph(family, N, beta)
        r = torch.tensor([G.nodes[i]["radius"] for i in G], dtype=torch.float64)
        th = torch.tensor([G.nodes[i]["theta"] for i in G], dtype=torch.float64)
        v = torch.stack([torch.cos(th), torch.sin(th)], dim=1)
        refs = {"latent": polar_distances_torch(r, v).numpy()}
    refs["graph"] = csgraph.shortest_path(nx.to_numpy_array(G), directed=False,
                                          unweighted=True)
    return G, refs


# ── Fits ──────────────────────────────────────────────────────────────────

def _loss(rep, Q):
    return symkl_loss_from_dist(rep.dist(), Q, 1.0)


def _polar(X_init, device):
    return build_representation(PolarRepresentation, X_init, input_chart="ball",
                                device=device)


def fit_fixed(A_filled, X_init, steps, lr_x, device):
    rep = _polar(X_init, device)
    riemannian_optimize(rep, forest_matrix(A_filled).cpu().numpy(), _loss, lr=lr_x,
                        n_steps=steps, log_every=0, device=device)
    return rep


def fit_joint(G, unknown, X_init, steps, lr_x, lr_a, device):
    rep = _polar(X_init, device)
    out = joint_optimize(G, rep, lambda rep_, A: _loss(rep_, forest_matrix(A)), unknown,
                         lr_X=lr_x, lr_a=lr_a, n_steps=steps, log_every=0,
                         device=device, verbose=False)
    return rep, out["a_omega"]


def filled(A_known, unknown, values):
    A = A_known.clone()
    m, n = torch.as_tensor(unknown, dtype=torch.long).unbind(dim=1)
    A[m, n] = A[n, m] = torch.as_tensor(values, dtype=A.dtype)
    return A


def logistic_fill(D, known_mask, A_full, unknown):
    """sigma(a − b·d) fitted on the known pairs, evaluated on the hidden ones."""
    iu = np.triu_indices_from(D, k=1)
    keep = known_mask[iu]
    model = LogisticRegression(C=1e6, max_iter=1000)
    model.fit(D[iu][keep].reshape(-1, 1), A_full[iu][keep])
    d_hidden = np.array([D[m, n] for m, n in unknown]).reshape(-1, 1)
    return model.predict_proba(d_hidden)[:, 1], float(-model.coef_[0, 0])


def joint_summary(name, a, labels):
    """Ranking AUC and calibration (mean on hidden edges / non-edges) of a_Omega."""
    a = np.asarray(a)
    return {f"{name}_aomega_auc": float(roc_auc_score(labels, a)),
            f"{name}_aomega_mean": dict(edges=float(a[labels == 1].mean()),
                                        non_edges=float(a[labels == 0].mean()))}


def score_arm(rep, unknown, labels, Q_true, R_refs):
    """(metrics, saved polar coordinates) of one fitted arm."""
    with torch.no_grad():
        D = rep.dist().cpu().numpy()
        symkl = _loss(rep, Q_true).item()
    d_hidden = np.array([D[m, n] for m, n in unknown])
    metrics = dict(auc=float(roc_auc_score(labels, -d_hidden)), symkl_true=symkl,
                   **{name: quality(R, D) for name, R in R_refs.items()})
    r, v = (t.cpu().numpy() for t in rep.to_polar())
    return metrics, np.concatenate([r[:, None], v], axis=1)


def cell_setup(G, refs, f, seed, device="cpu", keep_connected=False):
    """Hidden pairs, their labels, the adjacencies and the reference ranks of a cell."""
    unknown = hide(G, f, seed, keep_connected=keep_connected)
    labels = np.array([int(G.has_edge(m, n)) for m, n in unknown])
    A_full = torch.tensor(nx.to_numpy_array(G), dtype=torch.float64)
    A_known = filled(A_full, unknown, np.zeros(len(unknown)))
    Q_true = forest_matrix(A_full.to(device))
    R_refs = {name: rank_matrix(D) for name, D in refs.items()}
    return unknown, labels, A_full, A_known, Q_true, R_refs


def run_cell(G, refs, d, f, seed, steps, lr_x, lr_a, device="cpu",
             keep_connected=False):
    setup = cell_setup(G, refs, f, seed, device, keep_connected)
    unknown, labels, A_full, A_known, Q_true, R_refs = setup
    known_mask = np.ones(A_full.shape, dtype=bool)
    for m, n in unknown:
        known_mask[m, n] = known_mask[n, m] = False
    X_init = np.random.default_rng(seed).standard_normal((G.number_of_nodes(), d)) * 0.1

    reps, extra = {}, {}
    torch.manual_seed(seed)
    reps["zeroed"] = fit_fixed(A_known, X_init, steps, lr_x, device)
    reps["oracle"] = fit_fixed(A_full, X_init, steps, lr_x, device)
    a_prior = imputation.compute_a_omega_init(G, unknown)
    A_prior = filled(A_known, unknown, a_prior)
    reps["prior"] = fit_fixed(A_prior, X_init, steps, lr_x, device)
    rep, slopes = reps["zeroed"], []
    for _ in range(EM_ITERS):
        D = rep.dist().detach().cpu().numpy()
        a_fill, slope = logistic_fill(D, known_mask, A_full.numpy(), unknown)
        slopes.append(slope)
        rep = fit_fixed(filled(A_known, unknown, a_fill), X_init, steps, lr_x, device)
    reps["em_prior"], extra["em_slopes"] = rep, slopes
    reps["joint"], a_joint = fit_joint(G, unknown, X_init, steps, lr_x, lr_a, device)
    extra.update(joint_summary("joint", a_joint, labels))
    reps["joint_warm"], a_warm = fit_joint(G, unknown, reps["zeroed"], steps, lr_x,
                                           lr_a, device)
    extra.update(joint_summary("joint_warm", a_warm, labels))

    G0 = G.copy()
    G0.remove_edges_from(unknown)
    row = dict(n_nodes=G.number_of_nodes(), n_pos=int(labels.sum()),
               n_neg=int(len(labels) - labels.sum()),
               components_zeroed=nx.number_connected_components(G0), **extra)
    coords = {}
    for arm in ARMS:
        row[arm], coords[arm] = score_arm(reps[arm], unknown, labels, Q_true, R_refs)
    return row, coords


def main():
    p = argparse.ArgumentParser()
    p.add_argument("family", choices=["gsc", "gsc21", "myeloid"])
    p.add_argument("sizes")
    p.add_argument("steps", type=int)
    p.add_argument("lr_x", type=float)
    p.add_argument("lr_a", type=float)
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--dims", default="2,8")
    p.add_argument("--fracs", default="0.2,0.5")
    p.add_argument("--device", default="cpu")
    p.add_argument("--keep-connected", action="store_true",
                   help="hide only edges whose removal keeps the graph connected")
    args = p.parse_args()
    sizes = [int(n) for n in args.sizes.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    dims = [int(x) for x in args.dims.split(",")]
    fracs = [float(x) for x in args.fracs.split(",")]
    betas = GSC_BETAS if args.family in GSC_GAMMA else [None]
    stem = (f"imputation_quality_screen_{args.family}_N{'-'.join(map(str, sizes))}"
            f"{'_connected' if args.keep_connected else ''}_{args.device}")
    out_json, out_npz = RESULTS / f"{stem}.json", RESULTS / f"{stem}.npz"

    rows = json.loads(out_json.read_text()) if out_json.exists() else []
    coords = dict(np.load(out_npz)) if out_npz.exists() else {}
    done = {(r["N"], r["beta"], r["d"], r["f"], r["seed"]) for r in rows}
    for N in sizes:
        for beta in betas:
            G, refs = load(args.family, N, beta)
            for d in dims:
                for f in fracs:
                    for seed in seeds:
                        if (N, beta, d, f, seed) in done:
                            continue
                        t0 = time.perf_counter()
                        row, cell_coords = run_cell(G, refs, d, f, seed, args.steps,
                                                    args.lr_x, args.lr_a, args.device,
                                                    args.keep_connected)
                        cell = dict(N=N, beta=beta, d=d, f=f, seed=seed)
                        row = dict(family=args.family, **cell, steps=args.steps,
                                   lr_x=args.lr_x, lr_a=args.lr_a, em_iters=EM_ITERS,
                                   keep_connected=args.keep_connected,
                                   **row, seconds=time.perf_counter() - t0)
                        rows.append(row)
                        key = "|".join(f"{k}={v}" for k, v in cell.items())
                        for arm, X in cell_coords.items():
                            coords[f"{key}|{arm}"] = X
                        out_json.write_text(json.dumps(rows, indent=1))
                        np.savez(out_npz, **coords)
                        ref = "features" if args.family == "myeloid" else "latent"
                        q_glob = " ".join(f"{a}={row[a][ref]['q_global']:.3f}"
                                          for a in ARMS)
                        aucs = " ".join(f"{a}={row[a]['auc']:.3f}" for a in ARMS)
                        print(f"{args.family} {key} Q_global[{ref}] {q_glob}"
                              f"  AUC {aucs} ({row['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
