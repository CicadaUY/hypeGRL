"""Summarise the imputation quality screen with the split-free R_NX criterion.

Recomputes, from the coordinates the screen saved (no refitting), every arm's
R_NX(K) curve and its area AUC_lnK against each reference (see
``imputation_quality_screen``: Lee et al. 2013 Eq. (44), 2015 Eq. (17)). Prints the
paired difference of each arm against zeroed per family and dimension (mean, 95%
bootstrap CI, cells better than zeroed), writes the AUCs to json, and plots the mean
R_NX(K) curve of every arm against the primary reference (true latent geometry for
GSC, the paper's feature geodesics for Myeloid), pooled over seeds and, for GSC, beta.

Usage (from the repository root):
  python -m experiments.imputation.summarise_quality
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.imputation.imputation_quality_screen import (
    ARMS,
    RESULTS,
    auc_lnk,
    coranking,
    load,
    q_nx,
    r_nx,
    rank_matrix,
)
from hypegrl.manifolds.polar import polar_distances_torch

PRIMARY = {"gsc": "latent", "gsc21": "latent", "myeloid": "features"}
COMPARED = ("oracle", "prior", "em_prior", "joint", "joint_warm")
COLORS = {"zeroed": "#2a78d6", "prior": "#1baf7a", "em_prior": "#eda100",
          "joint": "#e87ba4", "joint_warm": "#008300",
          "oracle": "#6b6b67"}  # oracle: neutral reference
INK, MUTED, GRID = "#1f1f1e", "#6b6b67", "#e6e5e1"
CELL_KEYS = ("N", "beta", "d", "f", "seed")


def cell_id(row):
    """The screen's coordinate key prefix for a row."""
    return "|".join(f"{k}={row[k]}" for k in CELL_KEYS)


def embedding_distances(X):
    """Distances of saved polar coordinates X = [r, v]."""
    X = torch.as_tensor(X, dtype=torch.float64)
    return polar_distances_torch(X[:, 0], X[:, 1:]).numpy()


def rescore(family, rows, coords):
    """Per row: {arm: {ref: AUC_lnK}} and the arms' R_NX curves on the primary ref."""
    ref_ranks = {}
    scores, curves = [], []
    for row in rows:
        key = (row["N"], row["beta"])
        if key not in ref_ranks:
            _, refs = load(family, row["N"], row["beta"])
            ref_ranks[key] = {name: rank_matrix(D) for name, D in refs.items()}
        cell = cell_id(row)
        score, curve = {}, {}
        for arm in (a for a in ARMS if f"{cell}|{a}" in coords):
            R_emb = rank_matrix(embedding_distances(coords[f"{cell}|{arm}"]))
            score[arm] = {}
            for ref, R_ref in ref_ranks[key].items():
                rnx = r_nx(q_nx(coranking(R_ref, R_emb)))
                score[arm][ref] = auc_lnk(rnx)
                if ref == PRIMARY[family]:
                    curve[arm] = rnx
        scores.append(score)
        curves.append(curve)
    return scores, curves


def paired_table(family, rows, scores, rng):
    refs = list(scores[0]["zeroed"])
    for d in sorted({r["d"] for r in rows}) + ["all"]:
        idx = [i for i, r in enumerate(rows) if d == "all" or r["d"] == d]
        print(f"\n== {family}  d={d}  ({len(idx)} cells)  AUC_lnK(R_NX), "
              "Δ = arm − zeroed (mean [95% CI] cells better)")
        for ref in refs:
            z = np.array([scores[i]["zeroed"][ref] for i in idx])
            cols = [f"zeroed {z.mean():.3f}"]
            for arm in COMPARED:
                has = [i for i in idx if arm in scores[i]]  # an arm added later
                if not has:
                    continue
                dl = np.array([scores[i][arm][ref] - scores[i]["zeroed"][ref]
                               for i in has])
                boot = [rng.choice(dl, len(dl)).mean() for _ in range(2000)]
                lo, hi = np.percentile(boot, [2.5, 97.5])
                cols.append(f"{arm} {dl.mean():+.3f} [{lo:+.3f},{hi:+.3f}] "
                            f"{int((dl > 0).sum())}/{len(dl)}")
            print(f"  {ref:>8}: " + " | ".join(cols))


def plot(per_family):
    families = list(per_family)
    panels = sorted({(r["d"], r["f"]) for fam in families for r in per_family[fam][0]})
    size = (4 * len(panels), 3.3 * len(families))
    fig, axes = plt.subplots(len(families), len(panels), sharey="row", figsize=size,
                             squeeze=False)
    for i, fam in enumerate(families):
        rows, curves = per_family[fam]
        for j, (d, f) in enumerate(panels):
            ax = axes[i, j]
            idx = [k for k, r in enumerate(rows) if r["d"] == d and r["f"] == f]
            for arm in ("oracle",) + tuple(a for a in ARMS if a != "oracle"):
                has = [k for k in idx if arm in curves[k]]
                if not has:
                    continue
                # Pooled GSC giant components can differ by a node: common K range.
                n = min(len(curves[k][arm]) for k in has)
                mean = np.mean([curves[k][arm][:n] for k in has], axis=0)
                K = np.arange(1, len(mean) + 1)
                ax.plot(K, mean, color=COLORS[arm], lw=1.6,
                        ls="--" if arm == "oracle" else "-", label=arm)
            ax.set_xscale("log")
            ax.grid(True, color=GRID, lw=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.set_title(f"{fam}  d={d}  {int(f * 100)}% hidden  (n={len(idx)})",
                         fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(f"R_NX(K) vs {PRIMARY[rows[0]['family']]}", color=INK)
            if i == len(families) - 1:
                ax.set_xlabel("K (log scale)", color=INK)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Poincaré Maps (polar): mean R_NX(K) per arm "
                 "— 0 = random, 1 = perfect", color=INK)
    fig.tight_layout()
    out = RESULTS / "imputation_quality_screen_rnx.png"
    fig.savefig(out, dpi=130, facecolor="white")
    return out


def main():
    rng = np.random.default_rng(0)
    per_family, all_scores = {}, []
    for path in sorted(RESULTS.glob("imputation_quality_screen_*_cpu.json")):
        rows = json.loads(path.read_text())
        coords = dict(np.load(path.with_suffix(".npz")))
        # Skip a cell whose coordinates are not on disk yet (a run still writing);
        # an arm added to finished cells later is scored where present.
        rows = [r for r in rows if f"{cell_id(r)}|zeroed" in coords]
        family = rows[0]["family"]
        # Runs with connectivity-preserving hiding are a different protocol.
        label = family + (" connected" if rows[0].get("keep_connected") else "")
        scores, curves = rescore(family, rows, coords)
        paired_table(label, rows, scores, rng)
        per_family[label] = (rows, curves)
        all_scores += [dict(family=label, **{k: r[k] for k in CELL_KEYS}, auc_lnk=s)
                       for r, s in zip(rows, scores)]
    (RESULTS / "imputation_quality_screen_auclnk.json").write_text(
        json.dumps(all_scores, indent=1))
    print(f"\nfigure: {plot(per_family)}")


if __name__ == "__main__":
    main()
