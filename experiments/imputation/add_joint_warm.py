"""Add the ``joint_warm`` arm to quality-screen cells that were run without it.

Each cell's zeroed fit is rebuilt from the coordinates the screen saved (no refit), and
joint_optimize is started from it with the cell's own steps / lr_x / lr_a, exactly as
``run_cell`` does. The row and the .npz are updated after every cell; cells that
already have the arm are skipped, so the script can be stopped and rerun.

Usage (from the repository root, NOT while the screen is writing the same family):
  python -m experiments.imputation.add_joint_warm <family> <N,N,...> [--device cpu]
"""
import argparse
import json
import time

import numpy as np

from experiments.imputation.imputation_quality_screen import (
    RESULTS,
    cell_setup,
    fit_joint,
    joint_summary,
    load,
    score_arm,
)
from hypegrl.representations import PolarRepresentation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("family", choices=["gsc", "gsc21", "myeloid"])
    p.add_argument("sizes")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    stem = (f"imputation_quality_screen_{args.family}_N{args.sizes.replace(',', '-')}"
            f"_{args.device}")
    out_json, out_npz = RESULTS / f"{stem}.json", RESULTS / f"{stem}.npz"
    rows = json.loads(out_json.read_text())
    coords = dict(np.load(out_npz))

    graphs = {}
    for row in rows:
        if "joint_warm" in row:
            continue
        t0 = time.perf_counter()
        key = (row["N"], row["beta"])
        if key not in graphs:
            graphs[key] = load(args.family, row["N"], row["beta"])
        G, refs = graphs[key]
        setup = cell_setup(G, refs, row["f"], row["seed"], args.device)
        unknown, labels, _, _, Q_true, R_refs = setup
        cell = "|".join(f"{k}={row[k]}" for k in ("N", "beta", "d", "f", "seed"))
        X = coords[f"{cell}|zeroed"]
        zeroed = PolarRepresentation.from_polar(X[:, 0], X[:, 1:], device=args.device)
        rep, a = fit_joint(G, unknown, zeroed, row["steps"], row["lr_x"], row["lr_a"],
                           args.device)
        row.update(joint_summary("joint_warm", a, labels))
        metrics, X_warm = score_arm(rep, unknown, labels, Q_true, R_refs)
        row["joint_warm"], coords[f"{cell}|joint_warm"] = metrics, X_warm
        out_json.write_text(json.dumps(rows, indent=1))
        np.savez(out_npz, **coords)
        m = row["joint_warm_aomega_mean"]
        print(f"{args.family} {cell} joint_warm: a_Omega edges {m['edges']:.3f} "
              f"non-edges {m['non_edges']:.3f}, AUC {row['joint_warm_aomega_auc']:.3f} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
