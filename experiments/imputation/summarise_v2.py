"""Summarise the imputation gap screen: per cell, mean ± sd over seeds and the paired
difference of each arm against zeroed (wins = seeds where the arm beats zeroed); then,
per graph family, the pooled paired difference with a 95% bootstrap CI."""
import json
from collections import defaultdict

import numpy as np

from imputation_gap_screen_v2 import RESULTS

ARMS = ["zeroed", "oracle", "prior", "joint", "joint_slow", "cn", "aa"]
COMPARED = ["prior", "joint", "joint_slow", "aa"]


def main():
    rows = []
    for f in sorted(RESULTS.glob("imputation_gap_screen_v2_*.json")):
        rows += json.loads(f.read_text())

    cells = defaultdict(list)
    for r in rows:
        cells[(r["family"], r["N"], r["beta"], r["d"], r["f"])].append(r)

    print(f"{'family':>6} {'N':>4} {'beta':>4} {'d':>2} {'f':>5} {'n':>2} | "
          + " ".join(f"{a:>11}" for a in ARMS)
          + " | Δ vs zeroed (mean, wins/n): " + "  ".join(COMPARED))
    for key in sorted(cells, key=lambda k: (k[0], k[1], k[2] or 0, k[3], k[4])):
        rs = cells[key]
        auc = {a: np.array([r[a]["auc"] for r in rs]) for a in ARMS}
        cols = " ".join(f"{auc[a].mean():.3f}±{auc[a].std():.3f}".rjust(11) for a in ARMS)
        deltas = []
        for a in COMPARED:
            dlt = auc[a] - auc["zeroed"]
            deltas.append(f"{dlt.mean():+.3f} {int((dlt > 0).sum())}/{len(dlt)}")
        fam, N, beta, d, f = key
        print(f"{fam:>6} {N:>4} {str(beta):>4} {d:>2} {f:>5} {len(rs):>2} | {cols} | "
              + "  ".join(deltas))

    print("\nPooled paired Δ vs zeroed per family (mean, 95% bootstrap CI, wins/n):")
    rng = np.random.default_rng(0)
    for fam in sorted({k[0] for k in cells}):
        fr = [r for r in rows if r["family"] == fam]
        for a in COMPARED + ["oracle"]:
            dlt = np.array([r[a]["auc"] - r["zeroed"]["auc"] for r in fr])
            boot = [rng.choice(dlt, len(dlt)).mean() for _ in range(2000)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            print(f"  {fam:>6} {a:>10}: {dlt.mean():+.3f}  [{lo:+.3f}, {hi:+.3f}]  "
                  f"{int((dlt > 0).sum())}/{len(dlt)}")


if __name__ == "__main__":
    main()
