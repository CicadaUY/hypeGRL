"""Run every experiment in the manifest, skipping the ones already complete.

Each run is one invocation of ``two_stage_chart_schedule``, in manifest order, so a
graph's stage 1 finishes before the stage-2 blocks that branch from it. Results go to
``experiments/results/`` under names derived from the manifest, so the table and
figure scripts find them without being told where. An interrupted run is redone from
its start: a stage-2 block writes after every rate but cannot resume mid-block.

The full set is long (see ``--dry-run``); ``--only`` runs a subset of graphs.

Run, from the repository root:
    python -m experiments.icassp2027.run_experiments --dry-run
    python -m experiments.icassp2027.run_experiments --only caterpillar falconiformes
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from experiments.icassp2027.manifest import GRAPHS, runs
from experiments.icassp2027.two_stage_chart_schedule import load_graph

REPO = Path(__file__).resolve().parents[2]

# Seconds per iteration on an RTX 3060 in float64, measured: a fixed latency plus the
# O(N^2) stress evaluation. Another GPU shifts both terms; treat the result as a guide.
SECONDS_PER_ITERATION = (1.499e-3, 5.545e-9)


def hours(n_nodes: int, iterations: int) -> float:
    latency, per_pair = SECONDS_PER_ITERATION
    return (latency + per_pair * n_nodes ** 2) * iterations / 3600


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=list(GRAPHS), metavar="GRAPH",
                    help=f"subset of graphs: {', '.join(GRAPHS)}")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would run and its estimated cost, run nothing")
    args = ap.parse_args()

    todo = [r for r in runs(args.only) if not r.done(args.device)]
    pending = {r.name for r in todo}
    n_nodes = {name: load_graph(GRAPHS[name].graph).number_of_nodes()
               for name in GRAPHS if name in pending}
    cost = {name: 0.0 for name in n_nodes}
    for r in todo:
        cost[r.name] += hours(n_nodes[r.name], r.iterations * len(r.rates))

    print(f"{len(todo)} runs to do, ~{sum(cost.values()):.1f} GPU-hours "
          f"(RTX 3060 estimate)")
    for name, h in cost.items():
        print(f"  {name:16s} N={n_nodes[name]:5d}  ~{h:5.1f} h")
    if args.dry_run:
        return

    for i, r in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {r.name} {r.kind} {r.curvature} "
              f"rates={list(r.rates)}", flush=True)
        subprocess.run([sys.executable, "-m",
                        "experiments.icassp2027.two_stage_chart_schedule",
                        *r.argv(args.device)], cwd=REPO, check=True)


if __name__ == "__main__":
    main()
