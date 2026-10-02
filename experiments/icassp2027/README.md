# ICASSP 2027: reproduction

Every number and figure in the paper's experimental section comes from the runs
listed in [`manifest.py`](manifest.py). Each run minimizes the Hydra+ stress from the
closed-form Hydra initialization by Riemannian Adam, with the curvature `c` of the
optimization metric as the only knob (`c = 1e-6` stands in for the flat limit `c -> 0`,
`c = 1` is the exact hyperbolic metric).

| File | What it does |
|---|---|
| `two_stage_chart_schedule.py` | The runner: one stage-1 sweep, or one stage-2 block, or one single-curvature run |
| `manifest.py` | Every run in the paper, its rate grid and budget, and the rule that picks the best rate |
| `run_experiments.py` | Runs the manifest, skipping what is already done |
| `make_table.py` | The results table, recomputed from the runs |
| `figures.py` | The two figures |
| `reference.py` | Writes `reference/` from the runs, and reads it back |
| `reference/` | The paper's results in compact form: the final stress of every run, and the plotted curves |

## Without running anything

`reference/` holds the paper's own results: `summary.csv` has the final stress of all
270 learning rates, and `curves.npz` the curve of each run's best rate. The table and
the figures are built from it exactly as from a full rerun:

```bash
python -m experiments.icassp2027.make_table --reference
python -m experiments.icassp2027.figures --reference
```

## Setup

From the repository root:

```bash
pip install -e ".[dev]"
pip install -r experiments/requirements.txt
```

The graphs are vendored in [`experiments/data/phylogeny/`](../data/phylogeny/) (the
four phylogenies) and [`experiments/data/hubs/`](../data/hubs/) (the two hub graphs);
`caterpillar(40,4)` is generated. Each data folder's README says where its files come
from and how to rebuild them byte-identically.

## Running the experiments

All commands run from the repository root. Results go to `experiments/results/`,
which is not tracked; compare your table with `make_table --reference`.

```bash
python -m experiments.icassp2027.run_experiments --dry-run      # what is left, and its cost
python -m experiments.icassp2027.run_experiments                # everything
python -m experiments.icassp2027.run_experiments --only caterpillar falconiformes
python -m experiments.icassp2027.make_table
python -m experiments.icassp2027.figures
```

The full set is 182 runs and about 86 GPU-hours on an RTX 3060, half of it cichlidae
(3134 nodes); `--dry-run` gives the estimate per graph. The runs are independent apart
from the second stage, which branches from the first stage's saved handover, and the
driver orders them accordingly. A run interrupted part-way is redone from its start.

## What to expect

- **Exact values depend on the device.** The pipeline has no randomness (the
  initialization is an eigendecomposition and the loss is full-batch), so a run is
  bit-for-bit reproducible on one device. Across devices it is not: a difference in
  the last bit at the first iteration grows to several percent over 10^5 iterations.
  The paper's numbers are from an NVIDIA RTX 3060, in double precision. Comparisons
  between curvatures are made on one device and are what should reproduce.
- **The rate grids are irregular in places.** Each grid was widened until its best
  rate had a worse or divergent neighbour on both sides, or until every rate gave the
  same stress; `manifest.py` records the grids that were finally run.
- **Two budgets differ, as the paper reports.** For cichlidae the single-curvature
  runs at `c` in {0.2, 0.3, 1} use 100k iterations, half its two-stage budget; its
  flat single-curvature run uses the full 200k at the best 100k rate.
