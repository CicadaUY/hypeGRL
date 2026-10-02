# experiments/

Reproduction scripts for the papers built on hypeGRL — one folder per paper,
plus the dataset loaders and per-graph descriptors they share.

**This is not part of the installed library.** `pip install hypegrl` does not
ship it, and it carries no API-stability contract. It sits next to the library
rather than inside `hypegrl/` because it pulls in heavy, niche dependencies that
the library itself deliberately avoids (RDPG/ASE spectral baselines, single-cell
and airport dataset formats, PyTorch Geometric). Everything here *uses* the
library through its public API (`hypegrl.embedders`, `hypegrl.evaluation`); the
library never imports back.

That separation is why this folder has its own `requirements.txt`: it lists the
extra packages the experiments need *on top of* an installed `hypegrl`, kept out
of the library's own dependency list so a plain install stays lean.

## Setup

```bash
pip install -e ".[dev]"            # the library, from the repo root
pip install -r experiments/requirements.txt   # experiment-only extras
```

## Layout

| Path | Tracked? | What it is |
|---|---|---|
| `datasets.py` | yes | Shared dataset loaders (single-cell k-NN graphs, airport networks, OpenFlights) |
| `graph_stats.py` | yes | Shared per-graph descriptors (e.g. Gromov `delta_mean`) |
| `hypegrl_paper/` | yes | The library paper: Table I link prediction (`run_table_i()`), the hierarchy ladder, ogbl-ddi, the geometry diagnostics, and their figures |
| `icassp2027/` | yes | The ICASSP 2027 paper: the two-stage curvature schedule under the stress loss (`two_stage_chart_schedule.py --graph`) |
| `exploratory/` | yes | Studies and sanity checks that informed the papers or the library's design notes but produce no number in either; may lag the library |
| `data/single_cell/` | yes | Small vendored CSVs (see that folder's README) |
| `data/phylogeny/` | yes | Small vendored Open Tree of Life clade trees (see that folder's README) |
| `data/` (other) | no | Download-on-demand caches (OpenFlights, torch_geometric Airports) — gitignored |
| `results/` | no | All run outputs, shared by every folder — gitignored, see below |

Scripts import each other as package modules (`from experiments.graph_stats import …`),
and the editable install exposes only `hypegrl`, not `experiments`. Run them as modules
from the repository root, which works for every script:
`python -m experiments.hypegrl_paper.link_prediction_experiment`.

## `results/` is scratch output — nothing in it is committed

The whole `results/` folder is gitignored. Everything in it is regenerable by
re-running the scripts:

- `table_i.md`, `table_i.json`, `table_i.log` — output of `run_table_i()`.
- `*.png` — diagnostic and paper figures.

Treat it as a scratchpad: safe to delete, never a source of truth. The canonical
place for a number or figure that matters is the paper (or `docs/`), not a
committed copy here — that keeps the repo free of large binaries and of tables
that would silently drift out of sync with the code that produces them.
