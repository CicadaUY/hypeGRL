# experiments/

Reproduction scripts for the hypeGRL paper — dataset loaders, baselines,
per-graph descriptors, and the figure/table runners.

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
| `datasets.py` | yes | Paper dataset loaders (single-cell k-NN graphs, airport networks, OpenFlights) |
| `link_prediction_experiment.py` | yes | Table I link-prediction runner (`run_table_i()`) |
| `graph_stats.py` | yes | Per-graph descriptors (e.g. Gromov `delta_mean`) reported alongside the tables |
| `data/single_cell/` | yes | Small vendored CSVs (see that folder's README) |
| `data/` (other) | no | Download-on-demand caches (OpenFlights, torch_geometric Airports) — gitignored |
| `results/` | no | All run outputs — gitignored, see below |

## `results/` is scratch output — nothing in it is committed

The whole `results/` folder is gitignored. Everything in it is regenerable by
re-running the scripts:

- `table_i.md`, `table_i.json`, `table_i.log` — output of `run_table_i()`.
- `*.png` — diagnostic and paper figures.

Treat it as a scratchpad: safe to delete, never a source of truth. The canonical
place for a number or figure that matters is the paper (or `docs/`), not a
committed copy here — that keeps the repo free of large binaries and of tables
that would silently drift out of sync with the code that produces them.
