# Single-cell expression datasets

Gene-regulatory / single-cell datasets used for the Table I link-prediction
experiments. Vendored from the [Poincaré Maps](https://github.com/facebookresearch/PoincareMaps)
repository (`datasets/`).

| CSV | Cells (nodes) | Notes |
|---|---|---|
| `ToggleSwitch.csv` | 200 | synthetic toggle-switch trajectories |
| `Olsson.csv` | 382 | Olsson et al. myeloid differentiation |
| `MyeloidProgenitors.csv` | 640 | Paul et al. myeloid progenitors (no `labels` column) |

Each row is a cell; columns are expression features plus an optional `labels`
column (cell type). `experiments.datasets.single_cell_graph` turns these into a
symmetric k-NN graph (k=15, minkowski distance).
