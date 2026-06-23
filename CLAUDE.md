# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`hypeGRL` is a Python library for Graph Representation Learning in hyperbolic (and Euclidean) spaces. Its key contributions are:
1. A unified encoder-decoder framework across multiple embedding methods and geometries
2. Principled handling of **partially observed graphs** (unknown edges) via joint optimization — rather than zero-imputing missing entries, unknown edge weights are treated as free variables optimized jointly with embeddings

The Python package is `hypegrl` (all lowercase), installed from this directory.

## Commands

```bash
# Install for development
pip install -e ".[dev]"

# Run all tests with coverage
pytest

# Run a single test file
pytest tests/test_embedders.py

# Run a single test by name
pytest tests/test_embedders.py::test_embedder_fit_shape

# Lint
ruff check hypegrl/
```

## Architecture

The library is organized around the encoder-decoder formalism. Every method encodes a graph into a low-dimensional manifold and defines a decoder that reconstructs pairwise similarity. Unknown edges are handled uniformly across all gradient-based methods.

### Core abstractions

**`hypegrl/embedders/base.py` — `HyperbolicEmbedder` (ABC)**
All embedders inherit from this. The mandatory interface is:
- `fit(G, unknown_edges)` — encode the graph into embeddings
- `embeddings()` → `(N, d)` array
- `structural_similarity(G)` → `(N, N)` — the target the encoder tries to match (e.g., adjacency matrix, forest matrix)
- `decode(X)` → `(N, N)` — the decoder output (e.g., connection probabilities)
- `distance(X, A)` → scalar tensor — the loss, called by the joint optimizer

Capability flags (`is_gradient_based`, `is_generative`, `supports_update`, `supports_node_update`) let callers query what a method supports without try/except.

**`hypegrl/inference/joint_optimizer.py` — `joint_optimize()`**
The central optimization engine. Gradient-based embedders delegate entirely to this function. It:
- Wraps embeddings `X` as `geoopt.ManifoldParameter` (for Riemannian Adam on the correct manifold)
- Reparametrizes unknown edge weights `a_Omega` through sigmoid to keep them in `(0, 1)`
- Initializes `a_Omega` via `imputation.compute_a_omega_init()` (row/column mean heuristic)
- Runs `RiemannianAdam` over both `X` and `a_Omega` simultaneously

Adding a new gradient-based embedder means implementing `distance()` and calling `joint_optimize()` from `fit()`.

### Embedders

| File | Class | Geometry | Structural similarity | Loss |
|---|---|---|---|---|
| `embedders/poincare_maps.py` | `PoincareMapsEmbedder` | Poincaré disk | Forest matrix `Q = (I+L)^{-1}` | Symmetric KL divergence |
| `embedders/hypermap.py` | `HyperMapEmbedder` | Poincaré ball (d-dim) | Adjacency matrix | Fermi-Dirac NLL |
| `embedders/hydra.py` | `HydraEmbedder` | Poincaré disk | Shortest-path distances | Stress (RMS distance error) |
| `embedders/hydra_plus.py` | `HydraPlusEmbedder` | Poincaré disk | Shortest-path distances | Stress (RMS distance error) |

Stubs exist for `lorentz.py`, `ase.py`, `dmercator.py`, `poincare_embeddings.py`, `out_of_sample.py`.

### Manifold helpers

**`hypegrl/manifolds/poincare.py`** — shared `POINCARE_BALL = geoopt.PoincareBall()` instance (curvature `c=1`), plus coordinate conversion utilities (both NumPy and Torch/autograd versions):
- Poincaré ↔ H² polar (2D)
- Poincaré ↔ hyperspherical (d-dim)
- Poincaré ↔ Lorentz/hyperboloid

`hypegrl/manifolds/lorentz.py` is a stub.

### Imputation utilities (`hypegrl/inference/imputation.py`)

- `compute_a_omega_init(G, unknown_edges)` — row/column mean heuristic for initializing unknown weights
- `compute_results_imputation_unweighted(G, unknown_edges, a_omega)` — evaluation helper returning actual vs. predicted dicts for unweighted graphs
- `compute_threshold_two_clusters(G, X, unknown_edges, manifold)` — distance-based threshold from embedding geometry

### Generation and streaming (partial)

- `hypegrl/generation/` — `GraphGenerator` ABC + stub implementations (`fermi_dirac.py`, `rdpg.py`, `threshold.py`)
- `hypegrl/streaming/woodbury.py` — math for O(N²) incremental Woodbury updates to the forest matrix (documented but not fully implemented)
- `hypegrl/streaming/buffer.py`, `scheduler.py` — streaming infrastructure stubs

### Visualization

- `hypegrl/visualization/disk.py` — Poincaré disk plots
- `hypegrl/visualization/graph.py`, `lorentz.py` — additional visualization utilities

## Key design notes

- **Node ordering in HyperMap**: `HyperMapEmbedder` sorts nodes by degree during the greedy initialization phase. The row order of `embeddings()` matches `embedder.nodes_sorted`, not the original NetworkX node order.
- **Connectivity constraint**: `update()` raises `ValueError` if removing edges/nodes would disconnect the graph — a disconnected Laplacian produces a block-diagonal forest matrix that changes the problem character.
- **`update()` is currently warm-started refit**: both `PoincareMapsEmbedder` and `HyperMapEmbedder` declare `supports_update=True` but implement `update()` as a warm-started call to `fit()`. True incremental updates via Woodbury are planned but not yet implemented.
- **All tensors use `float64`**: the optimization is numerically sensitive; `torch.float64` throughout.
- **Hydra embedders are non-gradient**: `HydraEmbedder` and `HydraPlusEmbedder` use closed-form spectral decomposition. They fall back to zero-imputation with a warning when `unknown_edges` is provided, since they cannot route through `joint_optimize`.
