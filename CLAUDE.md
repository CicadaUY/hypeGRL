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
| `embedders/dmercator.py` | `DMercatorEmbedder` | Poincaré ball (d-dim) | Binary adjacency | Fermi-Dirac NLL on hyperbolic distance |
| `embedders/poincare_embeddings.py` | `PoincareEmbeddingsEmbedder` | Poincaré ball (d-dim) | Adjacency matrix | Soft-ranking NLL with negative sampling (default), or Fermi-Dirac Bernoulli CE |

Stubs exist for `lorentz.py`, `ase.py`, `out_of_sample.py`.

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
- **D-Mercator = faithful original init + hyperbolic refinement**: `embedders/_dmercator_init.py` (`dmercator_init`) runs the full original D-Mercator pipeline — joint κ/β inference (clustering matching), S^D-corrected Laplacian Eigenmaps, likelihood-maximisation angular refinement, and final κ readjustment — to produce the warm start, mirroring how `_hypermap_init.py` provides HyperMap's greedy init. `DMercatorEmbedder.fit()` then maps the warm start `(v_i, r_i)` to the Poincaré ball point `x_i = tanh(r_i/2)·v_i` and refines it on the **Poincaré ball** (`POINCARE_BALL`) with Riemannian Adam, minimising a Fermi-Dirac NLL on the exact hyperbolic distances `p_ij = 1/(1+e^{(β/2)(d_H − R̂)})`. β and R̂ are fixed from the init; the refinement moves both angular and radial coordinates, so the hidden degree κ (recovered from the final radius) is learned jointly with the angle. The init is binary-only (edge weights ignored) and treats β as inferred unless fixed in the constructor. Verified that `D=1` reduces to Mercator (`R = N/2π`).
- **β inference is bidirectional with a `β = D` floor**: clustering matching (`infer_kappa_and_beta`) raises β when the model under-clusters and lowers it toward the floor `D + 0.01` when it over-clusters, clamping (with a warning) when the target clustering is unreachable. Low-clustering graphs — trees, where `c̄_emp = 0` — drive β to the floor. This mirrors the reference C++ `infer_parameters(dim)`/`infer_parameters()`; an earlier one-directional "grow β only" bracket left β pinned at its random init on such graphs, corrupting μ/R̂/radial scale.
- **Stage-2 init: LE for all D by default; optional Mercator ordering-init for D=1**: the reference C++ uses LE only for D≥2 and the classic Mercator ordering + expected-angular-gap re-spacing for D=1. The `d1_init` constructor argument `DMercatorEmbedder(..., d1_init=...)` (and the `dmercator_init(..., d1_init=...)` function param) default to `"le"` (the paper's generalisation, used for all D); pass `d1_init="mercator"` to use `mercator_ordering_init` (the C++ D=1 init) instead. The flag is ignored (with a warning) for `d > 2`. On leaf-heavy graphs (trees) `"mercator"` spaces degree-one nodes deterministically around their hub while `"le"` reinserts each leaf at a random angle (MLE never moves leaves): measured on `balanced_tree(2,4)`/`(2,5)` over 8 seeds it gives a ~6–10× larger minimum inter-leaf angle and comparable-to-better reconstruction (AUC 0.83 vs 0.77 on `(2,4)`, ~tied on `(2,5)`). Default stays `"le"`.
- **Why D-Mercator refines on the Poincaré ball, not the hyperboloid**: both are exact charts of H^{D+1} and `geoopt.PoincareBall.dist` is the exact hyperbolic distance, but the choice is numerical. Low-degree leaves belong at large radius `r ≈ R̂` (which reaches ~16–20 even on tiny graphs), where the hyperboloid timelike coordinate `cosh(r) ≈ 1e5–1e7` overflows off the manifold during optimisation — fatal in the D=1/Mercator, leaf-heavy regime (NaN on most seeds of a Barabási–Albert tree). The Poincaré ball keeps coordinates in (−1, 1) and is stable there. An earlier Lorentz/`geoopt.Lorentz` implementation was abandoned for this reason; `tests/test_embedders.py::test_dmercator_robust_on_leaf_heavy_graph_d1` guards against regressing to it.
- **Poincaré Embeddings has two interchangeable losses, default = ranking**: `PoincareEmbeddingsEmbedder(loss=...)` selects the encoder-decoder instantiation. `"ranking"` (default, the original Nickel & Kiela 2017 objective) decodes raw hyperbolic distances and minimises a soft-ranking NLL; its negatives are **re-sampled every step** inside `distance()` (`sample_negatives` draws `n_negatives` per node from the *known* non-edges `A_ik==0`, excluding self and imputed unknown pairs), so the loss is stochastic — tests compare smoothed loss windows, not `hist[-1] < hist[0]`. Each positive pair is weighted by `A_ij`, which is `1` for ordinary edges and the imputed value for unknown edges, so gradients reach `a_Omega` through the joint optimiser. `"fermi_dirac"` instead decodes the Fermi-Dirac edge probability `1/(exp((d_H−r)/t)+1)` and minimises Bernoulli cross-entropy (the same family as HyperMap/D-Mercator but with a single global `r`, `t`); only this mode reports `is_generative()=True`. `decode()` returns the distance matrix under `"ranking"` and the probability matrix under `"fermi_dirac"`.
