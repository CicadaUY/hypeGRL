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
| `embedders/lorentz_embeddings.py` | `LorentzEmbeddingsEmbedder` | Lorentz / hyperboloid (d-dim) | (Weighted) adjacency `K` | Soft-ranking NLL over the similarity neighbourhood `N(i,j)` |

Stubs exist for `ase.py`, `out_of_sample.py`.

### Manifold helpers

**`hypegrl/manifolds/poincare.py`** — shared `POINCARE_BALL = geoopt.PoincareBall()` instance (curvature `c=1`), plus coordinate conversion utilities (both NumPy and Torch/autograd versions):
- Poincaré ↔ H² polar (2D)
- Poincaré ↔ hyperspherical (d-dim)
- Poincaré ↔ Lorentz/hyperboloid

**`hypegrl/manifolds/lorentz.py`** — `StableLorentz` (a `geoopt.Lorentz` subclass) plus the shared `LORENTZ = StableLorentz()` instance (curvature `k=1`). `StableLorentz` clamps each point's spatial norm `‖x'‖` to `max_norm` (default `1e2`, Poincaré radius `≈0.99`) inside `projx` and the retraction (`expmap`/`retr`), so every `RiemannianAdam` step (via `retr_transp → retr`) stays bounded. This prevents the hyperboloid `NaN` failure — one high-lr `expmap` step scales coords by `cosh(‖v‖_L)` and overflows `x_0` — mirroring the reference impl's `set_dim0` renorm. Defined on the manifold (not in an embedder) so any future hyperboloid method can reuse it; the D-Mercator Lorentz attempt was abandoned for exactly this overflow (it refines on the Poincaré ball instead — see that design note).

### Imputation utilities (`hypegrl/inference/imputation.py`)

- `compute_a_omega_init(G, unknown_edges)` — row/column mean heuristic for initializing unknown weights
- `compute_results_imputation_unweighted(G, unknown_edges, a_omega)` — evaluation helper returning actual vs. predicted dicts for unweighted graphs
- `compute_threshold_two_clusters(G, X, unknown_edges, manifold)` — distance-based threshold from embedding geometry

### Parameter estimation (`hypegrl/inference/parameters.py`)

**Scope / where estimators go.** This module is the single home for estimating the **scalar hyperparameters of the latent-geometry model** from an observed graph — the quantities a method needs *before* it embeds (e.g. HyperMap / E-PSO's `(m, L, γ, T, ζ)`). It is deliberately method-agnostic and imports no embedders, so any method can reuse it.

Decision rule for what belongs here:
- **Belongs here:** anything that maps a graph (or its degree sequence / clustering) to a scalar model parameter — `γ` (power-law exponent), `k_min` (power-law cutoff), and, when added, `T` (temperature, via clustering matching) and the `m`/`L` average-degree heuristics.
- **Does *not* belong here:** estimating the *unknown adjacency entries* `a_Ω` (that is `imputation.py`), solving for embeddings (`joint_optimizer.py` / `riemannian_optimizer.py`), or anything specific to one embedder's warm-start pipeline (e.g. the greedy init in `_hypermap_init.py`, the κ/β inference baked into `_dmercator_init.py`).

Current contents:
- `estimate_gamma(G, k_min=None, fallback_gamma=DEFAULT_GAMMA)` — discrete power-law MLE (Clauset–Shalizi–Newman, arXiv:0706.1062, §3.2). `k_min=None` (default) selects the cutoff automatically by KS minimisation; an explicit integer uses that cutoff directly. When no cutoff can be selected (`choose_kmin_ks` returns `None`: < 3 distinct degrees, or no cutoff with a well-supported tail), it warns and returns `fallback_gamma` (`DEFAULT_GAMMA = 2.5`, a neutral scale-free value; chosen over fitting because the `k_min=1` MLE can return an invalid `γ < 2`). Also importable as `hypegrl.embedders.hypermap.estimate_gamma`.
- `choose_kmin_ks(degrees, min_tail=DEFAULT_MIN_TAIL)` — CSN §3.3 KS-minimising choice of the cutoff. Candidates are the distinct degrees **with the largest dropped** (following the `powerlaw` package's `find_xmin`): the max-degree cutoff has a single-value tail that any distribution fits perfectly (KS=0), so scoring it spuriously wins (this was giving trees a meaningless `γ≈6.5`). Candidates whose tail retains fewer than `min_tail` degrees are also skipped (`DEFAULT_MIN_TAIL = 50`, CSN §3.2's "n ≳ 50 for reliable estimates" rule of thumb; used as a hard candidate filter is our choice). Without it, pure KS-min drifts into the sparse upper tail of a bell-shaped sequence and reports a spuriously steep `γ` off a handful of points. **The floor bounds tail size, not `γ` — the exponent is left uncapped, because HyperMap/E-PSO admits any `γ ≥ 2` (β = 1/(γ−1) ∈ (0,1]); we deliberately do *not* adopt `powerlaw`'s α ≤ 3 normalizability cap, which would reject legitimate `γ > 3`.** Returns `{k_min, gamma, ks, n_tail}` (large `ks` ⇒ not really a power law, e.g. a tree), or `None` when there are < 3 distinct degrees or no candidate reaches `min_tail` (pass `min_tail=1` to disable). Consequence: graphs too small for a 50-degree tail (e.g. karate, N=34) now return `None` → `estimate_gamma` falls back.
- `power_law_gof(degrees, n_bootstrap=1000, seed=None)` — CSN §4 semiparametric-bootstrap goodness-of-fit for the power-law hypothesis; **user-called, never automatic**. Returns `{p_value, plausible, D, k_min, gamma, n_tail, n_bootstrap}` with `plausible = p_value >= 0.1`, or `None` when the data can't be fit (`choose_kmin_ks` → `None`). Small `p` **rejects** the power law; large `p` only means it's *plausible*, not correct/best. Robustly rejects body-dominated non-power-laws (trees, `p≈0` across seeds) and passes true power laws (BA, zipf). **Known method limitation (not a bug):** a bell-shaped sequence (Poisson/Erdős–Rényi) is *not* robustly rejected — its sparse upper tail is weakly power-law-compatible, so the single-distribution GOF can call it plausible. Discriminating power-law from such an alternative is the job of a likelihood-ratio test (e.g. `powerlaw`'s `distribution_compare`), which we don't implement.

The E-PSO temperature `T` is **not** estimated here — see the `HyperMapEmbedder.estimate_temperature` design note below. Planned for this module: the `m` / `L` average-degree heuristics (both pure graph → scalar).

### Generation and streaming (partial)

- `hypegrl/generation/` — `GraphGenerator` ABC + stub implementations (`fermi_dirac.py`, `rdpg.py`, `threshold.py`)
- `hypegrl/streaming/woodbury.py` — math for O(N²) incremental Woodbury updates to the forest matrix (documented but not fully implemented)
- `hypegrl/streaming/buffer.py`, `scheduler.py` — streaming infrastructure stubs

### Visualization

- `hypegrl/visualization/disk.py` — Poincaré disk plots
- `hypegrl/visualization/graph.py`, `lorentz.py` — additional visualization utilities

## Key design notes

- **Node ordering — the uniform `nodes()` accessor**: every embedder exposes `emb.nodes()` (a method on the base `HyperbolicEmbedder`, backed by `self._nodes`, set in `fit`) returning the node labels in `embeddings()` row order — `None` before fit. Reordering methods report their order (`HyperMapEmbedder` sorts by degree during greedy init; `DMercatorEmbedder` has its own order); non-reordering methods (`PoincareMapsEmbedder`, `Hydra`, `HydraPlus`, `PoincareEmbeddings`) return `list(G.nodes())`. This replaced the old per-embedder heterogeneity (HyperMap's `nodes_sorted` property, DMercator's `nodes` property). Consequence (a real silent footgun): `plot_poincare_graph`/`plot_polar` and any adjacency you build to pair with the rows must use that order, else every node is drawn at the wrong point and the plot looks like noise — pass `nodes=emb.nodes()` and the plotters align internally (`nodes=None` assumes `G.nodes()` order). Documented in the base `nodes()`, `HyperMapEmbedder.embeddings()`, and both plotter docstrings.
- **HyperMap temperature estimation is post-embedding, not in `parameters.py`**: neither the HyperMap paper nor the official C++ estimates `T` (the C++ takes it as a required `-t` input). The E-PSO paper (ref [5], Papadopoulos, Psomas & Krioukov, IEEE/ACM ToN 23, 198 (2015), §V) *does*: its global connection probability (Eq. 10) is Fermi-Dirac in the hyperbolic distance, `p(x)=1/(1+e^{(ζ/2T)(x−R)})`, so its steepness is exactly `ζ/2T`. `HyperMapEmbedder.estimate_temperature(X=None, A=None)` reads `T` off a **fixed embedding** by a one-parameter logistic MLE of the edge indicator on `u_ij = x_ij − R_{max(i,j)}` (`p=σ(−s·u)`, `s=ζ/2T`, `T=ζ/2s`) — the max-likelihood form of the paper's binned "tail slope" read-off (our reformulation, not verbatim). Because it needs an embedding (chicken-and-egg with `fit`), it lives on the embedder, **not** in `inference/parameters.py`. `T` is an *output* (it characterises the network's average clustering and calibrates the connection-probability decoder), not a knob to improve the coordinates: the paper reports (its Fig. 5) that embedding *quality* is not significantly affected by the input `T`, so reading `T` off a single embedding made with any reasonable input already gives a good estimate. For a better, self-consistent value, set `self.T` to the result and refit a couple of times. (Distance-ranked link prediction needs no `T` at all; `T` only calibrates the probability values.) Verified to recover a known `T` to <1% on synthetic Fermi-Dirac graphs (`tests/test_embedders.py::test_hypermap_estimate_temperature_recovers_known_slope`). We deliberately chose this over clustering-matching (the D-Mercator-style route via `infer_kappa_and_beta`), which would fit `parameters.py` but needs an E-PSO generator we don't have.
- **Connectivity constraint**: `update()` raises `ValueError` if removing edges/nodes would disconnect the graph — a disconnected Laplacian produces a block-diagonal forest matrix that changes the problem character.
- **`update()` is currently warm-started refit**: both `PoincareMapsEmbedder` and `HyperMapEmbedder` declare `supports_update=True` but implement `update()` as a warm-started call to `fit()`. True incremental updates via Woodbury are planned but not yet implemented.
- **All tensors use `float64`**: the optimization is numerically sensitive; `torch.float64` throughout.
- **Hydra embedders are non-gradient**: `HydraEmbedder` and `HydraPlusEmbedder` use closed-form spectral decomposition. They fall back to zero-imputation with a warning when `unknown_edges` is provided, since they cannot route through `joint_optimize`.
- **D-Mercator = faithful original init + hyperbolic refinement**: `embedders/_dmercator_init.py` (`dmercator_init`) runs the full original D-Mercator pipeline — joint κ/β inference (clustering matching), S^D-corrected Laplacian Eigenmaps, likelihood-maximisation angular refinement, and final κ readjustment — to produce the warm start, mirroring how `_hypermap_init.py` provides HyperMap's greedy init. `DMercatorEmbedder.fit()` then maps the warm start `(v_i, r_i)` to the Poincaré ball point `x_i = tanh(r_i/2)·v_i` and refines it on the **Poincaré ball** (`POINCARE_BALL`) with Riemannian Adam, minimising a Fermi-Dirac NLL on the exact hyperbolic distances `p_ij = 1/(1+e^{(β/2)(d_H − R̂)})`. β and R̂ are fixed from the init; the refinement moves both angular and radial coordinates, so the hidden degree κ (recovered from the final radius) is learned jointly with the angle. The init is binary-only (edge weights ignored) and treats β as inferred unless fixed in the constructor. Verified that `D=1` reduces to Mercator (`R = N/2π`).
- **β inference is bidirectional with a `β = D` floor**: clustering matching (`infer_kappa_and_beta`) raises β when the model under-clusters and lowers it toward the floor `D + 0.01` when it over-clusters, clamping (with a warning) when the target clustering is unreachable. Low-clustering graphs — trees, where `c̄_emp = 0` — drive β to the floor. This mirrors the reference C++ `infer_parameters(dim)`/`infer_parameters()`; an earlier one-directional "grow β only" bracket left β pinned at its random init on such graphs, corrupting μ/R̂/radial scale.
- **Stage-2 init: LE for all D by default; optional Mercator ordering-init for D=1**: the reference C++ uses LE only for D≥2 and the classic Mercator ordering + expected-angular-gap re-spacing for D=1. The `d1_init` constructor argument `DMercatorEmbedder(..., d1_init=...)` (and the `dmercator_init(..., d1_init=...)` function param) default to `"le"` (the paper's generalisation, used for all D); pass `d1_init="mercator"` to use `mercator_ordering_init` (the C++ D=1 init) instead. The flag is ignored (with a warning) for `d > 2`. On leaf-heavy graphs (trees) `"mercator"` spaces degree-one nodes deterministically around their hub while `"le"` reinserts each leaf at a random angle (MLE never moves leaves): measured on `balanced_tree(2,4)`/`(2,5)` over 8 seeds it gives a ~6–10× larger minimum inter-leaf angle and comparable-to-better reconstruction (AUC 0.83 vs 0.77 on `(2,4)`, ~tied on `(2,5)`). Default stays `"le"`.
- **Why D-Mercator refines on the Poincaré ball, not the hyperboloid**: both are exact charts of H^{D+1} and `geoopt.PoincareBall.dist` is the exact hyperbolic distance, but the choice is numerical. Low-degree leaves belong at large radius `r ≈ R̂` (which reaches ~16–20 even on tiny graphs), where the hyperboloid timelike coordinate `cosh(r) ≈ 1e5–1e7` overflows off the manifold during optimisation — fatal in the D=1/Mercator, leaf-heavy regime (NaN on most seeds of a Barabási–Albert tree). The Poincaré ball keeps coordinates in (−1, 1) and is stable there. An earlier Lorentz/`geoopt.Lorentz` implementation was abandoned for this reason; `tests/test_embedders.py::test_dmercator_robust_on_leaf_heavy_graph_d1` guards against regressing to it.
- **Poincaré Embeddings has two interchangeable losses, default = ranking**: `PoincareEmbeddingsEmbedder(loss=...)` selects the encoder-decoder instantiation. `"ranking"` (default, the original Nickel & Kiela 2017 objective) decodes raw hyperbolic distances and minimises a soft-ranking NLL; its negatives are **re-sampled every step** inside `distance()` (`sample_negatives` draws `n_negatives` per node from the *known* non-edges `A_ik==0`, excluding self and imputed unknown pairs), so the loss is stochastic — tests compare smoothed loss windows, not `hist[-1] < hist[0]`. Each positive pair is weighted by `A_ij`, which is `1` for ordinary edges and the imputed value for unknown edges, so gradients reach `a_Omega` through the joint optimiser. `"fermi_dirac"` instead decodes the Fermi-Dirac edge probability `1/(exp((d_H−r)/t)+1)` and minimises Bernoulli cross-entropy (the same family as HyperMap/D-Mercator but with a single global `r`, `t`); only this mode reports `is_generative()=True`. `decode()` returns the distance matrix under `"ranking"` and the probability matrix under `"fermi_dirac"`.
- **Lorentz Embeddings = the Poincaré-Embeddings ranking objective on the hyperboloid (Nickel & Kiela 2018)**: same soft-ranking loss, different (numerically better) chart of the same space, with an exact-geodesic optimiser. `LorentzEmbeddingsEmbedder` embeds on `manifolds.lorentz.LORENTZ` (`StableLorentz`, the spatial-norm-clamped `geoopt.Lorentz`, whose `expmap` is the paper's Eq. 9), so intrinsic dim `d` uses `d+1` ambient coords. Near-origin init is Eq. 6 (`x' ~ U(-init_scale, init_scale)`, `x_0 = sqrt(1+‖x'‖²)`, `init_scale=1e-3` per the paper). `embeddings()` returns the **Poincaré-ball image** `(N,d)` via the Eq. 11 isometry (`lorentz_to_poincare`) so it drops into the disk plotters and the rest of the library; raw `(N,d+1)` hyperboloid coords are on `hyperboloid_embeddings()`, and `X_init`/warm-start (`update()`) use the hyperboloid representation, not the ball. **Similarity generalisation**: `structural_similarity` is the *weighted* adjacency `K`, and the negative set is the paper's `N(i,j)={z: K_{iz}<K_{ij}}` (less-similar nodes), which collapses to Poincaré Embeddings' non-edges on a binary graph — this is how weighted graphs are supported. **Unknown edges (the key design call)**: the `K_{iz}<K_{ij}` membership test is a non-differentiable hard threshold (same obstruction that blocks Hydra's shortest-path target from routing unknown-edge gradients), so (1) unknown pairs are excluded from every `N(i,j)` — negatives come only from *known* structure, keeping `a_Omega` out of the combinatorial part — and (2) positives are weighted by `K_{ij}` **only when `Omega` is non-empty** (`distance()` sets `weighted = len(unknown_edges) > 0`), which routes a smooth gradient to `a_Omega`. Consequence: with `Omega={}` the loss is the paper's *unweighted* Eq. 12 exactly (binary *and* weighted graphs); the `K_{ij}`-weighting is hypeGRL's device confined to the unknown-edge regime the paper never covers (and a no-op on binary graphs). **Difference from the paper**: optimiser is geoopt `RiemannianAdam`, not the paper's exact-geodesic Riemannian *SGD* (Algorithm 1) — geometry still exact, update rule Adam, matching the rest of the library. Ranking-only (the paper's sole loss); no Fermi-Dirac variant, so `is_generative()=False` and `decode()` returns the Lorentz distance matrix. **Verified against the theSage21 PyTorch reference** (`scripts/lorentz-embeddings`): its `N(i,j)` (`pairwise[i,x] < min`) and unweighted-positive loss confirm our similarity-generalisation and `Omega`-gated weighting; it treats the matrix as a *directed* DAG and samples one positive per anchor per epoch (minibatch SGD), where we sum symmetrically over all edges full-batch (closer to the paper's symmetric Eq. 12). **Defaults `lr_X=0.3` / `n_steps=2000` are tuned, not paper values**: leaves must reach large radius for generality=norm to emerge, and a timid rate leaves the embedding under-spread (~5-10 AUC points worse on trees/karate/Les Mis). This aggressive rate is only safe because of the `StableLorentz` clamp — without it, `lr_X≳0.5` NaNs (the reference guards the same way via `set_dim0`). We deliberately skip the reference's lr burn-in (no measurable effect under Adam).

## Conventions (for contributors and Claude)

- **Documentation reflects the present.** Comments and docstrings describe the code as it is *now* — no change history ("previously…", "for backward compatibility", "used to…"). Record *decisions* in this file, not in code comments.
- **Be explicit about extrapolation.** Keep a clear line between what a cited source (paper, reference code) actually says, what the task asked for, and anything added on top — home-grown heuristics, default values, thresholds, design choices. Don't attribute invented specifics to a source; when citing from memory rather than a checked reference, say so.
- **Readability over premature optimization.** This is research code: prefer clear code that reads like the math / pseudocode. Optimize only when asked, and state a change's true scope and readability cost honestly.
