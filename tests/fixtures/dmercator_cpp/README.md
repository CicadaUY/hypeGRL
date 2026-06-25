# D-Mercator C++ reference fixtures

Golden outputs produced by the **official D-Mercator C++ implementation**
(version 0.9, Jankowski/Allard/Boguñá/Serrano), used by
`tests/test_dmercator_cpp_equivalence.py` to verify that our pure-Python
re-implementation (`hypegrl.embedders._dmercator_init`) reproduces the original
method's deterministic, closed-form quantities exactly.

Committing the outputs means the test suite needs **no C++ toolchain** — neither
to run `pytest` nor for end users. To regenerate them, build the upstream
`dmercator` pybind module and run, e.g.:

```python
import dmercator, networkx as nx
nx.write_edgelist(nx.karate_club_graph(), "karate_club.edge")
dmercator.embed("karate_club.edge", dimension=1)   # -> karate_club.inf_coord
```

## Files

| Fixture | Graph | Dimension `D` (sphere S^D) |
|---|---|---|
| `karate_d1.inf_coord`        | Zachary karate club (`nx.karate_club_graph()`) | 1 |
| `karate_d2.inf_coord`        | Zachary karate club                            | 2 |
| `balanced_tree_d1.inf_coord` | `nx.balanced_tree(2, 4)`                       | 1 |

Note: the embedder's `dimension` argument is the *similarity-space* dimension
`D`; the hyperbolic embedding lives in H^{D+1}. In our library `DMercatorEmbedder(d=...)`
uses the ambient dimension `d = D + 1`, so these correspond to `d=2` and `d=3`.

## File format

A header block of `# ...` comment lines reports the global parameters:

```
#   - nb. vertices:   N
#   - beta:           β   (inverse temperature)
#   - mu:             μ
#   - radius_S1 / radius_S^D:     R   (sphere radius)
#   - radius_H2 / radius_H^D+1:   R̂   (hyperbolic radial offset)
#   - kappa_min:      κ_min
```

followed by per-node rows:

- **D=1:** `Vertex  Inf.Kappa  Inf.Theta  Inf.Hyp.Rad.`
- **D≥2:** `Vertex  Inf.Kappa  Inf.Hyp.Rad  Inf.Pos.1 ... Inf.Pos.(D+1)`
  (the `Inf.Pos.*` columns are the unit-vector coordinates on S^D, *scaled by
  the hyperbolic radius*).

## What is (and isn't) checked

The Tier-0 equivalence test asserts the **deterministic closed-form identities**
the C++ must satisfy given its own inferred `β`/`κ`:

- `R = compute_R(N, D)`,
- `R̂` and the radial map `r_i = R̂ − (2/D)·ln(κ_i/κ_min)` from `radial_map`,
- `κ_min = min_i κ_i`,
- `μ = compute_mu(β, D, ⟨k⟩)` — **only where β is comfortably above D**.

It does **not** try to reproduce the stochastic stages (κ/β inference via
Monte-Carlo clustering, Laplacian-Eigenmaps angles, likelihood-maximisation
refinement): those use the C++'s own RNG stream and cannot be matched bit-for-bit
in NumPy.

### Known upstream quirk: μ in the β→D limit

For `balanced_tree_d1` the inference drives β down to ≈1.0067, i.e. the D=1
boundary where `sin(π/β)→0` and μ becomes pathologically sensitive to β (it
roughly *halves* between consecutive bisection iterates). The C++ header prints a
μ evaluated at the **penultimate** β-iterate (≈1.0135) alongside the **final**
β it reports (1.0067), so the printed μ disagrees with `compute_mu(β_final, …)`
by ~2×. This is an upstream output inconsistency, not a formula difference —
`compute_mu` is term-for-term identical to the C++ `calculate_mu`/`calculateMu`.
The test therefore skips the tight μ check for this near-degenerate case.
