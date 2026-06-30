# HyperMap C++ reference fixtures

Golden outputs produced by the **official C++ HyperMap implementation** (the
"fast hybrid" common-neighbors method of Papadopoulos, Aldecoa & Krioukov,
*Network Geometry Inference using Common Neighbors*, Physical Review E 92,
022807, 2015), used by `tests/test_hypermap_cpp_equivalence.py` to verify that
our pure-Python re-implementation (`hypegrl.embedders._hypermap_init`) reproduces
the original method.

Committing the outputs means the test suite needs **no C++ toolchain**. To
regenerate them, build the upstream binary (`make` in the C++ source tree) and
run:

```bash
python generate_fixtures.py --binary /path/to/hypermap
```

(The default `--binary` path is the developer checkout; override it on another
machine.)

## Files

| Fixture | Graph | Temperature `T` |
|---|---|---|
| `karate_T0.3.txt` | Zachary karate club (`nx.karate_club_graph()`) | 0.3 |
| `karate_T0.5.txt` | Zachary karate club | 0.5 |
| `karate_T0.7.txt` | Zachary karate club | 0.7 |

All use `gamma = 2.7249` (Clauset Eq. 3.7 estimate on the karate degrees with
`k_min=4`), `m_in=1`, `zeta=1`, `k_speedup=0`, corrections **on** — the same
inputs recorded in each file's header.

## File format

A header block of `# ...` comment lines records the model inputs (graph, gamma,
T, m_in, zeta, k_speedup, corrections), followed by per-node rows:

```
id  theta  r
```

where `theta` is the angular coordinate in `[0, 2π)` and `r` the radial
coordinate. The rows are in the **C++ arrival order** (degree-descending rank),
so `r` is non-decreasing down the file. Values are printed to 8 decimals — the
precision of the C++ `cout` output, which bounds how tightly the test can compare
(≈`1e-8`).

## What is checked, and the tie-break caveat

HyperMap's greedy MLE is **deterministic**, so — unlike the D-Mercator
fixtures next door — we can assert that the actual coordinates match, not just
closed-form identities. With the arrival order fixed, our init reproduces the
C++ output to `~1e-8 rad` in angle and `~1e-15` in radius (see the
`hypermap-cpp-verification` project note).

The one subtlety: the C++ ranks nodes with a **non-stable `std::sort`** by
degree, whereas our Python uses a *stable* sort (ties broken by ascending node
id). On karate, 22/34 nodes are in degree-tie groups, so the two can pick
**different (but equally valid) arrival orders**, which then changes the
embedding. This is a gauge freedom, not a bug. The test removes it by recovering
the C++ arrival order from the fixture (the rows are already in that order; `r`
is strictly increasing in rank) and relabeling the graph so our stable sort
reproduces exactly that order before comparing.
