# HyperMap — pseudocode for debugging

```{note}
This page is the algorithmic specification that
``hypegrl/embedders/_hypermap_init.py`` implements faithfully — the greedy
*initialization* of {class}`~hypegrl.embedders.hypermap.HyperMapEmbedder`. That
warm start is then lifted to ``d`` dimensions and refined on the Poincaré ball
with Riemannian Adam (see {doc}`hypermap`). The ⚠️ notes record where the paper
and the upstream C++ disagree; §7 records how the refinement layer is kept
consistent with this specification.
```

Reconstructed from Papadopoulos, Aldecoa & Krioukov, *Network geometry inference using common
neighbors*, Phys. Rev. E **92**, 022807 (2015), building on the original HyperMap of Papadopoulos,
Psomas & Krioukov, IEEE/ACM Trans. Netw. **23**, 198 (2015). Equation/figure numbers below refer to
the 2015 paper. Reference C++ implementation: `2015_code_hypermap/src/` (`hypermap.cpp`, `Graph.cpp`,
`Node.h`) — the **fast hybrid** method. Consult it for the bits flagged ⚠️ below, which the paper
underspecifies or states differently from what the code does.

The method embeds a graph into the **S¹/H² (E-PSO) model**: each node gets a *radial* coordinate
`r` (popularity, fixed analytically by degree rank) and an *angular* coordinate `θ` (similarity,
inferred). It is **not** a single-pass link-based MLE — the released code is a three-part hybrid:
a **common-neighbors (CN)** phase for the high-degree nodes, a **link-based (FD)** phase for the
rest, and periodic **correction steps**, plus an optional `k_speedup` heuristic.

---

## 0. Model constants and primitives

Inputs: adjacency `A` (a_ij ∈ {0,1}), power-law exponent `γ ≥ 2`, temperature `T ∈ (0,1)`,
curvature `ζ > 0` (set ζ = 1 WLOG; it only rescales radii). Optional rates `m`, `L`.

```
β  = 1 / (γ - 1)                              # γ = 1 + 1/β,  β ∈ (0,1]
N  = number of nodes
k̄ = 2·|E| / N                               # mean degree
m  = (given, else) min_i degree(i)           # ⚠️ code default = MIN DEGREE, not 1 (see note A)
L  = (given, else) max( (k̄ - 2m)/2 , 0 )     # internal-link rate
```

Sort nodes by **descending degree**; the rank (1-indexed) is the node's MLE birth time `i`.
Index 0 = highest degree = oldest. Higher index = later = "younger".

```
r⁰(i)              = (2/ζ) · ln(i)                          # birth radius (step 5, Fig. 1), i ≥ 1
RADIAL_AT(j, i)    = β · r⁰(j) + (1−β) · r⁰(i)              # faded radius of OLDER j at YOUNGER i's birth
ANG_SEP(θ₁, θ₂)    = π − | π − |θ₁ − θ₂| |                  # ∈ [0, π]
HYP_DIST(rₐ,r_b,Δθ)= (1/ζ)·arccosh( cosh(ζrₐ)cosh(ζr_b) − sinh(ζrₐ)sinh(ζr_b)cos Δθ )   # Eq. 2
                     ( = |rₐ − r_b| when Δθ = 0 )
CONN_PROB(x, R)    = 1 / ( 1 + exp( (ζ/2T)·(x − R) ) )      # Eq. 1  (Fermi–Dirac)
```

`ζ` cancels in `CONN_PROB` (it only rescales radii), so the inferred **angles are independent of ζ**.

---

## 1. Top-level driver

```
function HYPERMAP(A, γ, T, ζ, k_speedup=0, corrections=true):
    sort nodes by descending degree              # node i (1-indexed) = the i-th highest degree
    (r⁰, R, numCN) = ASSIGN_RADII(N, β, m, L, T, ζ)

    θ = array(N)
    # --- Phase 1: common-neighbors, for the first numCN (high-degree) nodes ---
    for i = 1 .. numCN:
        if i == 1: θ[1] = π   (any value in [0,2π]; fixes rotational gauge); continue
        θ[i] = argmax_θ  CN_LOGLIK(i, θ, ...)    # Eqs. 9–14, brute-force grid

    # --- Phase 2: link-based Fermi–Dirac MLE, for the remaining nodes ---
    for i = numCN+1 .. N:
        θ[i] = argmax_θ  FD_LOGLIK(i, θ, ...)    # Eq. 5, brute-force grid (with k_speedup)
        if corrections and trigger(i):           # at degree thresholds 60/40/20/10
            CORRECTION_STEPS(i, ...)             # re-optimize angles of ALL Phase-2 nodes

    # --- final radii: popularity fade evaluated at the final time N ---
    for i = 1 .. N:
        r[i] = β · r⁰(i) + (1−β) · (2/ζ)·ln(N)   # = RADIAL_AT(i, N)
    return { (r[i], θ[i]) }
```

**Complexity.** Pure link-based: `O(N³)`. Common-neighbors: `O(N⁴)` (the extra factor is the
integral over all third nodes `k`). The released hybrid is `O(N³)`; with `k_speedup` it drops to
`O(N²)`.

---

## 2. Radius and threshold assignment (Fig. 1 step 5 + Eqs. 1, 3, 4)

All radii derive from the birth radius `r⁰(i)`. The per-node cutoff `R_i` (Eq. 3) needs the expected
number of back-connections `m̄_i` (Eq. 4). The code precomputes these in one pass and, while doing
so, counts how many nodes go to the CN phase.

```
function ASSIGN_RADII(N, β, m, L, T, ζ):
    cteR    = (2/ζ)·ln( 2T / sin(Tπ) )                              # i-independent part of Eq. 3
    cteL_t  = 2L(1−β) / ( (1 − N^{−(1−β)})² · (2β−1) )              # ⚠️ Eq. 4 prefactor — see note B
    numCN   = 0
    for i = 1 .. N:
        r⁰(i) = (2/ζ)·ln(i)
        I_i   = (1/(1−β)) · (1 − i^{−(1−β)})                        # I_i in Eq. 3

        # L_t(i): expected internal back-links of node i (with removable-singularity cases)
        if   β == 1   : L_t = 2L(N−i)·ln(i) / ( i·ln(N)² )                          # note C
        elif β == 0.5 : L_t = L · (1 − i^{−0.5}) / (1 − N^{−0.5})² · ln(N/i)        # note C
        else          : L_t = cteL_t · ( (N/i)^{2β−1} − 1 ) · ( 1 − i^{−(1−β)} )    # Eq. 4
        m̄_i  = m + L_t

        if (m̄_i ≥ i − 1) or (i == 1): numCN += 1                   # CN-phase membership, note D
        R[i] = r⁰(i) − cteR − (2/ζ)·ln( I_i / m̄_i )                # Eq. 3   ⚠️ note E (no extra m!)
    return (r⁰, R, numCN)
```

⚠️ **note A — `m` default is the minimum degree, not 1.** `Graph::readGraph` sets `m = degree of the
last (lowest-degree) node` when `m` is not supplied. The README says "default 1", but the code uses
min-degree. They coincide only when min-degree = 1. Match the code.

⚠️ **note B — Eq. 4 prefactor (the most common pseudocode bug).** It is
`2L(1−β) / [ (1 − N^{−(1−β)})² · (2β−1) ]`. Two things people get wrong:
the `(1 − N^{−(1−β)})` factor is **squared**, and the denominator has a plain **`(2β−1)`**, not
`2(2β−1)`. Here `N` is the **final** network size; the per-node dependence is only in the
`(N/i)^{2β−1}` and `(1 − i^{−(1−β)})` factors. (Code: `cteL_t` uses `pow(N,…)` and `2*beta-1`.)

⚠️ **note C — special cases β = 1 (γ = 2) and β = 1/2 (γ = 3).** `2β−1 = 0` at β = 1/2 makes the Eq. 4
prefactor blow up (removable singularity: `[(N/i)^{2β−1} − 1]/(2β−1) → ln(N/i)`). The code uses the
two closed forms above. Guard both if you allow γ near 2 or 3.

⚠️ **note D — `numCN` is a COUNT, used as a PREFIX.** The code increments `numCN` for every node with
`m̄_i ≥ i−1` (or i = 1), then runs the CN phase on nodes `1 .. numCN`. This is only correct if the
condition holds for an initial prefix of the degree-sorted order (it does in practice, since
high-degree/early nodes are the ones expected to connect to all predecessors). Don't re-derive
per-node membership independently.

⚠️ **note E — `R_i` denominator is `I_i / m̄_i`, NOT `I_i / (m · m̄_i)`.** A spurious `m` factor here is
a real bug. (Code: `R_t = r_t − cteR − (2/ζ)·log(I_t / m_t)` with `m_t = m + L_t = m̄_i`.)
At i = 1, `I_1 = 0` so `R_1 → +∞` (the central hub connects to everything); guard `ln(0)` with a
floor if needed.

---

## 3. Phase 1 — common-neighbors angular inference (Sec. III A–B, Eqs. 6–14)

Place each high-degree node at the angle that maximizes a **Gaussian** log-likelihood on the
*number of common neighbors* it shares with each already-placed node. The expected count comes from
integrating the product of two connection probabilities over the third node's angle.

```
function CN_LOGLIK(i, θ_v, r⁰, R, θ, A):                # ln L^i_CN, Eq. 14
    s = 0
    for u = 1 .. i−1:
        emp_CN  = | neighbors(i) ∩ neighbors(u) |        # empirical, from the FULL adjacency
        (λ, σ²) = EXPECTED_CN(i, u, θ_v, θ[u], r⁰, R)    # Eqs. 10–11
        if σ² < ε: σ² = ε                                # ε = 1e-12
        lp = −(emp_CN − λ)² / (2σ²) − ½·ln(2π σ²)        # log N(emp; λ, σ²)
        if lp > 0:                                       # σ² tiny ⇒ −½ln(2πσ²) dominates, note F
            lp = (−1e9 if |emp_CN − λ| > 1 else 0)
        s += lp
    return s

function EXPECTED_CN(i, u, θ_v, θ_u, r⁰, R):            # λ = E[#CN], σ² = Σ p(1−p)
    λ = 0;  σ² = 0
    for k = 1 .. N, k ∉ {i, u}:                          # ⚠️ ALL N nodes are candidate common neighbors
        # radii for the (i,k) and (u,k) pairs: OLDER node faded to YOUNGER's birth; R = YOUNGER's R
        (r_ik_a, r_ik_b, R_ik) = PAIR_RADII(i, k, r⁰, R)
        (r_uk_a, r_uk_b, R_uk) = PAIR_RADII(u, k, r⁰, R)
        # prob that k is a common neighbor of i and u, averaged over k's unknown angle φ:
        p = (1/2π) · ∫₀^{2π}  CONN_PROB( HYP_DIST(r_ik_a, r_ik_b, ANG_SEP(φ,θ_v)), R_ik )
                            · CONN_PROB( HYP_DIST(r_uk_a, r_uk_b, ANG_SEP(φ,θ_u)), R_uk )  dφ
        λ  += p
        σ² += p·(1 − p)
    return (λ, σ²)

function PAIR_RADII(a, b, r⁰, R) -> (faded_older, younger_birth, R_younger):   # HYP_DIST is symmetric
    if r⁰[b] > r⁰[a]:  return ( β·r⁰[a]+(1−β)·r⁰[b],  r⁰[b],  R[b] )   # b younger (born later)
    else:              return ( β·r⁰[b]+(1−β)·r⁰[a],  r⁰[a],  R[a] )   # a younger
```

The integral has no closed form; the code uses **48-point Gauss–Legendre** over `[0, 2π]` then
divides by `2π`.

```
ARGMAX_ANGLE_CN(i): grid search θ ∈ {0, Δθ, 2Δθ, …, 2π},  Δθ = min(1/i, 0.01),  keep STRICT max (>)
```

⚠️ **note F — the positive-log-prob clamp.** When `σ²` is tiny, `−½ln(2πσ²)` can exceed 0, making the
"probability" meaningless. The code then forces `lp = 0` if `emp_CN` is within 1 of `λ`, else
`lp = −1e9`. Replicate this exactly; it materially shapes the CN landscape for early nodes.

⚠️ **note G — empirical CN uses the FULL final adjacency**, including links to nodes not yet placed.
That is precisely why CN beats link-based for early nodes (Sec. III C): it uses "future" connectivity.

---

## 4. Phase 2 — link-based Fermi–Dirac MLE (Fig. 1 step 7, Eq. 5)

Place each remaining node at the angle maximizing the link-based local likelihood over older nodes.

```
function FD_LOGLIK(i, θ_v, r, R, θ, A, compare):        # ln L^i_L, log of Eq. 5
    s = 0
    for j in compare (default: 1..i−1, that are already placed):
        Δθ = ANG_SEP(θ_v, θ[j])
        x  = HYP_DIST(r_i, r_j, Δθ)                      # r_i = r⁰(i) (its birth radius);
                                                         # r_j = RADIAL_AT(j, i) (older j faded to i)
        p  = clip( CONN_PROB(x, R[i]), ε, 1−ε )          # ⚠️ R of node i (the YOUNGER node), note H
        s += ln p           if j ∈ neighbors(i)
             ln(1 − p)      otherwise
    return s
```

Radius bookkeeping when placing node `i`:
- node `i` itself uses its **birth** radius `r⁰(i)` (code: `setRadius(getInitRadius())`);
- every older node `j < i` is faded to `RADIAL_AT(j, i) = β r⁰(j) + (1−β) r⁰(i)`.

Angle search: grid `θ ∈ {0, Δθ, …, 2π}`, `Δθ = min(1/i, 0.01)`, keep **non-strict** max (`≥`,
last-wins on ties — differs from Phase 1's strict `>`).

### `k_speedup` heuristic (Sec. IV, Eq. 16)

For nodes with degree `< k_speedup` (default 0 = OFF):
1. coarse search comparing only against **neighbors already present** (Eq. 16);
2. then a refined full-likelihood search in `[θ* − C·Δθ, θ* + C·Δθ]`, `C = 200`, comparing against
   **all** older nodes.

⚠️ **note H — pairwise `R` is the YOUNGER node's threshold.** When placing node `i` against older `j`,
the connection probability uses `R[i]` (the node born later). This follows Eqs. 6–8: each connection
forms at the birth time of the younger endpoint, so its `R` governs the probability. Getting this
backwards (using the older node's `R`) is a silent bug — see the implementation note in §7.

---

## 5. Correction steps (Sec. III D, Eq. 15)

After placing a node whose degree is one of `{60, 40, 20, 10}` and the next node's degree is strictly
smaller (i.e. we just finished a degree class), re-optimize the angle of **every Phase-2 node**, using
the current angles of **all** nodes (Phase 1 and Phase 2) as context. Repeat `⌊k̄⌋` rounds.

```
function CORRECTION_STEPS(i, ...):                       # triggered at degree-class boundaries
    for round = 1 .. ⌊k̄⌋:                               # ⚠️ note I: code uses floor(k̄)
        for v in Phase-2 nodes (indices numCN .. i):
            θ[v] = argmax_θ  CORR_LOGLIK(v, θ, over l ∈ all placed nodes, l ≠ v)
                   # grid Δθ = min(1/(i+1), 0.01) in the code — note J;  non-strict (≥) max

function CORR_LOGLIK(v, θ_v, ...):                       # log of Eq. 15
    s = 0
    for l ≠ v in all placed nodes:
        (r_a, r_b, R_use) = PAIR_RADII(v, l, r⁰, R)         # older faded to younger; R_use = YOUNGER's R
        Δθ = ANG_SEP(θ_v, θ[l])
        x  = HYP_DIST(r_a, r_b, Δθ)
        p  = clip( CONN_PROB(x, R_use), ε, 1−ε )            # note H again
        s += ln p  if l ∈ neighbors(v) else ln(1−p)
    return s
```

Note that `CORR_LOGLIK` recomputes pairwise radii from `r⁰` per pair (the older node faded to the
younger node's birth time), unlike Phase 2 which reuses the running `r[j]` set to node `i`'s birth.
For CN-phase nodes the correction step **only uses them as context** — their angles are never moved
(they were inferred from richer CN information).

⚠️ **note I — rounds = ⌊k̄⌋.** The C++ loop is `for(round=1; round ≤ k̄; round++)`, i.e. it runs
`floor(k̄)` times. (Our Python currently uses `round(k̄)` — a 1-round discrepancy when the fractional
part ≥ 0.5; see §7.)

⚠️ **note J — correction `Δθ` uses the TRIGGERING node's `i`, not the node being moved.** In the C++
the grid step in the correction loop is `min(1/(i+1), 0.01)` with `i` = the node that triggered the
correction, held constant while sweeping all `v`. For real networks the trigger fires at large `i`
(degrees ≥ 10 ⇒ many nodes already placed) so `Δθ = 0.01` in both conventions; it only matters on
tiny graphs.

---

## 6. Sanity checks

- **Karate club** (γ supplied, T fixed): our Python (`hypermap_init`) reproduces the C++
  `coordinates_embedding.txt` to numerical tolerance. Use it as a regression fixture.
- **ζ-invariance of angles:** rerun with ζ = 2; angles must be identical, radii scale by 1/ζ.
- **Single hub:** node 1 sits at r = 0, R₁ → +∞ ⇒ `CONN_PROB ≈ 1` to everyone (it is the global hub).
- **`numCN` prefix:** check the `(m̄_i ≥ i−1)` flags really form a prefix of the degree-sorted order;
  if not, the count-then-prefix shortcut (note D) silently mislabels nodes.

---

## 7. The refinement layer — additions and consistency

`hypegrl` keeps the greedy method above as a faithful warm start
({class}`~hypegrl.embedders.hypermap.HyperMapEmbedder`'s
``_hypermap_init.py``) and then adds a **Riemannian-Adam refinement on the
Poincaré ball** ({meth}`~hypegrl.embedders.hypermap.HyperMapEmbedder.fit`),
lifting the 2-D layout to arbitrary dimension `d`. The original method has no
such step (cf. how D-Mercator adds its own refinement, {doc}`dmercator`).

The refinement uses the paper's **per-node** thresholds, consistent with this specification. The whole
pipeline (`_hypermap_init`, the refinement loss `fermi_dirac_nll` /
{meth}`~hypegrl.embedders.hypermap.HyperMapEmbedder.distance`, and
{meth}`~hypegrl.embedders.hypermap.HyperMapEmbedder.decode`) operates in **degree-descending
(`nodes_sorted`) order**, and every pair `(i,j)` uses the **younger** node's threshold `R[max(i,j)]`
(note H). Three points worth keeping in mind:

- The refinement loss takes the **full** `(N,)` vector of per-node `R_i` and forms the per-pair matrix
  `R[max(i,j)]` (a scalar is still accepted as a legacy single global threshold).
- **Node ordering:** {func}`~hypegrl.inference.joint_optimizer.joint_optimize` builds the adjacency
  `A` from the graph in its node-iteration order, but `X` / `R` / embeddings are in `nodes_sorted`
  order. {meth}`~hypegrl.embedders.hypermap.HyperMapEmbedder.fit` therefore relabels the graph (and
  remaps `unknown_edges`) into `nodes_sorted` order before optimising, so each hyperbolic distance is
  paired with the right adjacency entry. Regression guard:
  ``test_hypermap_refinement_adjacency_matches_embedding_order`` (and the symptom if it breaks:
  `loss_history[-1]` no longer matches `fermi_dirac_nll` recomputed with `A` in `nodes_sorted` order).

⚠️ **Still open:** {meth}`~hypegrl.embedders.hypermap.HyperMapEmbedder.structural_similarity` returns
`nx.to_numpy_array(G)` in **original** node order, whereas `decode(embeddings())` is in `nodes_sorted`
order. Any caller comparing the two (reconstruction metrics) must reconcile the orders.

⚠️ **Two small fidelity gaps in corrections** (negligible on real graphs, break exact C++ parity on
tiny ones): rounds use `round(k̄)` vs C++ `floor(k̄)` (note I); the grid step uses the per-node
`1/(jj+1)` vs C++'s triggering-node `1/(i+1)` (note J).

Everything else in `_hypermap_init.py` (radius/threshold formulas, CN integration, FD likelihood,
tie-breaking, `k_speedup`, correction triggers) matches the C++ reference.
