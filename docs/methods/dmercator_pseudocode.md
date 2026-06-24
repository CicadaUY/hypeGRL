# D-Mercator — pseudocode for debugging

```{note}
This page is the algorithmic specification that
``hypegrl/embedders/_dmercator_init.py`` implements faithfully — the
*initialization* (Stages 1–4) of
{class}`~hypegrl.embedders.dmercator.DMercatorEmbedder`. Stage 6 (the radial map
to hyperbolic coordinates) becomes the embedder's warm start, which is then
refined on the Poincaré ball (see {doc}`dmercator`). The section numbers below
are referenced from that module's comments — keep the two in sync.
```

Reconstructed from Jankowski, Allard, Boguñá, Serrano, *The D-Mercator method for the
multidimensional hyperbolic embedding of real networks* (Nat. Commun. 2023). Equation
numbers below refer to that paper. Reference C++/Python implementation:
`github.com/networkgeometry/d-mercator` — consult it for the bits flagged ⚠️ below, which the
paper underspecifies.

The embedding lives in the **S^D model** (similarity = D-sphere of radius R in R^{D+1}, plus a
hidden-degree "popularity" coordinate). The optional final map to H^{D+1} is just a radial
transform (Eq. 7). Everything before that happens in S^D.

---

## 0. Model constants and primitives

Inputs: adjacency `A` (a_ij ∈ {0,1}), similarity dimension `D`, observed degrees `k_i`,
mean degree `⟨k⟩ = (1/N) Σ_i k_i`, empirical mean local clustering `c̄_emp`.

```
R(N, D)            = ( N / (2 * π^{(D+1)/2}) * Γ((D+1)/2) )^{1/D}              # Eq. 2
μ(β, D, ⟨k⟩)       = β * Γ(D/2) * sin(Dπ/β) / ( 2 * π^{1+D/2} * ⟨k⟩ )         # Eq. 3
Δθ(v_i, v_j)       = arccos( clip( (v_i · v_j) / R^2 , -1, 1 ) )              # angular distance
χ(Δθ, κ_i, κ_j)    = R * Δθ / (μ κ_i κ_j)^{1/D}                                # Eq. 1
p_connect(Δθ,κi,κj)= 1 / ( 1 + χ(Δθ,κi,κj)^β )                                 # Eqs. 1, 15
```

Angular-measure density and its normalizer (needed everywhere an integral over Δθ appears):

```
ρ(Δθ)        = Γ((D+1)/2) / (Γ(D/2) √π) * sin^{D-1}(Δθ)        # Eq. 16, normalized on [0,π]
C_D          = Γ((D+1)/2) / (Γ(D/2) √π)                        # = 1 / ∫_0^π sin^{D-1}θ dθ
```

Note `R` and `μ` depend on `β` only through `μ` (R is fixed once N, D are fixed). **μ must be
recomputed every time β changes.**

---

## 1. Top-level driver

```
function D_MERCATOR(A, D):
    N      = number_of_nodes(A)
    R      = R(N, D)                                  # fixed
    k      = observed_degrees(A)
    ⟨k⟩    = mean(k)
    c̄_emp = mean_local_clustering(A)

    # --- Stage 1: jointly infer hidden degrees κ and inverse temperature β ---
    (κ, β) = INFER_KAPPA_AND_BETA(A, k, ⟨k⟩, c̄_emp, R, D)

    # --- Stage 2: initial angular positions via S^D-corrected Laplacian Eigenmaps ---
    V = MODEL_CORRECTED_LE(A, κ, β, R, D)             # one (D+1)-vector per node, on sphere R

    # --- Stage 3: refine angular positions by maximum likelihood ---
    V = MLE_REFINE(A, κ, β, R, D, V)

    # --- Stage 4: final readjustment of hidden degrees given the inferred positions ---
    κ = FINAL_ADJUST_KAPPA(A, κ, β, R, D, V)

    # --- Optional: map to hyperbolic coordinates H^{D+1} ---
    r = RADIAL_MAP(κ, β, R, D)                         # Eq. 7
    return (κ, β, V, r)                                # angular = V/||V||, radial = r
```

Degree-one nodes are dropped before Stages 2–3 and reinserted afterward (see §2.5).

---

## 2. Stage 1 — hidden degrees κ and inverse temperature β

These are coupled: κ depends on (β, μ); β is chosen so the model clustering matches `c̄_emp`.
Clustering is **monotone increasing in β**, so β is found by bracket-then-bisect, and the κ
inner loop is re-run for every β evaluated.

### 2.1 Outer loop on β (IV B 2)

```
function INFER_KAPPA_AND_BETA(A, k, ⟨k⟩, c̄_emp, R, D):
    β   = uniform(D, D+1)                              # initial guess; quality is independent of it
    κ   = INFER_KAPPA(k, β, μ(β,D,⟨k⟩), R, D)
    c̄  = EXPECTED_CLUSTERING(k, κ, β, μ(β,D,⟨k⟩), R, D)

    # bracket: grow β by ×1.5 until model clustering exceeds empirical
    β_lo = β
    while c̄ < c̄_emp:
        β_lo = β
        β    = 1.5 * β
        κ    = INFER_KAPPA(k, β, μ(β,D,⟨k⟩), R, D)
        c̄   = EXPECTED_CLUSTERING(k, κ, β, μ(β,D,⟨k⟩), R, D)
    β_hi = β

    # bisection on β in [β_lo, β_hi]
    repeat:
        β   = (β_lo + β_hi)/2
        μ   = μ(β, D, ⟨k⟩)
        κ   = INFER_KAPPA(k, β, μ, R, D)
        c̄  = EXPECTED_CLUSTERING(k, κ, β, μ, R, D)
        if c̄ < c̄_emp: β_lo = β else: β_hi = β
    until |c̄ - c̄_emp| < ε_c̄                           # ε_c̄ = 0.01

    return (κ, β)
```

### 2.2 Inferring hidden degrees for fixed β (IV B 1)

Adjust κ so each node's *expected* degree under the model equals its observed degree.

```
function INFER_KAPPA(k, β, μ, R, D):
    κ = copy(k)                                        # init κ_i = k_i
    repeat:
        for each node i:
            k̄_i = EXPECTED_DEGREE_INTEGRAL(i, κ, β, μ, R, D)   # Eq. 13
        ε_max = max_i | k̄_i - k_i |
        if ε_max ≤ ε: break                            # ε = 0.01
        for each degree class with value k_i:
            u  = uniform(0,1)                          # random step avoids local minima
            κ_i ← | κ_i + (k_i - k̄_i) * u |            # Eq. (IV B 1, step 3)
    return κ
```

Expected degree as an integral over the angular distribution (this is the form used during
inference, *before* positions are known):

```
function EXPECTED_DEGREE_INTEGRAL(i, κ, β, μ, R, D):                   # Eq. 13
    s = 0
    for each j ≠ i:
        s += ∫_0^π  sin^{D-1}(θ) / ( 1 + ( R θ / (μ κ_i κ_j)^{1/D} )^β )  dθ
    return C_D * s
```

⚠️ The sum is over **all** j ≠ i (degree classes can be aggregated: nodes with equal κ give the
same integrand, so loop over distinct κ-values weighted by their multiplicity for speed).

### 2.3 Distribution of angular distance given a link (IV B 2)

```
ρ(Δθ | connected; κ, κ') = p_connect(Δθ,κ,κ') · ρ(Δθ) / P_link(κ,κ')        # Eq. 14
P_link(κ, κ')            = ∫_0^π  sin^{D-1}(Δθ) / (1 + (RΔθ/(μκκ')^{1/D})^β) dΔθ   # Eq. 17
                           (up to the C_D normalizer; it cancels in Eq. 14)
```

### 2.4 Expected mean local clustering (IV B 2)

Monte-Carlo: clustering of a degree-k node = probability two of its random neighbors are linked.

```
function EXPECTED_CLUSTERING(k, κ, β, μ, R, D):
    for each degree class value kc:
        c̄(kc) = 0
        repeat m times:                                # m = 600 works for ε_c̄ = 0.01
            draw  k1, k2  ~ P(k'|kc) = k' P(k') / ⟨k⟩  # uncorrelated neighbor degrees
            draw  Δθ1 ~ ρ(Δθ | connected; κ(kc), κ(k1))    # Eq. 14
            draw  Δθ2 ~ ρ(Δθ | connected; κ(kc), κ(k2))    # Eq. 14
            place node, neighbor1, neighbor2 on the D-sphere so that
                 angle(node, n1) = Δθ1  and  angle(node, n2) = Δθ2,
                 with the two neighbors otherwise random           ⚠️ see note (a)
            Δθ12  = arccos( v_{n1} · v_{n2} / R^2 )
            c̄(kc) += p_connect(Δθ12, κ(k1), κ(k2)) / m
    return  Σ_kc  c̄(kc) * N_kc / N                     # average over degree classes
```

⚠️ note (a): "place with a given angular separation to a fixed reference" means: fix the node at,
e.g., the pole; sample n1 at polar angle Δθ1 with uniform azimuth on the (D−1)-sphere; same for
n2; then Δθ12 is the great-circle angle between them. For D=1 this collapses to angles on a
circle. Getting the **uniform azimuth on S^{D−1}** right matters — a common bug is sampling the
azimuth non-uniformly, which biases Δθ12 and therefore the inferred β.

### 2.5 Degree-one nodes

Removed before LE/MLE (they carry no geometric info). Reinsert each degree-one node i (neighbor
j) afterward by drawing `Δθ_ij ~ ρ(Δθ | connected; κ_i, κ_j)` (Eq. 14) and placing i at that
angular separation from j (uniform azimuth).

---

## 3. Stage 2 — S^D-corrected Laplacian Eigenmaps (IV B 3)

Goal: a good initial guess for the angular positions by solving a weighted-Laplacian
eigenproblem whose target distances come from the model.

```
function MODEL_CORRECTED_LE(A, κ, β, μ, R, D):
    # 1. expected angular distance for each connected pair
    for each edge (i,j) in A:
        ⟨Δθ_ij⟩ = [ ∫_0^π θ sin^{D-1}θ / (1+(Rθ/(μκ_iκ_j)^{1/D})^β) dθ ]
                  / [ ∫_0^π   sin^{D-1}θ / (1+(Rθ/(μκ_iκ_j)^{1/D})^β) dθ ]      # Eqs. 19–20
        d_ij    = 2 * sin( ⟨Δθ_ij⟩ / 2 )               # chord length on UNIT sphere   # Eq. 5

    # 2. weights (only connected pairs contribute)
    t   = mean( { d_ij^2 : (i,j) ∈ edges } )            # ⚠️ MEAN of squared chords, see note (e)
    ω_ij = a_ij * exp( - d_ij^2 / t )                   # Eq. 5  (0 for non-edges)

    # 3. RANDOM-WALK-normalized Laplacian and its spectrum   # ⚠️ see note (f)
    s_i = Σ_j ω_ij                                      # row strength (weighted degree)
    L   = I - D_s^{-1} Ω                                # L_ij = δ_ij - ω_ij / s_i  (NOT I_diag - Ω)
    solve the (general, non-symmetric) eigenproblem of L

    # 4. take D+1 NON-NULL eigenvectors (request D+2 smallest, drop the trivial λ≈0 constant one)
    U = [ u_1, u_2, ..., u_{D+1} ]                       # each u_m ∈ R^N
    V_LE[i] = ( U[i,1], U[i,2], ..., U[i,D+1] )          # node i's (D+1)-vector

    # 5. project onto the D-sphere of radius R
    for each node i:
        V[i] = R * V_LE[i] / || V_LE[i] ||
    return V
```

⚠️ Pitfalls here:
- note (e): **the official code uses `t = ⟨d_ij²⟩` (mean of squared chord lengths over connected
  pairs), NOT the variance** the paper text claims. Concretely it accumulates `Σ_edges 2·d² /
  (2·#edges)`. If you implemented the literal variance (subtracting the mean²) your weights are
  off — switch to the raw second moment.
- note (f): **the code solves the random-walk-normalized eigenproblem `(I − D_s^{-1}Ω) x = λx`,
  not the unnormalized `(diag(s) − Ω) x = λx`** the paper writes. It builds `−ω_ij/s_i` off-diagonal
  with 1 on the diagonal (a row-stochastic transform of the generalized problem). Because it's
  non-symmetric, the reference uses a general (complex) solver. You can instead solve the
  *symmetric* normalized version `I − D_s^{-1/2} Ω D_s^{-1/2}` and get real eigenvectors with the
  same eigenvalues — just be consistent.
- **Eigenvector count.** Target space is R^{D+1}, so you need **D+1** non-trivial eigenvectors,
  not D. The code requests the **D+2** smallest and discards the trivial one. (For D=1 → Mercator
  that's 2 kept eigenvectors = a point on the circle.)
- **Drop the null vector.** L has a ~0 eigenvalue (constant eigenvector). Skip it; take the next D+1.
- **Sign/rotation gauge.** LE eigenvectors are defined up to sign (and up to rotation within
  degenerate eigenspaces) — the same spectral ambiguity you know from ASE. Fine here, because
  Stage 3 (MLE) and any downstream comparison must be invariant to global O(D+1)
  rotations/reflections of the sphere. Don't try to fix signs at this stage.
- **`d_ij` is a UNIT-sphere chord** (Eq. 5 uses `2 sin(⟨Δθ⟩/2)`, no R). Keep the radius out of
  the weight; reintroduce R only at the projection step.

---

## 4. Stage 3 — likelihood maximization (IV B 4)

Visit nodes in a fixed order; for each, propose candidate positions near the mean of its
neighbors and keep the most likely.

```
function MLE_REFINE(A, κ, β, μ, R, D, V):
    order = onion_decomposition(A)                      # layers outer→inner; random within a layer
    repeat until local log-likelihood plateaus:
        for each node i in order:
            # (a) hidden-degree-weighted mean of neighbor positions
            v̄_i = Σ_{j ∈ N(i)}  (1/κ_j^2) * v_j         # Eq. 21  (NOT normalized; off-sphere)

            # (b) propose candidates around v̄_i (project v̄_i to the sphere first)
            v̄_i = R * v̄_i / ||v̄_i||
            Δθ_max = max_{j ∈ N(i)} angle(v̄_i, v_j)
            σ      = max( π , Δθ_max ) / 2               # = max(π/2, Δθ_max/2)   # Eq. 22
            n_cand = round( 100 * max( ln N , 1 ) )
            candidates = {}
            repeat n_cand times:                          # multivariate normal on the UNIT sphere
                c[m] = σ * N(0,1) + (v̄_i)[m] / R   for each coord m = 0..D     # note (c)
                c    = R * c / ||c||                       # renormalize to radius R
                candidates ∪= { c }
            candidates ∪= { current v_i }                 # incumbent is the baseline

            # (c) pick the candidate maximizing the LOCAL log-likelihood
            best = argmax_{c ∈ candidates} LOCAL_LOGLIK(i, c, V, κ, β, μ, R, D)
            v_i = best
    return V
```

```
function LOCAL_LOGLIK(i, pos, V, κ, β, μ, R, D):                        # Eq. 23
    ll = 0
    for each j ≠ i:
        Δθ = arccos( (pos · v_j) / R^2 )
        p  = p_connect(Δθ, κ_i, κ_j)
        ll += a_ij * ln(p) + (1 - a_ij) * ln(1 - p)
    return ll
```

⚠️ Pitfalls here (resolved against the official `refine_angle`):
- note (b) — RESOLVED: σ is `max(π, Δθ_max)/2 = max(π/2, Δθ_max/2)`. The code's constant
  `MIN_TWO_SIGMAS_NORMAL_DIST = π` is the floor on **2σ**, i.e. σ ≥ π/2. It is a `max`, not a
  `min`; the broad spread is deliberate (and the comment notes higher D wants larger σ).
- note (c) — RESOLVED: the sampling is literally Cartesian. Take v̄_i, **normalize to the unit
  sphere** (`v̄_i/R`), add i.i.d. `N(0, σ²)` to each of the D+1 coordinates, then **renormalize to
  radius R**. σ is therefore a standard deviation in unit-sphere Cartesian coordinates even though
  it's derived from an angle — the units "mismatch" is real but harmless because of the renormalize
  step. Don't sample an angle and rotate; match the code's add-Gaussian-then-renormalize.
- **v̄_i must be projected to the sphere** before use (`normalize_and_rescale_vector`). The raw
  `Σ_j v_j/κ_j²` is off-sphere.
- **Incumbent is the baseline.** Initialize `best` to the current position's local log-likelihood;
  a candidate replaces it only if strictly better. (So a node can stay put.)
- `LOCAL_LOGLIK` sums over all j (full row), not just neighbors — both link and non-link terms
  matter. Only the terms involving i change when i moves, so cache the rest.

---

## 5. Stage 4 — final hidden-degree readjustment (IV B 5)

Now positions are known, so use the **actual** angular distances (a sum, not an integral):

```
function FINAL_ADJUST_KAPPA(A, κ, β, μ, R, D, V):
    repeat:
        for each node i:
            k̄_i = Σ_{j ≠ i}  1 / ( 1 + ( R Δθ(v_i,v_j) / (μ κ_i κ_j)^{1/D} )^β )   # Eq. 24
        ε_max = max_i | k̄_i - k_i |
        if ε_max ≤ ε: break
        for each node i:
            u = uniform(0,1)
            κ_i ← | κ_i + (k_i - k̄_i) * u |      # '+', same routine as Stage 1 — see note (d)
    return κ
```

✅ note (d) — RESOLVED: the official code uses **`+`** here, and in fact Stage 1 and Stage 4 call
the **same routine** (`infer_kappas_given_beta_for_all_vertices`), which does
`kappa[v] += (degree[v] - expected[v]) * u; kappa[v] = fabs(kappa[v]);`. So the `−` printed in the
paper's IV B 5 **is a typo** — both stages add the residual. If your κ diverged in Stage 4, that
sign was the bug.

⚠️ Also: the operative **convergence tolerance is on `max_i |k̄_i − k_i|`** and the per-vertex
post-inference routine uses a threshold of **0.5** (`NUMERICAL_CONVERGENCE_THRESHOLD_3`), not the
`ε = 0.01` the paper quotes for the degree-class inference. Iteration is also capped
(`KAPPA_MAX_NB_ITER_CONV = 500`). Match whichever stage you're debugging.

---

## 6. Optional — map to hyperbolic coordinates (IV A, Eq. 7)

```
function RADIAL_MAP(κ, β, μ, R, D):
    κ_0 = min_i κ_i                                     # reference (smallest hidden degree)
    R̂   = 2 * ln( 2R / (μ κ_0^2)^{1/D} )                # Eq. 7
    for each node i:
        r_i = R̂ - (2/D) * ln( κ_i / κ_0 )
    return r
```

Final hyperbolic embedding of node i: radial coordinate `r_i`, angular coordinate `v_i/||v_i||`
on S^D. With these, `x_ij = r_i + r_j + 2 ln(Δθ_ij/2)` approximates the hyperbolic distance and
`p_ij = 1/(1 + e^{(β/2)(x_ij − R̂)})` reproduces Eq. 1.

---

## 7. Sanity checks (D = 1 reduces to Mercator)

Run your code at `D = 1` and compare against known Mercator formulas — a clean way to localize bugs:

- `R(N,1) = N / (2π)`.
- `μ(β,1,⟨k⟩) = β sin(π/β) / (2π⟨k⟩)`  (Γ(1/2)=√π cancels).
- Angular distribution `ρ(Δθ)` is **uniform** on [0,π] at D=1, and becomes increasingly peaked at
  π/2 as D grows (paper Fig. S1) — a good unit test for your `ρ(Δθ)` sampler.
- At D=1, "place at angular separation Δθ" is just placing on a circle; `arccos((v_i·v_j)/R^2)`
  must equal the circular angle. If it doesn't, your sphere-sampling or the R^2 normalization is off.

## 8. Numerical-integration notes

- Eqs. 13, 17, 19–20 have no closed form for general D; the paper uses the **trapezoid rule**
  over θ ∈ [0,π] (SI §III). Integration resolution affects pre-factors only, not the O(N²) scaling.
- Watch the integrand near θ→0: `sin^{D-1}θ → 0` so it's well-behaved, but `χ→0` makes
  `p_connect→1`; make sure the grid resolves the sigmoidal transition around `χ≈1`
  (i.e. `θ ≈ (μκ_iκ_j)^{1/D}/R`), which is where the integrand has its structure.
- Overall complexity is **O(N²)** (same scaling as Mercator; D-Mercator only adds pre-factors
  from the numerical integrals).

---

### Quickest-to-check failure points, in order
1. `μ` not recomputed when `β` changes (§0). D=1 must reduce to `μ = β sin(π/β)/(2π⟨k⟩)`
   (the code's `calculateMu()`); the general form is `calculate_mu(dim)`.
2. LE: wrong eigenvector count (need D+1; code requests D+2 and drops the trivial one); using the
   unnormalized Laplacian instead of the random-walk-normalized `I − D_s^{-1}Ω` (note f); or using
   the variance instead of `t = ⟨d²⟩` for the weight scale (note e).
3. MLE candidates: not renormalizing `σ·N(0,1) + v̄_i/R` back to radius R, or not seeding `best`
   with the incumbent position (§4, notes b/c).
4. Stage-4 κ-update sign — it's `+`, not `−` (note d); paper IV B 5 typo.
5. Non-uniform azimuth when sampling on S^{D−1} for the clustering MC (note a, §2.4) → biased β.
6. Integer division in `R` / dimension prefactors — use `(D+1)/2.0`, `D/2.0` (float).
