"""
Tier-0 equivalence tests: our D-Mercator init vs. the official C++ implementation.
===================================================================================

Goal
----
Give high confidence that ``hypegrl.embedders._dmercator_init`` reproduces the
*original* D-Mercator method, by checking it against golden outputs from the
official C++ implementation (version 0.9). The fixtures live in
``tests/fixtures/dmercator_cpp/`` and are committed, so this suite needs **no C++
toolchain** to run (see that directory's ``README.md`` for provenance and how to
regenerate them).

Scope — what "equivalent" means here
------------------------------------
D-Mercator has both deterministic and stochastic stages. Two facts bound what we
can assert:

1. **Only the initialisation is comparable.** Our ``DMercatorEmbedder`` adds a
   Poincaré-ball gradient *refinement* that the C++ does not have, so the
   meaningful comparison is C++ ``embed()`` ↔ ``dmercator_init(...)`` (equivalently
   ``DMercatorEmbedder(n_steps=0)``).

2. **Bit-exact equality is impossible for the stochastic stages.** The C++ has its
   own RNG stream driving the Monte-Carlo clustering estimate, the random candidate
   angles in likelihood maximisation, and the random initial positions. We cannot
   reproduce that stream in NumPy.

So this is **Tier 0**: we verify the *deterministic, closed-form identities* the
C++ must satisfy **given its own inferred** ``β`` and ``κ``. These pin down the
Γ/π global-constant machinery and the S^D→H^{D+1} radial map exactly, in both
D=1 (Mercator) and D=2:

* sphere radius          ``R   = compute_R(N, D)``
* hyperbolic offset      ``R̂``  and the radial map ``r_i = R̂ − (2/D)·ln(κ_i/κ_min)``
* minimum hidden degree  ``κ_min = min_i κ_i``
* model parameter        ``μ   = compute_mu(β, D, ⟨k⟩)``   (see the β→D caveat below)

The stochastic stages (angular positions, the κ/β inference loop) are out of
Tier-0 scope and are not checked here.
"""

from __future__ import annotations

import re
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from hypegrl.embedders._dmercator_init import compute_mu, compute_R, radial_map

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "dmercator_cpp"

# The C++ writes parameters to ~6 significant figures, so closed-form quantities
# can only be expected to agree to roughly that relative precision.
RTOL = 1e-4


# ── Fixture parsing ──────────────────────────────────────────────────────────
# Header keys differ between the D=1 ("radius_S1"/"radius_H2") and D≥2
# ("radius_S^D"/"radius_H^D+1") output formats; map both onto canonical names.
_HEADER_KEYS = {
    "nb. vertices": "N",
    "beta": "beta",
    "mu": "mu",
    "radius_S1": "R",
    "radius_S^D": "R",
    "radius_H2": "R_hat",
    "radius_H^D+1": "R_hat",
    "kappa_min": "kappa_min",
}


def _parse_inf_coord(path: Path) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Parse a C++ ``.inf_coord`` file.

    Returns ``(params, kappa, r)`` where ``params`` holds the header globals
    (N, beta, mu, R, R_hat, kappa_min), ``kappa`` is the inferred-hidden-degree
    column and ``r`` is the inferred hyperbolic-radius column.

    Column layout (see fixtures/README.md):
      * D=1: ``Vertex  Inf.Kappa  Inf.Theta  Inf.Hyp.Rad.``        -> radius is col 3
      * D≥2: ``Vertex  Inf.Kappa  Inf.Hyp.Rad  Inf.Pos.1 ...``     -> radius is col 2
    """
    params: dict[str, float] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("#"):
            break
        # Strip leading "#", whitespace, and the "- " bullet used in the header.
        s = line.lstrip("#").strip().lstrip("-").strip()
        for key, canonical in _HEADER_KEYS.items():
            if s.startswith(key):
                rest = s[len(key):].lstrip(": ").strip()
                m = re.match(r"[-+0-9.eE]+", rest)
                if m:
                    params[canonical] = float(m.group())
    data = np.loadtxt(path)  # comment lines are skipped automatically
    kappa = data[:, 1]
    # The number of value columns tells us the format (4 cols => D=1).
    radius_col = 3 if data.shape[1] == 4 else 2
    r = data[:, radius_col]
    return params, kappa, r


# ── Cases ────────────────────────────────────────────────────────────────────
# Each case carries enough to recompute the closed-form quantities independently:
# the fixture file, the similarity dimension D, and a factory for the *exact*
# graph the C++ was run on (so we can recover the observed mean degree ⟨k⟩).
#
# ``mu_well_conditioned`` flags whether β sits comfortably above D. When β → D the
# closed-form μ ∝ sin(Dπ/β) → 0 and becomes hypersensitive to β; the balanced
# tree lands there (β ≈ 1.0067 for D=1), so its *printed* μ is unreliable — see
# ``test_mu_degenerate_limit_is_an_upstream_artifact``.
class _Case:
    def __init__(self, cid, filename, D, graph_factory, mu_well_conditioned):
        self.id = cid
        self.path = FIXTURE_DIR / filename
        self.D = D
        self.graph_factory = graph_factory
        self.mu_well_conditioned = mu_well_conditioned

    @property
    def avg_degree(self) -> float:
        G = self.graph_factory()
        return 2.0 * G.number_of_edges() / G.number_of_nodes()


CASES = [
    _Case("karate_d1", "karate_d1.inf_coord", 1, nx.karate_club_graph, True),
    _Case("karate_d2", "karate_d2.inf_coord", 2, nx.karate_club_graph, True),
    _Case("tree_d1", "balanced_tree_d1.inf_coord", 1,
          lambda: nx.balanced_tree(2, 4), False),
]
_IDS = [c.id for c in CASES]


@pytest.mark.parametrize("case", CASES, ids=_IDS)
def test_sphere_radius_R_matches_cpp(case):
    """R(N, D) is a pure closed form in N and D — must match the C++ exactly.

    For D=1 this is the Mercator identity R = N/(2π); for D=2 it is the full
    Γ/π expression. Mismatches here localise bugs in the global-constant
    machinery (see §7 of the pseudo-code).
    """
    params, _, _ = _parse_inf_coord(case.path)
    R_ours = compute_R(int(params["N"]), case.D)
    assert R_ours == pytest.approx(params["R"], rel=RTOL)


@pytest.mark.parametrize("case", CASES, ids=_IDS)
def test_kappa_min_matches_cpp(case):
    """κ_min reported in the header must be the minimum of the inferred κ column."""
    params, kappa, _ = _parse_inf_coord(case.path)
    assert kappa.min() == pytest.approx(params["kappa_min"], rel=RTOL)


@pytest.mark.parametrize("case", CASES, ids=_IDS)
def test_radial_map_matches_cpp(case):
    """The S^D → H^{D+1} radial map reproduces both R̂ and every node radius.

    Feeding the C++'s own (κ, μ, R) into ``radial_map`` must regenerate the
    header R̂ = 2·ln(2R/(μ κ_min²)^{1/D}) and the per-node Inf.Hyp.Rad. column,
    r_i = R̂ − (2/D)·ln(κ_i/κ_min). We use the C++'s printed μ here (rather than
    recomputing it) so the check is internally self-consistent even in the
    near-degenerate tree case.
    """
    params, kappa, r_cpp = _parse_inf_coord(case.path)
    r_ours, R_hat_ours = radial_map(kappa, params["mu"], params["R"], case.D)
    assert R_hat_ours == pytest.approx(params["R_hat"], rel=RTOL)
    # Node radii reach ~16 on these graphs; an absolute tolerance tied to the
    # 6-significant-figure printout (≈1e-3) is the natural bound.
    assert r_ours == pytest.approx(r_cpp, rel=RTOL, abs=1e-3)


_MU_CASES = [c for c in CASES if c.mu_well_conditioned]


@pytest.mark.parametrize("case", _MU_CASES, ids=[c.id for c in _MU_CASES])
def test_mu_matches_cpp_when_beta_above_D(case):
    """μ(β, D, ⟨k⟩) matches the C++ when β is comfortably above D.

    ``compute_mu`` is term-for-term identical to the C++ ``calculate_mu`` /
    ``calculateMu``; this confirms the agreement numerically on the
    well-conditioned cases (karate: β ≈ 2.18 at D=1, β ≈ 6.18 at D=2).
    """
    params, _, _ = _parse_inf_coord(case.path)
    mu_ours = compute_mu(params["beta"], case.D, case.avg_degree)
    assert mu_ours == pytest.approx(params["mu"], rel=RTOL)


def test_mu_degenerate_limit_is_an_upstream_artifact():
    """Document the one place our μ and the C++'s printed μ diverge.

    For the balanced tree the inference drives β to ≈1.0067 — the D=1 boundary
    where μ ∝ sin(π/β) → 0 and μ roughly *halves* between consecutive bisection
    iterates. The C++ header prints a μ evaluated at the **penultimate** β-iterate
    (≈1.0135) while reporting the **final** β (1.0067) in the same header, so the
    printed μ is ~2× our ``compute_mu(β_final, …)``. This is an upstream output
    inconsistency, not a formula difference.

    We assert exactly that story: (a) at the *reported* β the values disagree by
    nearly 2×, and (b) evaluating the very same formula at the penultimate β
    reproduces the printed μ to <0.5%.
    """
    case = next(c for c in CASES if c.id == "tree_d1")
    params, _, _ = _parse_inf_coord(case.path)
    avg = case.avg_degree

    # (a) At the *reported* β, our μ is ~half the printed μ.
    mu_at_reported_beta = compute_mu(params["beta"], case.D, avg)
    ratio = params["mu"] / mu_at_reported_beta
    assert 1.9 < ratio < 2.1

    # (b) The same formula at the penultimate β-iterate (from the .inf_log
    # trajectory: ... 1.0135, then the final 1.0067) reproduces the printed μ.
    mu_at_penultimate_beta = compute_mu(1.0135, case.D, avg)
    assert mu_at_penultimate_beta == pytest.approx(params["mu"], rel=5e-3)
