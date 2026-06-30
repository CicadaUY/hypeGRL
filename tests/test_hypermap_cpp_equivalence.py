"""
Equivalence tests: our HyperMap init vs. the official C++ implementation.
========================================================================

Goal
----
Give high confidence that ``hypegrl.embedders._hypermap_init.hypermap_init``
reproduces the *original* HyperMap "fast hybrid" method (Papadopoulos, Aldecoa &
Krioukov, PRE 92, 022807, 2015), by checking it against golden outputs from the
official C++ binary. The fixtures live in ``tests/fixtures/hypermap_cpp/`` and are
committed, so this suite needs **no C++ toolchain** to run (see that directory's
``README.md`` for provenance and how to regenerate them).

Why this can be a near-exact coordinate check
---------------------------------------------
Unlike D-Mercator (whose stochastic stages can't be reproduced bit-for-bit, hence
its Tier-0 closed-form-only checks), HyperMap's greedy MLE is **deterministic**:
node 1 is pinned at θ=π and every later angle is the argmax of a fixed grid
search. So given the *same node arrival order*, the whole computation is
reproduced, and we can assert that the actual (θ, r) coordinates match to the C++
output's printed precision.

The arrival-order gauge
-----------------------
The only freedom is the order itself. The C++ ranks nodes with a **non-stable**
``std::sort`` by degree; our Python uses a *stable* sort (ties → ascending node
id). On karate 22/34 nodes share a degree with another, so the two can pick
different — but equally valid — arrival orders, which changes the embedding. That
is a gauge difference, not a bug. We remove it here: the fixture rows are in the
C++ arrival order (radius is strictly increasing in rank), so we recover that
order and relabel the graph so our stable sort reproduces it before comparing.
"""

from __future__ import annotations

import re
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from hypegrl.embedders._hypermap_init import hypermap_init

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "hypermap_cpp"

# Tolerances. The fixtures store coordinates to 8 decimals, so the comparison is
# bounded by that printed precision (~5e-9 rounding), not by how well the methods
# actually agree (radii match to ~1e-15, angles to ~1e-8 in memory). 1e-6/1e-7
# leave headroom for the rounding while still catching real algorithmic drift (a
# single flipped grid step is ~1e-2 rad; a radius bug is order 0.1+).
ANGLE_ATOL = 1e-6
RADIUS_ATOL = 1e-7


# ── Fixture parsing ──────────────────────────────────────────────────────────

def _parse_fixture(path: Path) -> tuple[dict, np.ndarray]:
    """Parse a golden file into (header dict, data array).

    Header values are read from the ``# key: value`` comment lines; ``data`` is
    the ``(N, 3)`` array of ``id theta r`` rows, in C++ arrival order.
    """
    header: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("#"):
            break
        m = re.match(r"#\s*([^:]+):\s*(.+)", line)
        if m:
            header[m.group(1).strip()] = m.group(2).strip()
    data = np.loadtxt(path)  # comment lines skipped automatically
    return header, data


def _params_from_header(header: dict) -> dict:
    """Coerce the header strings into the kwargs ``hypermap_init`` expects."""
    return dict(
        gamma=float(header["gamma"]),
        T=float(header["T"]),
        zeta=float(header["zeta"]),
        m_in=float(header["m_in"]),
        k_speedup=int(header["k_speedup"]),
        corrections=header["corrections"].lower() == "yes",
    )


def _build_karate() -> nx.Graph:
    """Karate club relabeled 0..N-1 by sorted node id (matches the generator)."""
    G = nx.karate_club_graph()
    return nx.relabel_nodes(G, {n: i for i, n in enumerate(sorted(G.nodes()))})


# ── Cases: one per committed fixture ─────────────────────────────────────────

_FIXTURES = sorted(FIXTURE_DIR.glob("karate_T*.txt"))
_IDS = [p.stem for p in _FIXTURES]


@pytest.mark.parametrize("fixture", _FIXTURES, ids=_IDS)
def test_hypermap_init_matches_cpp(fixture):
    """Our greedy init reproduces the C++ (θ, r) once the arrival order is fixed.

    Steps: read the golden output, recover the C++ arrival order from it, relabel
    the karate graph so our *stable* degree-sort reproduces that exact order, run
    ``hypermap_init``, then compare every node's radius and angle (circular) to
    the C++ values.
    """
    header, data = _parse_fixture(fixture)
    params = _params_from_header(header)

    G0 = _build_karate()
    N = G0.number_of_nodes()

    arrival = data[:, 0].astype(int).tolist()       # rows are in arrival order
    gold_theta = {int(i): t for i, t, _ in data}
    gold_r = {int(i): r for i, _, r in data}

    # Sanity: the radius column must be non-decreasing down the rows (the property
    # that lets us treat row order as arrival order).
    assert np.all(np.diff(data[:, 2]) >= -RADIUS_ATOL), \
        "fixture rows are not in arrival (radius-ascending) order"

    # Relabel so node id == arrival position; building the graph fresh in that
    # order makes our stable degree-sort break ties by arrival position, i.e.
    # reproduce the C++ order exactly.
    relabel = {orig: pos for pos, orig in enumerate(arrival)}
    Gr = nx.Graph()
    Gr.add_nodes_from(range(N))
    Gr.add_edges_from((relabel[u], relabel[v]) for u, v in G0.edges())

    thetas, r_final, nodes_sorted, _ = hypermap_init(Gr, verbose=False, **params)

    # Guard: our sort must have reproduced the forced arrival order.
    assert nodes_sorted == list(range(N)), \
        "stable degree-sort did not reproduce the forced arrival order"

    inv = {pos: orig for orig, pos in relabel.items()}
    for idx in range(N):
        orig = inv[idx]
        assert r_final[idx] == pytest.approx(gold_r[orig], abs=RADIUS_ATOL)
        # circular angular distance in [0, pi]
        d = np.pi - abs(np.pi - abs(thetas[idx] - gold_theta[orig]))
        assert abs(d) < ANGLE_ATOL, (
            f"node {orig}: angle {thetas[idx]:.8f} vs C++ {gold_theta[orig]:.8f}"
        )
