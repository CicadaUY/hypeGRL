"""Regenerate the HyperMap C++ golden fixtures.

Runs the **official C++ HyperMap binary** (Papadopoulos, Aldecoa, Krioukov,
"Network Geometry Inference using Common Neighbors", PRE 92, 022807, 2015) on the
Zachary karate club graph at a few temperatures and writes one golden coordinate
file per temperature, with a provenance header, into this directory.

This script needs the compiled C++ binary and is meant to be run **manually** by a
developer; the test suite itself only reads the committed ``.txt`` outputs and
needs no C++ toolchain (mirrors ``../dmercator_cpp``).

Usage:
    python generate_fixtures.py [--binary /path/to/hypermap]

The default binary path is the local checkout used during development; override
with ``--binary`` on another machine. See ``README.md`` for provenance and the
gamma/parameter choices.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import networkx as nx
import numpy as np

HERE = Path(__file__).parent
DEFAULT_BINARY = (
    "/home/flarroca/Nextcloud/facultad/proyectos/2024_csic_i+d/code/"
    "2015_code_hypermap/hypermap"
)

# Temperatures to embed (kept small: low / mid / high clustering regimes).
TEMPERATURES = (0.3, 0.5, 0.7)

# Fixed model inputs, matching how the embedding is run elsewhere in the project.
M_IN = 1
ZETA = 1
K_SPEEDUP = 0
CORRECTIONS = "yes"


def estimate_gamma(G: nx.Graph, k_min: int = 4) -> float:
    """Power-law exponent estimate (Clauset et al. 2009, Eq. 3.7).

    This is the same estimator used to pick gamma for the embedding; gamma is an
    *input* to HyperMap, so it is recorded in each fixture header for provenance.
    """
    degrees = np.array([d for _, d in G.degree()])
    degrees = degrees[degrees >= k_min]
    n = len(degrees)
    return 1.0 + n / np.sum(np.log(degrees / (k_min - 0.5)))


def build_karate() -> nx.Graph:
    """Karate club relabeled to 0..N-1 by sorted node id (identity here)."""
    G = nx.karate_club_graph()
    return nx.relabel_nodes(G, {n: i for i, n in enumerate(sorted(G.nodes()))})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", default=DEFAULT_BINARY,
                    help="Path to the compiled C++ hypermap binary.")
    args = ap.parse_args()

    binary = Path(args.binary)
    if not binary.exists():
        raise SystemExit(f"C++ binary not found: {binary}\n"
                         "Build it (make) and pass --binary, see README.md.")

    G = build_karate()
    gamma = estimate_gamma(G, k_min=4)

    with tempfile.TemporaryDirectory() as tmp:
        edgelist = os.path.join(tmp, "karate.edgelist")
        with open(edgelist, "w") as f:
            for u, v in G.edges():
                if u != v:
                    f.write(f"{u} {v}\n")

        for T in TEMPERATURES:
            out = os.path.join(tmp, f"coords_T{T}.txt")
            subprocess.run(
                [str(binary), "-i", edgelist, "-g", repr(gamma), "-t", str(T),
                 "-z", str(ZETA), "-k", str(K_SPEEDUP), "-m", str(M_IN),
                 "-o", out, "-c", CORRECTIONS],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            coords = np.loadtxt(out)  # columns: id theta r, in arrival order

            dst = HERE / f"karate_T{T}.txt"
            with open(dst, "w") as f:
                f.write(
                    "# HyperMap reference output (official C++ implementation)\n"
                    "# Papadopoulos, Aldecoa, Krioukov, PRE 92, 022807 (2015)\n"
                    "# graph: networkx.karate_club_graph "
                    "(relabeled 0..N-1 by sorted node id)\n"
                    f"# gamma: {gamma!r}\n"
                    f"# T: {T}\n"
                    f"# m_in: {M_IN}\n"
                    f"# zeta: {ZETA}\n"
                    f"# k_speedup: {K_SPEEDUP}\n"
                    f"# corrections: {CORRECTIONS}\n"
                    "# columns: id theta r "
                    "(rows in C++ arrival order = degree-descending rank)\n"
                )
                for nid, theta, r in coords:
                    f.write(f"{int(nid)} {theta:.8f} {r:.8f}\n")
            print(f"wrote {dst}")


if __name__ == "__main__":
    main()
