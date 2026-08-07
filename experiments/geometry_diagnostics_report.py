"""Run :mod:`experiments.geometry_diagnostics` across every dataset in
:mod:`experiments.datasets`, and print/save a comparison table.

Each dataset loader in ``experiments/datasets.py`` is tried in turn; a
loader that needs a dependency or download unavailable in the current
environment (``torch_geometric`` for PolBlogs/Airports, network access for
OpenFlights) is reported as skipped rather than aborting the whole run, so
this is safe to run with a partial ``experiments/requirements.txt`` install.

Run:
    python experiments/geometry_diagnostics_report.py
"""
import sys
import warnings
from pathlib import Path
from typing import Callable, Optional

import networkx as nx

REPO = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, REPO)

from geometry_diagnostics import diagnose, format_report 

RESULTS = Path(REPO) / "experiments" / "results"


def _largest_component(G: nx.Graph) -> nx.Graph:
    """diagnose() requires connected input; every loader below is already
    documented as returning the largest connected component except
    balanced_tree_graph (always connected) and single_cell_graph (connected
    by construction, via its component-joining step) -- this is a no-op
    safety net for those, not a correction to any of them."""
    if nx.is_connected(G):
        return G
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()


# ----------------------------------------------------------------------
# Dataset registry: name -> zero-arg loader returning nx.Graph (or
# (nx.Graph, labels), in which case only the graph is used here).
# ----------------------------------------------------------------------

def _dataset_loaders() -> dict:
    from experiments.datasets import (
        OFFICIAL_SETTINGS,
        airports_graph,
        balanced_tree_graph,
        openflights_graph,
        polblogs_graph,
        single_cell_graph,
    )

    loaders: dict[str, Callable[[], object]] = {
        "balanced_tree": lambda: balanced_tree_graph(),
    }
    for name, cfg in OFFICIAL_SETTINGS.items():
        loaders[f"single_cell/{name}"] = (
            lambda name=name, cfg=cfg: single_cell_graph(
                name, k=cfg["k"], n_pca=cfg["n_pca"]
            )
        )
    loaders["polblogs"] = lambda: polblogs_graph()[0]
    for region in ("USA", "Brazil", "Europe"):
        loaders[f"airports/{region}"] = lambda region=region: airports_graph(region)[0]
    loaders["openflights"] = lambda: openflights_graph()
    return loaders


def run_report(
    datasets: Optional[list] = None,
    n_hyperbolicity_samples: int = 20_000,
    n_ricci_edges: int = 500,
    seed: int = 0,
) -> dict:
    """
    Run :func:`~experiments.geometry_diagnostics.diagnose` on every
    requested dataset (default: all of them).

    Returns ``{name: diagnose()-report}`` for datasets that loaded
    successfully, plus a separate ``_skipped: {name: reason}`` entry for
    ones that didn't (missing dependency, network/data unavailable, etc.).
    """
    loaders = _dataset_loaders()
    names = datasets if datasets is not None else list(loaders.keys())

    reports: dict = {}
    skipped: dict = {}
    for name in names:
        print(f"--- {name} ---", flush=True)
        try:
            G = loaders[name]()
            G = _largest_component(G)
            reports[name] = diagnose(
                G, n_hyperbolicity_samples=n_hyperbolicity_samples,
                n_ricci_edges=n_ricci_edges, seed=seed,
            )
            print(format_report(name, reports[name]))
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            skipped[name] = reason
            print(f"  SKIPPED ({reason})\n")

    reports["_skipped"] = skipped
    return reports


def format_summary_table(reports: dict) -> str:
    """Render one Markdown row per successfully-diagnosed dataset."""
    lines = [
        "| Dataset | N | E | mean deg | clustering | gamma | delta/diam | "
        "Ricci mean | Verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in reports.items():
        if name == "_skipped":
            continue
        dc, pl, hyp, ricci, rec = (
            r["degree_clustering"], r["powerlaw"], r["hyperbolicity"], r["ricci"],
            r["recommendation"],
        )
        lines.append(
            f"| {name} | {dc['n']} | {dc['m']} | {dc['mean_degree']:.1f} | "
            f"{dc['clustering']:.3f} | {pl['gamma']:.2f} | "
            f"{hyp['delta_normalized']:.3f} | {ricci['mean']:.3f} | "
            f"**{rec['verdict']}** ({rec['score']}/{rec['max_score']}) |"
        )
    skipped = reports.get("_skipped", {})
    if skipped:
        lines.append("")
        lines.append("Skipped: " + ", ".join(f"{k} ({v})" for k, v in skipped.items()))
    return "\n".join(lines)


if __name__ == "__main__":
    import json

    warnings.filterwarnings("ignore")
    reports = run_report()
    RESULTS.mkdir(parents=True, exist_ok=True)

    table = format_summary_table(reports)
    (RESULTS / "geometry_diagnostics.md").write_text(table + "\n")
    # `nan`/`inf` are not valid JSON; allow_nan=True (json's default) emits
    # the non-standard NaN/Infinity tokens Python's own json.load reads back
    # fine, which is all this file needs to round-trip for.
    (RESULTS / "geometry_diagnostics.json").write_text(json.dumps(reports, indent=2))

    print("\n=== SUMMARY ===")
    print(table)
