"""Table I: link prediction on the single-cell networks.

Edge-removal protocol (retain each edge with probability ``q``), then rank the
held-out edges against true non-links. Hyperbolic methods rank by embedding
geodesic distance (smaller = more likely link); the RDPG baseline ranks by the
adjacency-spectral connection probability. Reports F1 at ``|omega_R|`` and the
first-decile lift, averaged over seeds.
"""
import time
from typing import Callable, Optional

import networkx as nx
import numpy as np

from hypegrl.embedders import (
    DMercatorEmbedder,
    LorentzEmbeddingsEmbedder,
    PoincareEmbeddingsEmbedder,
    PoincareMapsEmbedder,
)
from hypegrl.embedders.hydra import HydraEmbedder
from hypegrl.embedders.hydra_plus import HydraPlusEmbedder
from hypegrl.embedders.hypermap import HyperMapEmbedder
from hypegrl.evaluation import (
    LinkPredictionSplit,
    candidate_scores,
    lift_curve,
    link_prediction_split,
    pairwise_distance_matrix,
    precision_recall_f1_at_k,
    roc_auc,
    training_graph,
)

# ----------------------------------------------------------------------
# Method registry: name -> factory(d, seed, device) -> fresh embedder
# ----------------------------------------------------------------------
# The dimension is a factory argument rather than baked in at 2 so a hyperbolic
# method can be run at the same d as the RDPG baseline's n_components. An
# unmatched-capacity comparison would confound "hyperbolic beats Euclidean"
# with "more dimensions beat fewer".

EmbedderFactory = Callable[[int, int, str], object]

HYPERBOLIC_METHODS: dict[str, EmbedderFactory] = {
    # curvature=None runs the reference implementation's 1-D stress-minimising search
    # for the curvature; the library default curvature=1.0 fixes it and
    # underperforms
    # (38 -> 64 F1 on ToggleSwitch). The Hydra paper fixes -k = -1 throughout its own
    # experiments (Keller-Ressel & Nargang 2021, sec. 4.1), so estimating it is an
    # implementation option the paper does not exercise.
    "Hydra+": lambda d, s, dev: HydraPlusEmbedder(
        dim=d, curvature=None, random_state=s),
    # d1_init="mercator": the original Mercator ordering-init for D=1 (d=2),
    # which the paper's original-code wrapper used; the library default "le" is
    # the paper's D-dimensional generalisation and underperforms here.
    "D-Mercator": lambda d, s, dev: DMercatorEmbedder(
        d=d, d1_init="mercator", random_state=s),
    "Poincare Embeddings": lambda d, s, dev: PoincareEmbeddingsEmbedder(
        d=d, random_state=s, device=dev),
    # n_steps=5000: the default 500 undertrains at N~640 (Myeloid F1 500->8.8,
    # 2000->38.5, 5000->49.4). ToggleSwitch/Olsson converge well before 500 and are
    # flat past it, so a single higher budget fixes Myeloid without changing them.
    # Tuned at N~640; not revalidated at the ladder's N~1200-3300.
    "Poincare Maps": lambda d, s, dev: PoincareMapsEmbedder(
        d=d, n_steps=5000, random_state=s, device=dev),
    # Available but excluded from the paper's core comparison. Hydra is closed-form
    # spectral and HyperMap's greedy init is deterministic, so both ignore `s`; both
    # are CPU-only, so both ignore `dev`.
    "Hydra": lambda d, s, dev: HydraEmbedder(dim=d, curvature=None),
    # HyperMap's greedy init is deterministic (degree-sorted), so no seed.
    # n_steps=0 = init-only (the faithful original HyperMap). Our Fermi-Dirac
    # gradient refinement degrades distance-ranked link prediction here — mildly
    # on ToggleSwitch (32.5 refined vs 40.8 init) and severely on Myeloid (37.0 vs
    # 68.3) — so the reported number is the original method without the extra stage.
    "HyperMap": lambda d, s, dev: HyperMapEmbedder(d=d, n_steps=0, verbose_init=False),
    "Lorentz Embeddings": lambda d, s, dev: LorentzEmbeddingsEmbedder(
        d=d, random_state=s, device=dev),
}

# The four hyperbolic methods carried through the paper's Table I.
PAPER_METHODS = ["Hydra+", "D-Mercator", "Poincare Embeddings", "Poincare Maps"]


def _unweighted(G: nx.Graph) -> nx.Graph:
    """Topology-only copy of ``G`` (drops edge weights).

    The single-cell k-NN edge weights are *distances*, not affinities, so
    feeding them to methods that read edge weight as connection strength (the
    Poincaré Maps RFA / Laplacian, etc.) inverts the similarity signal. The
    experiments embed the graph topology, as in the original paper.
    """
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from(G.edges())
    return H


def hyperbolic_candidate_scores(G_train, embedder, split):
    """Fit an embedder on the (unweighted) training graph, score by distance."""
    embedder.fit(_unweighted(G_train))
    # Score on the exact representation, not embeddings() ball coordinates: the
    # ball chart saturates past r ≈ 12, so distance ranking on it silently
    # collapses large-radius candidates (leaves sit at large r on real graphs).
    D = pairwise_distance_matrix(embedder.embeddings_representation())
    return candidate_scores(split, D, nodes=embedder.nodes())


def rdpg_candidate_scores(G_train, split, n_components: int):
    """Score candidates by the RDPG connection probability from ASE.

    Node order is read off ``G_train`` rather than assumed to be ``0..N-1``:
    restricting a training graph to its giant connected component leaves the
    labels a subset of ``range(N)``, and indexing by ``range(N)`` there raises
    (or, worse, would score the wrong pairs).
    """
    from graspologic.embed import AdjacencySpectralEmbed

    nodes = list(G_train.nodes())
    A = nx.to_numpy_array(G_train, nodelist=nodes, weight=None)  # binary
    Xhat = AdjacencySpectralEmbed(n_components=n_components).fit_transform(A)
    P = Xhat @ Xhat.T
    return candidate_scores(split, P, nodes=nodes)


def _score_dict(scores, is_positive, higher_is_link):
    f1 = precision_recall_f1_at_k(scores, is_positive, higher_is_link=higher_is_link)
    lift = lift_curve(scores, is_positive, n_bins=10, higher_is_link=higher_is_link)
    captured, total = lift.captured_in_first_bin
    return {
        "f1": f1["f1"],
        "auc": roc_auc(scores, is_positive, higher_is_link=higher_is_link),
        "lift_captured": captured,
        "lift_total": total,
    }


def largest_component_split(
    G_train: nx.Graph, split: LinkPredictionSplit
) -> tuple[nx.Graph, LinkPredictionSplit, dict]:
    """Restrict a training graph and its split to ``G_train``'s giant component.

    Edge removal fragments sparse graphs — measured at ``q=0.9``: 51 components
    on Cora, 71 on OpenFlights, and 109 on the raw WordNet mammal tree, where
    every edge is a bridge. Several embedders also require a connected input
    (``PoincareMaps``' forest matrix goes block-diagonal otherwise). Following
    Jankowski et al. (2026), embedding *and* evaluation are both restricted to
    the giant connected component of the training graph; candidate pairs with an
    endpoint outside it are dropped, since the training graph carries no
    information linking them.

    Node labels are left alone — both scorers take an explicit ``nodes`` order —
    so the returned split's pairs still refer to ``G``'s original labels.

    Returns
    -------
    (G_sub, split_sub, stats)
        The induced subgraph, the filtered split, and retention counts so the
        drop is reported rather than assumed.
    """
    giant = max(nx.connected_components(G_train), key=len)

    def keep(pairs):
        return [(u, v) for u, v in pairs if u in giant and v in giant]

    split_sub = LinkPredictionSplit(
        omega_E=keep(split.omega_E),
        omega_R=keep(split.omega_R),
        omega_N=keep(split.omega_N),
    )
    stats = {
        "n_nodes": len(giant),
        "n_nodes_original": G_train.number_of_nodes(),
        "n_positives": len(split_sub.omega_R),
        "n_positives_original": len(split.omega_R),
        "n_candidates": len(split_sub.candidates),
    }
    return G_train.subgraph(giant).copy(), split_sub, stats


def _prepare_splits(
    G: nx.Graph,
    seeds: list[int],
    q: float,
    restrict_to_giant_component: bool = False,
) -> list[tuple[nx.Graph, LinkPredictionSplit, dict]]:
    """Build one (training graph, split, stats) per seed.

    Prepared once and shared by every arm, so the hyperbolic-minus-Euclidean
    difference at a given seed is a *paired* comparison on identical data
    rather than two independent draws that merely share a seed.
    """
    prepared = []
    for seed in seeds:
        split = link_prediction_split(G, q=q, seed=seed)
        G_train = training_graph(G, split)
        if restrict_to_giant_component:
            prepared.append(largest_component_split(G_train, split))
        else:
            stats = {
                "n_nodes": G_train.number_of_nodes(),
                "n_nodes_original": G_train.number_of_nodes(),
                "n_positives": len(split.omega_R),
                "n_positives_original": len(split.omega_R),
                "n_candidates": len(split.candidates),
            }
            prepared.append((G_train, split, stats))
    return prepared


def evaluate_on_splits(
    prepared: list[tuple[nx.Graph, LinkPredictionSplit, dict]],
    method: str,
    d: int = 2,
    rdpg_dim: Optional[int] = None,
    device: str = "cpu",
) -> dict:
    """Run one method over pre-built splits and aggregate the metrics.

    ``method`` is a key of :data:`HYPERBOLIC_METHODS`, or ``"RDPG"`` (then
    ``rdpg_dim`` selects the embedding dimension, defaulting to ``d``).
    """
    if method == "RDPG" and rdpg_dim is None:
        rdpg_dim = d

    f1s, aucs, captured, totals, times = [], [], [], [], []
    for seed, (G_train, split, _stats) in enumerate(prepared):
        t0 = time.perf_counter()
        if method == "RDPG":
            scores, is_pos = rdpg_candidate_scores(G_train, split, rdpg_dim)
            higher = True  # rank by probability
        else:
            embedder = HYPERBOLIC_METHODS[method](d, seed, device)
            scores, is_pos = hyperbolic_candidate_scores(G_train, embedder, split)
            higher = False  # rank by distance
        times.append(time.perf_counter() - t0)
        res = _score_dict(scores, is_pos, higher)
        f1s.append(100.0 * res["f1"])
        aucs.append(res["auc"])
        captured.append(res["lift_captured"])
        totals.append(res["lift_total"])

    stats = [st for _, _, st in prepared]
    return {
        "method": method if method != "RDPG" else f"RDPG (n={rdpg_dim})",
        "arm": "euclidean" if method == "RDPG" else "hyperbolic",
        "d": rdpg_dim if method == "RDPG" else d,
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "f1_per_seed": f1s,
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "auc_per_seed": aucs,
        "lift_captured_mean": float(np.mean(captured)),
        "lift_total_mean": float(np.mean(totals)),
        "time_mean": float(np.mean(times)),
        "n_nodes_mean": float(np.mean([st["n_nodes"] for st in stats])),
        "n_positives_mean": float(np.mean([st["n_positives"] for st in stats])),
        "n_candidates_mean": float(np.mean([st["n_candidates"] for st in stats])),
    }


def evaluate(
    G: nx.Graph,
    method: str,
    seeds: list[int],
    q: float = 0.9,
    d: int = 2,
    rdpg_dim: Optional[int] = None,
    device: str = "cpu",
    restrict_to_giant_component: bool = False,
) -> dict:
    """Run one method over several seeds and aggregate the metrics.

    Thin wrapper over :func:`_prepare_splits` + :func:`evaluate_on_splits`, kept
    as the single-method entry point. ``restrict_to_giant_component`` defaults
    to ``False`` so Table I's connected single-cell graphs behave exactly as
    before; the hierarchy ladder sets it.

    Returns per-seed values plus ``f1_mean``/``f1_std`` (percentages),
    ``auc_mean``/``auc_std``, and the mean captured/total first-decile lift.
    """
    prepared = _prepare_splits(G, list(seeds), q, restrict_to_giant_component)
    return evaluate_on_splits(prepared, method, d=d, rdpg_dim=rdpg_dim, device=device)


# ----------------------------------------------------------------------
# Full Table I driver
# ----------------------------------------------------------------------

DATASETS = ["ToggleSwitch", "Olsson", "MyeloidProgenitors"]

# Methods carried in the corrected Table I (paper's four + Lorentz + HyperMap).
TABLE_METHODS = [
    "Hydra+",
    "D-Mercator",
    "Poincare Embeddings",
    "Poincare Maps",
    "Lorentz Embeddings",
    "HyperMap",
]


def run_table_i(
    datasets: list[str] = DATASETS,
    methods: list[str] = TABLE_METHODS,
    rdpg_dims: tuple[int, ...] = (2, 8, 16),
    seeds: list[int] = (0, 1, 2, 3, 4),
    q: float = 0.9,
) -> list[dict]:
    """Run the full link-prediction table; returns one result row per model/dataset.

    Each graph is built with its official Poincaré-Maps per-dataset settings
    (:data:`~experiments.datasets.OFFICIAL_SETTINGS`): k and PCA. The v1 paper
    used a uniform k=15, no-PCA recipe (kept as ``table_i_uniform_k15.md``).
    """
    from experiments.datasets import OFFICIAL_SETTINGS, single_cell_graph

    seeds = list(seeds)
    rows = []
    for name in datasets:
        cfg = OFFICIAL_SETTINGS[name]
        G = single_cell_graph(name, k=cfg["k"], n_pca=cfg["n_pca"])
        for n in rdpg_dims:
            row = evaluate(G, "RDPG", seeds, q=q, rdpg_dim=n)
            rows.append({"dataset": name, **row})
        for m in methods:
            rows.append({"dataset": name, **evaluate(G, m, seeds, q=q)})
    return rows


# ----------------------------------------------------------------------
# Hierarchy ladder: does the hyperbolic advantage grow with hierarchy?
# ----------------------------------------------------------------------
# ogbl-ddi showed no hyperbolic advantage (Poincare Hits@20 test 0.2278 vs an
# RDPG/ASE baseline's 0.2406 at d=8). It is also the least tree-like graph in
# experiments/results/geometry_diagnostics.md (delta/diam 0.031, verdict
# "ambiguous"), with a mean degree of 500 against 2-30 everywhere else. This
# ladder tests whether the advantage appears once the graph *is* hierarchical.
#
# The claim under test is ordinal -- Delta on the hierarchical graphs exceeds
# Delta on ddi -- not a threshold, and deliberately not "Delta falls monotonically
# with delta-hyperbolicity": Chami et al.'s (2019) own *shallow* rows are not
# monotone in delta (their biggest Euclidean-to-Poincare gain is on Cora, their
# least hyperbolic dataset), so a clean monotone trend is not predicted.


def _cora_lcc() -> nx.Graph:
    from experiments.datasets import citation_graph, largest_connected_component

    return largest_connected_component(citation_graph("Cora"))


def _ladder_loaders() -> dict:
    """Dataset name -> zero-argument loader (imported lazily: heavy optional deps)."""
    from experiments.datasets import openflights_graph, wordnet_mammal_closure_graph

    return {
        "wordnet/mammal_closure": wordnet_mammal_closure_graph,
        "openflights": openflights_graph,
        "citation/Cora": _cora_lcc,
    }


# Ordered by the repo's own measured delta/diam (geometry_diagnostics.md).
HIERARCHY_LADDER = ["wordnet/mammal_closure", "openflights", "citation/Cora"]

# Provenance is reported alongside delta because delta alone does not separate
# these graphs from ddi: by raw delta_mean, ddi (0.154) sits *below* OpenFlights
# (0.186) -- its diameter is only 5 -- and the mammal closure has diameter 2, so
# its tiny delta is bounded by construction rather than earned by tree-likeness.
HIERARCHY_PROVENANCE = {
    "wordnet/mammal_closure":
        "literal is-a taxonomy; underlying DAG is a tree (diam 16)",
    "openflights": "air-route hub network; Mercator T=0.58 (Jankowski 2026)",
    "citation/Cora": "citation DAG; in both Jankowski 2026 and Chami 2019",
}


def run_hierarchy_ladder(
    datasets: list[str] = HIERARCHY_LADDER,
    methods: list[str] = ("Poincare Embeddings",),
    dims: tuple[int, ...] = (2, 8, 16),
    seeds: list[int] = (0, 1, 2, 3, 4),
    q: float = 0.9,
    device: str = "cpu",
    loaders: Optional[dict] = None,
    report_delta: bool = True,
    delta_samples: int = 20_000,
) -> list[dict]:
    """Run each dataset x dimension with both arms on identical splits.

    For every dimension in ``dims`` each hyperbolic method is run at ``d`` and
    the RDPG baseline at ``n_components=d``, so the arms are capacity-matched.
    Splits are prepared once per (dataset, seed) and shared by every arm, which
    is what makes the per-seed differences paired.

    ``loaders`` overrides :func:`_ladder_loaders` — the injection point that lets
    tests drive the whole driver on a toy graph with no downloads.
    """
    loaders = loaders if loaders is not None else _ladder_loaders()
    seeds = list(seeds)
    rows = []

    for name in datasets:
        G = loaders[name]()
        graph_info = {
            "dataset": name,
            "n": G.number_of_nodes(),
            "m": G.number_of_edges(),
            "provenance": HIERARCHY_PROVENANCE.get(name, ""),
        }
        if report_delta:
            from experiments.graph_stats import mean_hyperbolicity

            delta = mean_hyperbolicity(G, n_samples=delta_samples, seed=0)
            diam = nx.diameter(G)
            graph_info.update(
                delta_mean=float(delta),
                diameter=int(diam),
                delta_normalized=float(delta / diam) if diam else float("nan"),
            )

        # One set of splits per dataset, reused by every (method, d) below.
        prepared = _prepare_splits(G, seeds, q, restrict_to_giant_component=True)

        for d in dims:
            for m in methods:
                rows.append({**graph_info,
                             **evaluate_on_splits(prepared, m, d=d, device=device)})
            rows.append({**graph_info,
                         **evaluate_on_splits(prepared, "RDPG", d=d, device=device)})
    return rows


def hyperbolic_advantage(rows: list[dict]) -> list[dict]:
    """Pair each hyperbolic row with the RDPG row at the same (dataset, d).

    Both arms ran on the same prepared splits, so the per-seed differences are
    paired: ``delta_auc_std`` is the standard deviation of those differences,
    not the difference of the two arms' standard deviations.

    ``n_seeds_hyp_wins`` is the sign-test statistic. It is reported instead of a
    p-value because five paired samples bottom out at a two-sided p of 0.0625,
    so quoting significance from this many seeds would overclaim.
    """
    euclidean = {
        (r["dataset"], r["d"]): r for r in rows if r.get("arm") == "euclidean"
    }
    out = []
    for r in rows:
        if r.get("arm") != "hyperbolic":
            continue
        base = euclidean.get((r["dataset"], r["d"]))
        if base is None:
            continue
        diffs = np.array(r["auc_per_seed"]) - np.array(base["auc_per_seed"])
        out.append({
            "dataset": r["dataset"],
            "method": r["method"],
            "d": r["d"],
            "provenance": r.get("provenance", ""),
            "delta_normalized": r.get("delta_normalized"),
            "auc_hyp": r["auc_mean"],
            "auc_euc": base["auc_mean"],
            "delta_auc_mean": float(diffs.mean()),
            "delta_auc_std": float(diffs.std()),
            "n_seeds_hyp_wins": int((diffs > 0).sum()),
            "n_seeds": int(diffs.size),
        })
    return out


def format_ladder_table(rows: list[dict]) -> str:
    """Render ladder rows as a Markdown table (AUC first: see the caption)."""
    lines = [
        "| Dataset | delta/diam | N_train | d | Arm | ROC AUC "
        "| F1@|omega_R| (%) | Time (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        dn = r.get("delta_normalized")
        dn_s = f"{dn:.4f}" if isinstance(dn, float) else "-"
        lines.append(
            f"| {r['dataset']} | {dn_s} | {r['n_nodes_mean']:.0f} | {r['d']} "
            f"| {r['method']} | {r['auc_mean']:.4f} ± {r['auc_std']:.4f} "
            f"| {r['f1_mean']:.2f} ± {r['f1_std']:.2f} | {r['time_mean']:.1f} |"
        )
    lines.append("")
    lines.append(
        "ROC AUC is the headline metric here. F1 at |omega_R| collapses at this "
        "scale — with ~10^3 positives among ~10^6 candidates it is dominated by "
        "the candidate-set size — and is kept only for continuity with Table I."
    )
    return "\n".join(lines)


def format_delta_table(deltas: list[dict], ddi_note: Optional[str] = None) -> str:
    """Render the headline hyperbolic-minus-Euclidean AUC table."""
    lines = [
        "| Dataset | Provenance | delta/diam | d | AUC hyp | AUC euc "
        "| Delta AUC | seeds hyp wins |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in deltas:
        dn = r.get("delta_normalized")
        dn_s = f"{dn:.4f}" if isinstance(dn, float) else "-"
        lines.append(
            f"| {r['dataset']} | {r['provenance']} | {dn_s} | {r['d']} "
            f"| {r['auc_hyp']:.4f} | {r['auc_euc']:.4f} "
            f"| {r['delta_auc_mean']:+.4f} ± {r['delta_auc_std']:.4f} "
            f"| {r['n_seeds_hyp_wins']}/{r['n_seeds']} |"
        )
    if ddi_note:
        lines.append("")
        lines.append(ddi_note)
    return "\n".join(lines)


def format_table(rows: list[dict]) -> str:
    """Render result rows as a Markdown table."""
    lines = [
        "| Dataset | Model | F1 (%) | Lift (1st decile) | Time (s) |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lift = f"{r['lift_captured_mean']:.0f}/{r['lift_total_mean']:.0f}"
        f1 = f"{r['f1_mean']:.1f} ± {r['f1_std']:.1f}"
        lines.append(
            f"| {r['dataset']} | {r['method']} | {f1} "
            f"| {lift} | {r['time_mean']:.2f} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    import json
    import sys
    import warnings
    from pathlib import Path

    warnings.filterwarnings("ignore")
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)

    # `python -m experiments.link_prediction_experiment ladder` runs the
    # hierarchy ladder; no argument reproduces Table I as before.
    if len(sys.argv) > 1 and sys.argv[1] == "ladder":
        rows = run_hierarchy_ladder()
        deltas = hyperbolic_advantage(rows)
        (out_dir / "hierarchy_ladder.json").write_text(json.dumps(rows, indent=2))
        (out_dir / "hierarchy_advantage.json").write_text(json.dumps(deltas, indent=2))
        ladder_table = format_ladder_table(rows)
        delta_table = format_delta_table(deltas)
        (out_dir / "hierarchy_ladder.md").write_text(ladder_table + "\n")
        (out_dir / "hierarchy_advantage.md").write_text(delta_table + "\n")
        print("=== HIERARCHY LADDER ===")
        print(ladder_table)
        print("\n=== HYPERBOLIC ADVANTAGE (paired Delta AUC) ===")
        print(delta_table)
    else:
        results = run_table_i()
        table = format_table(results)
        (out_dir / "table_i.md").write_text(table + "\n")
        (out_dir / "table_i.json").write_text(json.dumps(results, indent=2))
        print("=== TABLE I ===")
        print(table)
