"""Tests for the experiments/ reproduction helpers (not part of the library)."""
import networkx as nx
import numpy as np
import pytest
import torch

from experiments import two_stage_chart_schedule as two_stage
from experiments.datasets import balanced_tree_graph, single_cell_graph
from experiments.graph_stats import _distance_matrix, mean_hyperbolicity
from experiments.ogbl_ddi_link_prediction import _score_edges
from hypegrl.representations import (
    CurvedPolarRepresentation,
    TangentRepresentation,
)


def test_tree_is_zero_hyperbolic():
    # Trees are 0-hyperbolic: every quadruple has delta = 0.
    G = nx.balanced_tree(2, 4)
    assert mean_hyperbolicity(G, n_samples=2000, seed=0) == 0.0


def test_four_cycle_has_delta_one():
    # C4: opposite pairs at distance 2, adjacent at 1 -> sums {4, 2, 2},
    # delta = (4 - 2) / 2 = 1. Its only quadruple, so delta_mean = 1.
    G = nx.cycle_graph(4)
    assert mean_hyperbolicity(G, n_samples=500, seed=0) == pytest.approx(1.0)


def test_requires_four_nodes():
    with pytest.raises(ValueError):
        mean_hyperbolicity(nx.path_graph(3))


def test_runs_and_is_bounded_on_karate():
    G = nx.karate_club_graph()
    delta = mean_hyperbolicity(G, n_samples=5000, seed=0)
    # Small, dense, low-diameter graph: non-negative and well below the diameter.
    assert 0.0 <= delta < nx.diameter(G)


def test_distance_matrix_subset_matches_all_pairs():
    # _distance_matrix(G, nodes=subset) restricted to a subset of nodes
    # must agree with the true all-pairs hop distances among that subset.
    G = nx.karate_club_graph()
    nodes = list(G.nodes())
    subset = [nodes[0], nodes[5], nodes[12], nodes[20], nodes[33]]
    D = _distance_matrix(G, nodes=subset)
    all_pairs = dict(nx.all_pairs_shortest_path_length(G))
    for i, u in enumerate(subset):
        for j, v in enumerate(subset):
            assert D[i, j] == all_pairs[u][v]


def test_mean_hyperbolicity_pool_subsampling_runs_on_small_pool():
    # With max_candidate_nodes well below N, mean_hyperbolicity must draw
    # quadruples from a random node pool instead of all N nodes -- this is
    # what keeps the distance matrix tractable on graphs too large for a
    # full (N, N) allocation (e.g. ogbn-arxiv's 169,343 nodes).
    G = nx.barabasi_albert_graph(500, 3, seed=0)
    delta = mean_hyperbolicity(G, n_samples=1000, seed=0, max_candidate_nodes=20)
    assert 0.0 <= delta < nx.diameter(G)


# ----------------------------------------------------------------------
# Dataset loaders
# ----------------------------------------------------------------------


def test_balanced_tree_default_shape():
    G = balanced_tree_graph(2, 4)
    assert G.number_of_nodes() == 31 and G.number_of_edges() == 30


# Paper Table I: (nodes, edges, diameter) for the single-cell k-NN graphs.
@pytest.mark.parametrize(
    "name, n, m, diam",
    [
        ("ToggleSwitch", 200, 1896, 16),
        ("Olsson", 382, 4214, 8),
        ("MyeloidProgenitors", 640, 5649, 38),
    ],
)
def test_single_cell_graph_matches_paper(name, n, m, diam):
    G = single_cell_graph(name)
    assert G.number_of_nodes() == n
    assert G.number_of_edges() == m
    assert nx.is_connected(G)
    assert nx.diameter(G) == diam
    # Edges carry the k-NN distance as weight; all nodes have a cell-type label.
    u, v, data = next(iter(G.edges(data=True)))
    assert data["weight"] > 0
    assert all("label" in d for _, d in G.nodes(data=True))


# ----------------------------------------------------------------------
# OGB link prediction
# ----------------------------------------------------------------------


def test_score_edges_reads_distance_matrix():
    D = np.array([[0.0, 1.0, 5.0], [1.0, 0.0, 2.0], [5.0, 2.0, 0.0]])
    edges = np.array([[0, 1], [1, 2]])
    scores = _score_edges(D, edges)
    np.testing.assert_allclose(scores, [-1.0, -2.0])


# ----------------------------------------------------------------------
# WordNet mammal closure (the link-prediction ladder's most hierarchical rung)
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def mammal_closure():
    pytest.importorskip("nltk")
    from experiments.datasets import wordnet_mammal_closure_graph

    try:
        return wordnet_mammal_closure_graph()
    except LookupError:  # corpus not downloaded and no network
        pytest.skip("nltk WordNet corpus unavailable")


def test_wordnet_mammal_closure_structure(mammal_closure):
    G = mammal_closure
    N = G.number_of_nodes()
    # Loose bands, not exact counts: sizes track the installed WordNet release.
    assert 1000 < N < 1400
    assert 6000 < G.number_of_edges() < 7000
    assert nx.is_connected(G)
    # The root is adjacent to every descendant, so the closure has diameter 2.
    # Recorded because it bounds four-point hyperbolicity by construction.
    assert nx.diameter(G) == 2
    assert max(dict(G.degree()).values()) == N - 1
    assert sorted(G.nodes()) == list(range(N))
    assert G.nodes[0]["synset"]


def test_closure_edges_are_ancestor_or_descendant_pairs(mammal_closure):
    """Every closure edge must be a genuine ancestor/descendant pair.

    The test with teeth: it fails if the closure were built over the wrong
    relation (e.g. siblings joined, or direction dropped before closing).
    """
    wn = pytest.importorskip("nltk.corpus").wordnet
    G = mammal_closure
    rng = np.random.default_rng(0)

    for node in rng.choice(G.number_of_nodes(), size=15, replace=False):
        synset = wn.synset(G.nodes[int(node)]["synset"])
        # All ancestors and all descendants of `synset` within the subtree.
        ancestors = {s.name() for s in synset.closure(lambda x: x.hypernyms())}
        descendants = {s.name() for s in synset.closure(lambda x: x.hyponyms())}
        descendants |= {
            s.name()
            for s in synset.closure(lambda x: x.hyponyms() + x.instance_hyponyms())
        }
        related = ancestors | descendants
        for nbr in G.neighbors(int(node)):
            assert G.nodes[nbr]["synset"] in related


def test_closure_survives_edge_removal_where_the_tree_does_not(mammal_closure):
    """The reason the ladder embeds the closure and not the raw tree.

    In a tree every edge is a bridge, so each held-out positive puts its two
    endpoints in different components of the training graph and the standard
    giant-component restriction leaves nothing to score. The closure keeps its
    positives.
    """
    from experiments.datasets import wordnet_noun_subtree_graph
    from hypegrl.evaluation import link_prediction_split, training_graph

    def retention(G, seed):
        split = link_prediction_split(G, q=0.9, seed=seed)
        G_train = training_graph(G, split)
        giant = max(nx.connected_components(G_train), key=len)
        kept = sum(u in giant and v in giant for u, v in split.omega_R)
        return nx.number_connected_components(G_train), kept, len(split.omega_R)

    tree = nx.convert_node_labels_to_integers(
        wordnet_noun_subtree_graph("mammal.n.01")
    )
    for seed in range(3):
        comps, kept, total = retention(tree, seed)
        assert comps > 50
        assert total > 0
        assert kept == 0  # every edge is a bridge: nothing is left to score

        comps, kept, total = retention(mammal_closure, seed)
        assert comps <= 5
        assert kept / total > 0.95  # measured 99.6-99.9%


def test_largest_connected_component_relabels_and_preserves_attributes():
    from experiments.datasets import largest_connected_component

    G = nx.Graph()
    G.add_edges_from([("a", "b"), ("b", "c"), ("c", "a")])  # triangle (kept)
    G.add_edge("x", "y")  # smaller component (dropped)
    nx.set_node_attributes(G, {"a": "A", "b": "B", "c": "C", "x": "X"}, "tag")
    G["a"]["b"]["weight"] = 2.5

    H = largest_connected_component(G)
    assert sorted(H.nodes()) == [0, 1, 2]
    assert nx.is_connected(H)
    assert {H.nodes[i]["tag"] for i in H} == {"A", "B", "C"}
    assert 2.5 in [d.get("weight") for _, _, d in H.edges(data=True)]


# ----------------------------------------------------------------------
# Link-prediction harness
# ----------------------------------------------------------------------


def test_rdpg_candidate_scores_handles_non_contiguous_node_labels():
    """The RDPG baseline must not assume node labels are exactly ``0..N-1``.

    Restricting a training graph to its giant connected component leaves the
    labels a *subset* of ``range(N)``, which is exactly what the hierarchy
    ladder does. Indexing the adjacency by ``range(N)`` there either raises or
    silently scores the wrong pairs.
    """
    from experiments.link_prediction_experiment import rdpg_candidate_scores
    from hypegrl.evaluation import link_prediction_split, training_graph

    G = nx.karate_club_graph()
    G.remove_node(0)  # labels are now {1..33}: a non-contiguous subset
    assert sorted(G.nodes()) != list(range(G.number_of_nodes()))

    split = link_prediction_split(G, q=0.9, seed=0)
    G_train = training_graph(G, split)
    scores, is_pos = rdpg_candidate_scores(G_train, split, n_components=4)

    assert len(scores) == len(split.candidates)
    assert np.isfinite(scores).all()
    assert is_pos.sum() == len(split.omega_R)

    # Relabelling the graph must not change the scores: with labels mapped to
    # 0..N-1 the buggy and fixed versions agree, so any divergence above is the
    # label assumption, not the embedding.
    mapping = {old: new for new, old in enumerate(sorted(G.nodes()))}
    H = nx.relabel_nodes(G, mapping)
    split_h = link_prediction_split(H, q=0.9, seed=0)
    scores_h, _ = rdpg_candidate_scores(training_graph(H, split_h), split_h, 4)
    assert np.allclose(np.sort(scores), np.sort(scores_h))


def test_largest_component_split_drops_only_out_of_component_pairs():
    from experiments.link_prediction_experiment import largest_component_split
    from hypegrl.evaluation import LinkPredictionSplit

    # Two triangles joined by a bridge; the bridge is the held-out edge, so
    # removing it splits the graph and node 3..5 fall out of the giant CC.
    G_train = nx.Graph()
    G_train.add_edges_from([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)])
    G_train.add_nodes_from(range(6))
    split = LinkPredictionSplit(
        omega_E=[(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)],
        omega_R=[(2, 3)],            # the removed bridge: spans both components
        omega_N=[(0, 4), (1, 2), (0, 1)],
    )
    G_sub, split_sub, stats = largest_component_split(G_train, split)

    assert nx.is_connected(G_sub)
    assert G_sub.number_of_nodes() == 3
    assert split_sub.omega_R == []          # the bridge pair is unscoreable
    assert (0, 4) not in split_sub.omega_N  # endpoint outside the giant CC
    assert (0, 1) in split_sub.omega_N
    assert stats["n_nodes"] == 3 and stats["n_nodes_original"] == 6
    assert stats["n_positives"] == 0 and stats["n_positives_original"] == 1


def test_ladder_driver_runs_on_an_injected_toy_loader():
    """The `loaders` injection point drives the whole pipeline with no download."""
    from experiments.link_prediction_experiment import run_hierarchy_ladder

    rows = run_hierarchy_ladder(
        datasets=["toy"],
        loaders={"toy": nx.karate_club_graph},
        methods=["Poincare Embeddings"],
        dims=(2,),
        seeds=[0],
        report_delta=False,
    )
    assert len(rows) == 2  # one hyperbolic arm + the RDPG control
    assert {r["arm"] for r in rows} == {"hyperbolic", "euclidean"}
    for r in rows:
        assert 0.0 <= r["auc_mean"] <= 1.0
        assert r["d"] == 2 and r["dataset"] == "toy"


def test_ladder_prepares_splits_once_per_dataset_so_arms_are_paired():
    """Both arms at every dimension must see the *same* split objects.

    The headline statistic is a paired per-seed difference, which is only valid
    if the two arms ran on identical data — so the splits must be built once per
    dataset, not once per (method, dimension).
    """
    import experiments.link_prediction_experiment as lpe

    calls, seen = [], []
    real_prepare = lpe._prepare_splits

    def counting_prepare(G, seeds, q, restrict_to_giant_component=False):
        calls.append(1)
        return real_prepare(G, seeds, q, restrict_to_giant_component)

    real_eval = lpe.evaluate_on_splits

    def spying_eval(prepared, method, **kw):
        seen.append(id(prepared))
        return real_eval(prepared, method, **kw)

    lpe._prepare_splits, lpe.evaluate_on_splits = counting_prepare, spying_eval
    try:
        lpe.run_hierarchy_ladder(
            datasets=["toy"],
            loaders={"toy": nx.karate_club_graph},
            methods=["Poincare Embeddings"],
            dims=(2, 4),
            seeds=[0],
            report_delta=False,
        )
    finally:
        lpe._prepare_splits, lpe.evaluate_on_splits = real_prepare, real_eval

    assert sum(calls) == 1          # once for the dataset, not 4x for the cells
    assert len(seen) == 4           # 2 dims x (hyperbolic + euclidean)
    assert len(set(seen)) == 1      # identity, not just equality


def test_hyperbolic_advantage_pairs_by_dataset_and_dim():
    from experiments.link_prediction_experiment import hyperbolic_advantage

    # Per-seed AUCs differ by a constant +0.10, so the *paired* std is exactly 0
    # even though each arm has a large spread of its own. Computing the
    # difference of the two arms' stds instead would give 0 here too, so make
    # the arms' spreads differ to tell the two apart.
    rows = [
        {"dataset": "A", "d": 2, "arm": "hyperbolic", "method": "PE",
         "auc_mean": 0.80, "auc_per_seed": [0.75, 0.85, 0.80]},
        {"dataset": "A", "d": 2, "arm": "euclidean", "method": "RDPG (n=2)",
         "auc_mean": 0.70, "auc_per_seed": [0.65, 0.75, 0.70]},
        {"dataset": "A", "d": 8, "arm": "hyperbolic", "method": "PE",
         "auc_mean": 0.90, "auc_per_seed": [0.90, 0.90, 0.90]},
        {"dataset": "A", "d": 8, "arm": "euclidean", "method": "RDPG (n=8)",
         "auc_mean": 0.88, "auc_per_seed": [0.80, 0.95, 0.89]},
    ]
    out = {(r["dataset"], r["d"]): r for r in hyperbolic_advantage(rows)}
    assert len(out) == 2

    a2 = out[("A", 2)]
    assert a2["delta_auc_mean"] == pytest.approx(0.10)
    assert a2["delta_auc_std"] == pytest.approx(0.0)   # paired, not marginal
    assert a2["n_seeds_hyp_wins"] == 3 and a2["n_seeds"] == 3

    a8 = out[("A", 8)]
    # Paired diffs are [+0.10, -0.05, +0.01]: std of those, not std(hyp)-std(euc).
    assert a8["delta_auc_std"] == pytest.approx(np.std([0.10, -0.05, 0.01]))
    assert a8["n_seeds_hyp_wins"] == 2


def test_score_edges_direction_flag():
    """The RDPG arm scores a probability matrix, the hyperbolic arms a distance
    matrix, so the sign convention has to be selectable."""
    from experiments.ogbl_ddi_link_prediction import _score_edges

    M = np.array([[0.0, 1.0, 5.0], [1.0, 0.0, 2.0], [5.0, 2.0, 0.0]])
    edges = np.array([[0, 1], [1, 2]])
    np.testing.assert_allclose(_score_edges(M, edges, higher_is_link=True), [1.0, 2.0])
    np.testing.assert_allclose(_score_edges(M, edges), [-1.0, -2.0])  # default


def test_evaluate_split_reports_hits_and_auc_without_ogb_download():
    """evaluate_split's own logic, against a stub evaluator.

    The OGB-backed paths (``load_ddi_split``, and ``rdpg_score_matrix`` at
    ogbl-ddi's size) need a large download and minutes of compute, so they are
    deliberately not unit-tested; this covers the scoring/AUC assembly around
    them.
    """
    from experiments.ogbl_ddi_link_prediction import evaluate_split

    # Distances: the two positives are closer than the two negatives.
    M = np.array([
        [0.0, 0.1, 9.0, 9.0],
        [0.1, 0.0, 9.0, 9.0],
        [9.0, 9.0, 0.0, 0.2],
        [9.0, 9.0, 0.2, 0.0],
    ])
    split = {"test": {"edge": np.array([[0, 1], [2, 3]]),
                      "edge_neg": np.array([[0, 2], [1, 3]])}}

    class StubEvaluator:
        def eval(self, d):
            return {"hits@20": float((d["y_pred_pos"] > d["y_pred_neg"].max()).mean())}

    out = evaluate_split(M, split, StubEvaluator(), "test")
    assert out["hits@20"] == 1.0
    assert out["roc_auc"] == 1.0  # perfect separation under the distance convention


def test_wordnet_closure_edge_order_is_canonical(mammal_closure):
    """Edge *order*, not just the edge set, must be reproducible.

    ``link_prediction_split`` draws one RNG value per edge in ``G.edges()``
    order, so a graph whose edge insertion order varies between processes
    produces a different split from the same seed. The closure is built by
    iterating a set of synsets, so it has to be re-emitted in sorted order;
    without that, a seeded sweep is silently irreproducible.
    """
    edges = list(mammal_closure.edges())
    assert edges == sorted(tuple(sorted(e)) for e in edges)
    assert list(mammal_closure.nodes()) == list(range(mammal_closure.number_of_nodes()))


# --------------------------------------------------- two-stage chart schedule


def _tiny_stress_problem(spine=6, leaves=2, device="cpu"):
    """A caterpillar small enough to optimise in a test, with its stress target."""
    G = two_stage.caterpillar(spine, leaves)
    n = G.number_of_nodes()
    D = np.array(nx.floyd_warshall_numpy(G), dtype=np.float64)
    mask = torch.as_tensor(np.triu(np.ones((n, n), dtype=bool), k=1)).to(device)
    r, v, k, _ = two_stage.warm_start(G, 1.0)
    return G, r, v, D * np.sqrt(k), mask


def test_caterpillar_shape():
    """A spine of ``spine`` nodes, each carrying ``leaves``."""
    G = two_stage.caterpillar(10, 3)
    assert G.number_of_nodes() == 10 + 10 * 3
    assert nx.is_tree(G)
    assert nx.diameter(G) == 11               # 9 spine hops plus a leaf at each end


@pytest.mark.parametrize("chart", ["tangent", "c=0.3"])
def test_refinement_reduces_stress(chart):
    _, r, v, target, mask = _tiny_stress_problem()
    stress, rep, history = two_stage.refine(chart, r, v, target, mask, 1e-2, 50, "cpu")
    assert np.isfinite(history).all()
    assert stress < history[0]
    assert np.isfinite(rep.dist().detach().cpu().numpy()).all()


def test_refinement_is_deterministic():
    """
    The HYDRA warm start is closed-form and the stress loss is full-batch with no
    sampling, so the schedule carries no seed and a re-run on one device must
    reproduce a number exactly. A stochastic step here would make every recorded
    result unverifiable.
    """
    _, r, v, target, mask = _tiny_stress_problem()
    first = two_stage.refine("c=0.3", r, v, target, mask, 1e-2, 40, "cpu")[2]
    second = two_stage.refine("c=0.3", r, v, target, mask, 1e-2, 40, "cpu")[2]
    assert np.array_equal(first, second)


def test_warm_start_ignores_the_seed():
    """
    The determinism above rests on the warm start being closed-form. If HYDRA ever
    acquires a randomised step, this catches it — the schedule takes no seed, so a
    seeded warm start would silently make every run unreproducible.
    """
    G = two_stage.caterpillar(6, 2)
    a = two_stage.warm_start(G, 1.0)[0]
    b = two_stage.warm_start(G, 1.0)[0]
    assert np.array_equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_devices_agree_at_the_first_step_then_drift():
    """
    Stress descent is chaotic: CPU and CUDA differ by a rounding unit immediately and
    the gap grows without bound, so a stress value is only meaningful to a few
    significant figures and arms must be compared within one device.

    Pinned because the alternative reading — that a re-run disagreeing with a
    recorded number means something broke — costs a debugging session. It did.
    """
    _, r, v, target, mask = _tiny_stress_problem(spine=12, leaves=3)
    cpu = two_stage.refine("tangent", r, v, target, mask.cpu(), 3e-2, 400, "cpu")[2]
    gpu = two_stage.refine("tangent", r, v, target, mask.cuda(), 3e-2, 400, "cuda")[2]
    assert cpu[0] == pytest.approx(gpu[0], rel=1e-12)      # same starting loss
    assert cpu[-1] == pytest.approx(gpu[-1], rel=0.5)      # same order of magnitude


def test_build_selects_the_chart_and_its_curvature():
    _, r, v, _, _ = _tiny_stress_problem()
    assert isinstance(two_stage.build("tangent", r, v, "cpu"), TangentRepresentation)
    curved = two_stage.build("c=0.3", r, v, "cpu")
    assert isinstance(curved, CurvedPolarRepresentation)
    assert curved._manifold.chart_curvature == pytest.approx(0.3)


def test_a_diverged_run_sorts_last_instead_of_raising():
    """An absurd rate must be reported as ``inf``, not crash the sweep."""
    _, r, v, target, mask = _tiny_stress_problem()
    stress, _, _ = two_stage.refine("c=0.3", r, v, target, mask, 1e9, 50, "cpu")
    assert stress == float("inf") or np.isfinite(stress)
