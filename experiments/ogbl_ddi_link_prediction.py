"""Link prediction on OGB's ogbl-ddi (drug-drug interaction) dataset.

ogbl-ddi is chosen over the other OGB datasets already wired into
geometry_diagnostics_report.py: it's the only one with an official
link-prediction edge split + Hits@20 Evaluator (ogbn-arxiv is node
classification only), and at 4,267 nodes it fits hypeGRL's O(N^2)
dense-distance-matrix embedders where ogbl-collab's ~235k / ogbn-arxiv's
169k nodes would not.

The embedding method is PoincareEmbeddingsEmbedder with the default
``loss="ranking"`` -- the original Nickel & Kiela (2017) link-prediction
objective (soft-ranking NLL with negative sampling), which is literally
this task rather than an adapted one.
"""
import time

import networkx as nx
import numpy as np
import torch

from hypegrl.embedders import LorentzEmbeddingsEmbedder, PoincareEmbeddingsEmbedder
from hypegrl.evaluation import pairwise_distance_matrix, roc_auc

_EMBEDDERS = {
    "poincare": PoincareEmbeddingsEmbedder,
    "lorentz": LorentzEmbeddingsEmbedder,
}


def _ogb_safe_globals():
    """Classes that need allow-listing to unpickle OGB-cached ``torch_geometric``
    objects under PyTorch 2.6+'s ``weights_only=True`` default ``torch.load``.

    Duplicated from (not imported from) ``geometry_diagnostics_report.py``: that
    module's own top-level import (``from geometry_diagnostics import ...``) is
    module-relative and only resolves when ``experiments/`` is on
    ``sys.path[0]`` (i.e. run as a script), so importing it as
    ``experiments.geometry_diagnostics_report`` fails.
    """
    import numpy as np
    from torch_geometric.data import Data
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage

    return [
        Data, DataEdgeAttr, DataTensorAttr, GlobalStorage,
        np.core.multiarray._reconstruct, np.ndarray, np.dtype,
        np.dtypes.Int64DType,
    ]


def load_ddi_split(root: str = "./data/ogbl-ddi"):
    """Load ogbl-ddi: the training graph plus the official valid/test edge dict."""
    from ogb.linkproppred import PygLinkPropPredDataset

    with torch.serialization.safe_globals(_ogb_safe_globals()):
        dataset = PygLinkPropPredDataset(name="ogbl-ddi", root=root)
        split = dataset.get_edge_split()
        num_nodes = dataset[0].num_nodes

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(split["train"]["edge"].tolist())
    return G, split


def _score_edges(M: np.ndarray, edges, higher_is_link: bool = False) -> np.ndarray:
    """Read edge scores off a decoder matrix, higher = more likely link.

    ``higher_is_link=False`` (the default) negates, for the hyperbolic arms
    whose ``M`` is a *distance* matrix; the RDPG baseline passes ``True``
    because its ``M`` is already a connection probability.
    """
    edges = np.asarray(edges)
    scores = M[edges[:, 0], edges[:, 1]]
    return scores if higher_is_link else -scores


def evaluate_split(
    M: np.ndarray, split: dict, evaluator, key: str, higher_is_link: bool = False
) -> dict:
    """Hits@20 (OGB's own Evaluator) plus ROC AUC on the same scores.

    ROC AUC is the bridge metric to the hierarchy ladder in
    ``link_prediction_experiment.py``. Its *absolute* value is not comparable to
    the ladder's: OGB supplies a fixed, *sampled* negative set (~10^5 pairs),
    where the ladder scores every non-edge (~10^6), and discriminating sampled
    negatives is a much easier problem. Only the within-dataset
    hyperbolic-minus-Euclidean difference is comparable across the two.
    """
    y_pred_pos = _score_edges(M, split[key]["edge"], higher_is_link)
    y_pred_neg = _score_edges(M, split[key]["edge_neg"], higher_is_link)
    result = dict(evaluator.eval({"y_pred_pos": y_pred_pos, "y_pred_neg": y_pred_neg}))
    scores = np.concatenate([y_pred_pos, y_pred_neg])
    is_pos = np.concatenate([
        np.ones(len(y_pred_pos), dtype=bool), np.zeros(len(y_pred_neg), dtype=bool)])
    result["roc_auc"] = roc_auc(scores, is_pos, higher_is_link=True)
    return result


def rdpg_score_matrix(G: nx.Graph, n_components: int) -> np.ndarray:
    """RDPG connection probabilities from an adjacency spectral embedding.

    The Euclidean control arm. This is
    ``link_prediction_experiment.rdpg_candidate_scores``' decoder specialised to
    a precomputed-matrix interface, since ogbl-ddi is scored from OGB's own edge
    arrays rather than through a ``LinkPredictionSplit``.
    """
    from graspologic.embed import AdjacencySpectralEmbed

    nodes = list(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodes, weight=None)
    Xhat = AdjacencySpectralEmbed(n_components=n_components).fit_transform(A)
    return Xhat @ Xhat.T


def run(
    d: int = 10,
    n_steps: int = 1000,
    seed: int = 0,
    root: str = "./data/ogbl-ddi",
    device: str = "cuda",
    method: str = "poincare",
    **embedder_kwargs,
) -> dict:
    """Fit ``method`` and evaluate Hits@20 and ROC AUC.

    ``method`` is ``"poincare"``, ``"lorentz"``, or ``"rdpg"``. The two
    hyperbolic embedders share the ranking (Nickel & Kiela) objective -- Lorentz
    is the same loss on the hyperboloid chart rather than the ball, so any gap
    between them is attributable to the chart's optimisation numerics, not the
    objective. ``"rdpg"`` is the Euclidean control: an adjacency spectral
    embedding decoded by dot product, matching the baseline arm of the hierarchy
    ladder in ``link_prediction_experiment.py`` so ogbl-ddi can be read as that
    ladder's anti-hierarchical anchor.

    Extra ``embedder_kwargs`` (e.g. ``lr_X``) are passed through so each method
    can use its own tuned defaults (Lorentz: lr_X=0.3; Poincare: lr_X=1e-2 --
    see CLAUDE.md) unless explicitly overridden; ``"rdpg"`` takes none.
    """
    from ogb.linkproppred import Evaluator

    if method not in _EMBEDDERS and method != "rdpg":
        raise ValueError(
            f"method must be one of {sorted(_EMBEDDERS) + ['rdpg']}; got {method!r}.")

    G, split = load_ddi_split(root=root)
    print(f"ogbl-ddi training graph: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")

    t0 = time.perf_counter()
    if method == "rdpg":
        M, higher_is_link = rdpg_score_matrix(G, n_components=d), True
    else:
        embedder = _EMBEDDERS[method](
            d=d, n_steps=n_steps, random_state=seed, device=device, **embedder_kwargs
        )
        embedder.fit(G)
        M = pairwise_distance_matrix(embedder.embeddings_representation())
        higher_is_link = False
    fit_time = time.perf_counter() - t0

    evaluator = Evaluator(name="ogbl-ddi")

    return {
        "method": method,
        "valid": evaluate_split(M, split, evaluator, "valid", higher_is_link),
        "test": evaluate_split(M, split, evaluator, "test", higher_is_link),
        "fit_time_s": fit_time,
        "d": d,
        "n_steps": n_steps if method != "rdpg" else None,
        "seed": seed,
    }


if __name__ == "__main__":
    import json
    results = run()
    print(json.dumps(results, indent=2))
