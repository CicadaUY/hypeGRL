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

from hypegrl.embedders import PoincareEmbeddingsEmbedder
from hypegrl.evaluation import pairwise_distance_matrix


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


def _score_edges(D: np.ndarray, edges) -> np.ndarray:
    """Score = -distance (higher = more likely link), read off the decoder matrix."""
    edges = np.asarray(edges)
    return -D[edges[:, 0], edges[:, 1]]


def evaluate_split(D: np.ndarray, split: dict, evaluator, key: str) -> dict:
    y_pred_pos = _score_edges(D, split[key]["edge"])
    y_pred_neg = _score_edges(D, split[key]["edge_neg"])
    return evaluator.eval({"y_pred_pos": y_pred_pos, "y_pred_neg": y_pred_neg})


def run(
    d: int = 10,
    n_steps: int = 500,
    seed: int = 0,
    root: str = "./data/ogbl-ddi",
) -> dict:
    from ogb.linkproppred import Evaluator

    G, split = load_ddi_split(root=root)
    print(f"ogbl-ddi training graph: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")

    embedder = PoincareEmbeddingsEmbedder(d=d, n_steps=n_steps, random_state=seed)
    t0 = time.perf_counter()
    embedder.fit(G)
    fit_time = time.perf_counter() - t0

    D = pairwise_distance_matrix(embedder.embeddings_representation())
    evaluator = Evaluator(name="ogbl-ddi")

    return {
        "valid": evaluate_split(D, split, evaluator, "valid"),
        "test": evaluate_split(D, split, evaluator, "test"),
        "fit_time_s": fit_time,
        "d": d,
        "n_steps": n_steps,
        "seed": seed,
    }


if __name__ == "__main__":
    import json
    results = run()
    print(json.dumps(results, indent=2))
