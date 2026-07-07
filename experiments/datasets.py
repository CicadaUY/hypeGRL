"""Dataset loaders for the paper experiments.

Each loader returns a NetworkX graph with ``0..N-1`` integer nodes (so rows of
an embedding align with node ids). Label-bearing datasets also return a
label array aligned to the node ids.

Datasets:

- ``balanced_tree_graph`` — the Fig. 1 balanced binary tree.
- ``single_cell_graph`` — the Table I gene-regulatory networks (Toggle Switch,
  Olsson, Myeloid Progenitors), built as a symmetric k-NN graph over the
  single-cell expression profiles (the Poincaré Maps construction).
- ``polblogs_graph`` — the Table II political-blogs network (largest component).
- ``airports_graph`` — struc2vec air-traffic networks (USA/Brazil/Europe): heavy-tailed,
  unweighted, anonymized, with an activity-class label.
- ``openflights_graph`` — the OpenFlights world airport route network (D-Mercator's own
  benchmark): heavy-tailed, with IATA / name / country / lat / lon node attributes.
"""
import csv
import os
import urllib.request
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csgraph
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph

# Vendored single-cell expression CSVs (from the Poincaré Maps repository).
SINGLE_CELL_DIR = Path(__file__).resolve().parent / "data" / "single_cell"

# Column names that hold cell-type labels rather than expression features.
_LABEL_COLUMNS = ("labels", "cell_type")

# Official Poincaré-Maps per-dataset settings, transcribed from that repo's
# README replication commands (github.com/facebookresearch/PoincareMaps). ``k``
# and ``n_pca`` shape the shared k-NN graph; ``sigma``/``gamma`` are Poincaré
# Maps' own kernel/decoder bandwidths (used by that method, not the graph). The
# v1 paper instead used a uniform k=15, n_pca=0 recipe for all datasets.
OFFICIAL_SETTINGS = {
    "ToggleSwitch":       {"k": 15, "n_pca": 0,  "sigma": 1.0, "gamma": 2.0},
    "Olsson":             {"k": 15, "n_pca": 20, "sigma": 1.0, "gamma": 2.0},
    "MyeloidProgenitors": {"k": 30, "n_pca": 0,  "sigma": 2.0, "gamma": 2.0},
}


def balanced_tree_graph(branching: int = 2, depth: int = 4) -> nx.Graph:
    """The Fig. 1 balanced tree (defaults: branching 2, depth 4 -> 31 nodes)."""
    return nx.balanced_tree(branching, depth)


def _connect_components(
    knn_matrix: np.ndarray,
    features: np.ndarray,
    component_labels: np.ndarray,
    metric: str,
) -> np.ndarray:
    """Join disconnected components by their closest cross-component pair.

    Repeatedly links component ``0`` to whichever other component holds the
    globally nearest node (by feature-space distance), merging until the graph
    is connected. Mirrors the Poincaré Maps ``connect_knn`` step.
    """
    distances = pairwise_distances(features, metric=metric)
    component_labels = component_labels.copy()
    n_components = len(np.unique(component_labels))
    while n_components > 1:
        idx_cur = np.where(component_labels == 0)[0]
        idx_rest = np.where(component_labels != 0)[0]
        d = distances[idx_cur][:, idx_rest]
        ia, ja = np.where(d == np.min(d))
        node_i, node_j = idx_cur[ia[0]], idx_rest[ja[0]]
        knn_matrix[node_i, node_j] = distances[node_i, node_j]
        knn_matrix[node_j, node_i] = distances[node_j, node_i]
        component_labels[component_labels == component_labels[node_j]] = 0
        n_components -= 1
    return knn_matrix


def single_cell_graph(
    name: str,
    k: int = 15,
    metric: str = "minkowski",
    symmetric: bool = True,
    n_pca: int = 0,
    normalize: bool = False,
    datasets_dir: Optional[os.PathLike] = None,
) -> nx.Graph:
    """Symmetric k-NN graph over a single-cell expression dataset.

    Reads ``{name}.csv`` (features per cell, optional ``labels`` column),
    builds a distance-weighted k-NN graph, symmetrises it, and reconnects any
    components so the result is a single connected weighted graph — the
    construction underlying the Table I link-prediction experiments.

    Parameters
    ----------
    name:
        Dataset stem: ``"ToggleSwitch"``, ``"Olsson"`` or ``"MyeloidProgenitors"``.
    k:
        Neighbours per node.
    metric:
        Distance metric passed to :func:`sklearn.neighbors.kneighbors_graph`.
    symmetric:
        ``True`` symmetrises by the larger directed weight (``max``); ``False``
        by the smaller (``min``).
    n_pca:
        If non-zero, reduce features to this many principal components before the
        k-NN (the official Poincaré-Maps preprocessing; e.g. 20 for Olsson).
        ``0`` (default) keeps the raw features.
    normalize:
        If ``True``, mean-variance normalise each feature before the k-NN (the
        official preprocessing for low-dimensional datasets).
    datasets_dir:
        Directory holding the CSVs (defaults to the vendored :data:`SINGLE_CELL_DIR`).

    Returns
    -------
    nx.Graph
        Weighted graph on ``0..N-1``; a ``label`` node attribute is set when the
        CSV has a ``labels`` column. Edge ``weight`` is the k-NN distance.
    """
    directory = Path(datasets_dir) if datasets_dir is not None else SINGLE_CELL_DIR
    df = pd.read_csv(directory / f"{name}.csv")
    label_col = next((c for c in _LABEL_COLUMNS if c in df.columns), None)
    labels = df[label_col].astype(str).to_numpy() if label_col is not None else None
    feature_df = df.drop(columns=[label_col]) if label_col is not None else df
    features = feature_df.select_dtypes(include="number").to_numpy(dtype=float)

    # Preprocessing (mirrors the official ``prepare_data``): optional
    # mean-variance normalisation, then optional PCA. ``random_state`` is set for
    # reproducibility (the official code leaves PCA's randomized solver unseeded).
    if normalize:
        std = features.std(axis=0)
        std[std == 0] = 1.0
        features = (features - features.mean(axis=0)) / std
    if n_pca:
        nc = min(n_pca, features.shape[1])
        features = PCA(n_components=nc, random_state=0).fit_transform(features)

    K = kneighbors_graph(
        features, k, mode="distance", metric=metric, include_self=False
    ).toarray()
    K = np.maximum(K, K.T) if symmetric else np.minimum(K, K.T)

    n_components, component_labels = csgraph.connected_components(K)
    if n_components > 1:
        K = _connect_components(K, features, component_labels, metric)

    G = nx.Graph()
    for i in range(len(features)):
        G.add_node(i, **({} if labels is None else {"label": labels[i]}))
    rows, cols = np.nonzero(K)
    for i, j in zip(rows, cols):
        if i < j and K[i, j] > 0:
            G.add_edge(int(i), int(j), weight=float(K[i, j]))
    return G


def polblogs_graph(root: Optional[os.PathLike] = None) -> tuple[nx.Graph, np.ndarray]:
    """The political-blogs network (largest connected component).

    Loads PolBlogs via PyTorch Geometric, drops edge direction, keeps the
    largest connected component (as required by the connectivity-sensitive
    embedders), and relabels nodes ``0..N-1``.

    Parameters
    ----------
    root:
        Download/cache directory for the PyG dataset (default ``./data/polblogs``).

    Returns
    -------
    (nx.Graph, np.ndarray)
        The graph and the length-``N`` political-leaning labels aligned to node ids.
    """
    from torch_geometric.datasets import PolBlogs

    data = PolBlogs(root=str(root) if root is not None else "./data/polblogs")[0]
    y_all = data.y.numpy()

    edge_index = data.edge_index.numpy()
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    G.add_edges_from(edge_index.T.tolist())

    kept = sorted(max(nx.connected_components(G), key=len))
    old_to_new = {old: new for new, old in enumerate(kept)}
    G = nx.relabel_nodes(G.subgraph(kept).copy(), old_to_new)
    y = y_all[kept]
    return G, y


# OpenFlights raw data (jpatokal/openflights); downloaded on first use and cached.
_OPENFLIGHTS_URLS = {
    "routes.dat": "https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat",
    "airports.dat": "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat",
}


def openflights_graph(root: Optional[os.PathLike] = None) -> nx.Graph:
    """The OpenFlights world airport route network (largest connected component).

    Nodes are airports, edges are commercial routes (undirected, unweighted). This
    is the airport network used by the D-Mercator paper (Jankowski et al., 2023,
    Table I). The raw ``routes.dat`` and ``airports.dat`` are downloaded from the
    OpenFlights repository on first use and cached under ``root``; the graph is the
    largest connected component with nodes relabelled ``0..N-1``.

    Each node carries the airport metadata as attributes: ``iata`` (3-letter code),
    ``name``, ``country``, ``lat``, ``lon``. So the busiest hubs are nameable, e.g.::

        G = openflights_graph()
        hubs = sorted(G, key=G.degree, reverse=True)[:5]
        [G.nodes[n]["iata"] for n in hubs]          # ['AMS', 'FRA', 'CDG', ...]

    Unlike ``airports_graph`` (anonymized regional struc2vec data), this is the full
    global network with identities. Heavy-tailed with high clustering — the S^1/H^2
    regime for HyperMap / D-Mercator; the paper embeds it in ``d=2`` (dimension D=1).

    Parameters
    ----------
    root:
        Download/cache directory (default ``./data/openflights``).

    Returns
    -------
    nx.Graph
        The airport network with ``iata``/``name``/``country``/``lat``/``lon`` node
        attributes.

    Notes
    -----
    OpenFlights tracks a live data file, so the exact ``N`` depends on the snapshot
    downloaded (~3300 airports at the time of writing). Once cached, subsequent calls
    are stable; delete the cache to refresh.
    """
    directory = Path(root) if root is not None else Path("./data/openflights")
    directory.mkdir(parents=True, exist_ok=True)
    for fname, url in _OPENFLIGHTS_URLS.items():
        fpath = directory / fname
        if not fpath.exists():
            urllib.request.urlretrieve(url, fpath)

    # airport id -> (iata, name, country, lat, lon)
    meta: dict[str, tuple] = {}
    with open(directory / "airports.dat", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 8:
                try:
                    lat, lon = float(row[6]), float(row[7])
                except ValueError:
                    lat, lon = float("nan"), float("nan")
                meta[row[0]] = (row[4], row[1], row[3], lat, lon)

    # edges from routes.dat by source/destination airport ID (cols 3 and 5)
    H = nx.Graph()
    with open(directory / "routes.dat", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 6:
                continue
            src, dst = row[3], row[5]        # source / destination airport IDs
            if src != "\\N" and dst != "\\N" and src != dst:
                H.add_edge(src, dst)

    H = H.subgraph(max(nx.connected_components(H), key=len)).copy()
    order = sorted(H.nodes())
    G = nx.relabel_nodes(H, {aid: i for i, aid in enumerate(order)})
    for i, aid in enumerate(order):
        iata, name, country, lat, lon = meta.get(
            aid, ("", "", "", float("nan"), float("nan")))
        G.nodes[i].update(iata=iata, name=name, country=country, lat=lat, lon=lon)
    return G


def airports_graph(
    name: str = "USA",
    root: Optional[os.PathLike] = None,
) -> tuple[nx.Graph, np.ndarray]:
    """Air-traffic network (Ribeiro et al., 2017), largest connected component.

    Nodes are airports, edges commercial flights; ``name`` selects the region
    (``"USA"``, ``"Brazil"``, ``"Europe"``). Node labels ``y`` are the activity
    classes (quartiles of passenger flow). Loaded via PyTorch Geometric;
    self-loops removed, largest component kept, nodes relabelled ``0..N-1``.

    Unlike the single-cell k-NN graphs, these are *observed* networks with a
    genuine heavy-tailed degree distribution and high clustering — the regime the
    S^1/H^2 methods (HyperMap, D-Mercator) are designed for. Their edges are also
    real (not k-NN constructed), so link prediction needs no weighting choice.
    """
    from torch_geometric.datasets import Airports

    data = Airports(
        root=str(root) if root is not None else f"./data/airports_{name.lower()}",
        name=name,
    )[0]
    y_all = data.y.numpy()

    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    G.add_edges_from(data.edge_index.numpy().T.tolist())
    G.remove_edges_from(nx.selfloop_edges(G))

    kept = sorted(max(nx.connected_components(G), key=len))
    old_to_new = {old: new for new, old in enumerate(kept)}
    G = nx.relabel_nodes(G.subgraph(kept).copy(), old_to_new)
    y = y_all[kept]
    return G, y
