"""Distance-based node classification on a manifold embedding.

Probes how well an embedding organises node labels geometrically: a k-nearest
-neighbour classifier under the embedding's own geodesic distance, with no
learned classifier on top. The manifold distance is computed once via
``manifold.dist`` (the library idiom) and fed to scikit-learn as a precomputed
matrix; the classifier and metrics are scikit-learn's.
"""
from typing import Optional

import geoopt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from hypegrl.manifolds.poincare import POINCARE_BALL


def pairwise_distance_matrix(
    X: np.ndarray,
    manifold: geoopt.Manifold = POINCARE_BALL,
) -> np.ndarray:
    """``(N, N)`` geodesic distance matrix of embeddings ``X`` on ``manifold``.

    Parameters
    ----------
    X:
        ``(N, d)`` embedding coordinates in the manifold's chart (Poincaré ball
        coordinates for the default manifold — what every embedder's
        :meth:`~hypegrl.embedders.base.HyperbolicEmbedder.embeddings` returns).
    manifold:
        Manifold the coordinates live on (defaults to the unit-curvature
        Poincaré ball).

    Returns
    -------
    np.ndarray
        ``(N, N)`` distances as ``float64``.
    """
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float64)
    D = manifold.dist(Xt.unsqueeze(1), Xt.unsqueeze(0))
    return D.detach().numpy()


def hyperbolic_knn_classification(
    X: np.ndarray,
    y,
    k: int = 5,
    manifold: geoopt.Manifold = POINCARE_BALL,
    test_size: float = 0.2,
    seed: Optional[int] = None,
) -> dict:
    """Stratified KNN node classification under the embedding's geodesic distance.

    Splits nodes into a stratified train/test partition, then classifies each
    test node by majority vote among its ``k`` nearest *training* nodes in the
    embedding, distances measured by ``manifold``. Row ``i`` of ``X`` must
    correspond to label ``y[i]`` (align with ``embedder.nodes()``).

    Parameters
    ----------
    X:
        ``(N, d)`` embedding coordinates (see :func:`pairwise_distance_matrix`).
    y:
        Length-``N`` node labels (any hashable type).
    k:
        Number of neighbours. Must not exceed the training-set size.
    manifold:
        Manifold the embedding lives on (defaults to the Poincaré ball).
    test_size:
        Fraction of nodes held out for testing.
    seed:
        Seed for the stratified split; ``None`` is nondeterministic.

    Returns
    -------
    dict
        ``accuracy``, ``f1`` (weighted, ``zero_division=0``), ``k``,
        ``n_train``, ``n_test``, ``n_classes``.
    """
    y = np.asarray(y)
    D = pairwise_distance_matrix(X, manifold)

    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=y
    )

    clf = KNeighborsClassifier(n_neighbors=k, metric="precomputed")
    clf.fit(D[np.ix_(train_idx, train_idx)], y[train_idx])
    y_pred = clf.predict(D[np.ix_(test_idx, train_idx)])

    return {
        "accuracy": float(accuracy_score(y[test_idx], y_pred)),
        "f1": float(f1_score(y[test_idx], y_pred, average="weighted", zero_division=0)),
        "k": k,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "n_classes": len(np.unique(y)),
    }
