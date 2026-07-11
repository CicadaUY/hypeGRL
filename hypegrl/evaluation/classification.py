"""Distance-based node classification on a manifold embedding.

Probes how well an embedding organises node labels geometrically: a k-nearest
-neighbour classifier under the embedding's own geodesic distance, with no
learned classifier on top. The distance comes from the fitted
:class:`~hypegrl.representations.Representation`'s exact ``dist()`` — the
chart-agnostic geometry — fed to scikit-learn as a precomputed matrix; the
classifier and metrics are scikit-learn's.

These functions take a ``Representation``, **not** ball coordinates, on purpose:
``embeddings()`` returns Poincaré-ball coordinates, which *saturate* past
``r ≈ 12`` (``tanh(r/2) → 1`` maps every large radius onto the boundary). Distances
recomputed from those coordinates are silently wrong for large-radius embeddings —
losing radial resolution and, far enough out, scrambling the very nearest-neighbour
ordering this classifier depends on — with no error raised. The representation
preserves the exact radius, so pass ``embedder.embeddings_representation()``.
"""
from typing import Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from hypegrl.representations import Representation


def pairwise_distance_matrix(representation: Representation) -> np.ndarray:
    """``(N, N)`` exact geodesic distance matrix of a fitted ``Representation``.

    Parameters
    ----------
    representation:
        A fitted :class:`~hypegrl.representations.Representation` (any chart —
        polar / ball / hyperboloid), typically
        ``embedder.embeddings_representation()``. Its ``dist()`` is the exact
        hyperbolic distance, so — unlike computing on ``embeddings()`` ball
        coordinates — the matrix is correct at all radii.

    Returns
    -------
    np.ndarray
        ``(N, N)`` distances as ``float64``, rows in the representation's node
        order (``embedder.nodes()``).
    """
    return representation.dist().detach().cpu().numpy()


def hyperbolic_knn_classification(
    representation: Representation,
    y,
    k: int = 5,
    test_size: float = 0.2,
    seed: Optional[int] = None,
) -> dict:
    """Stratified KNN node classification under the embedding's geodesic distance.

    Splits nodes into a stratified train/test partition, then classifies each
    test node by majority vote among its ``k`` nearest *training* nodes in the
    embedding, distances measured by the representation's exact ``dist()``. Row
    ``i`` of the representation must correspond to label ``y[i]`` (align with
    ``embedder.nodes()``).

    Parameters
    ----------
    representation:
        A fitted :class:`~hypegrl.representations.Representation`, typically
        ``embedder.embeddings_representation()`` (see :func:`pairwise_distance_matrix`
        for why a representation rather than ball coordinates).
    y:
        Length-``N`` node labels (any hashable type).
    k:
        Number of neighbours. Must not exceed the training-set size.
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
    D = pairwise_distance_matrix(representation)

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
