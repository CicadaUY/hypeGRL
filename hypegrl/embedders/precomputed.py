"""Precomputed-embedding wrapper: turn a raw coordinate matrix into an embedder."""
from typing import Optional

import networkx as nx
import numpy as np

from hypegrl.embedders.base import HyperbolicEmbedder


class PrecomputedEmbedder(HyperbolicEmbedder):
    """
    Wraps a precomputed ``(N, d)`` Poincaré-ball coordinate matrix so it can
    be used anywhere the library expects a fitted
    :class:`~hypegrl.embedders.base.HyperbolicEmbedder` — most directly, as
    the ``embedder`` passed to a
    :class:`~hypegrl.generation.base.GraphGenerator` such as
    :class:`~hypegrl.generation.fermi_dirac.FermiDiracGenerator`.

    This is the bridge for embeddings that were **not** produced by fitting
    one of hypeGRL's own methods — points sampled directly on the manifold
    (e.g. uniformly at random on the Poincaré disk), hand-crafted
    coordinates, or embeddings imported from another library.

    Parameters
    ----------
    X:
        ``(N, d)`` array of Poincaré-ball coordinates. Every row must lie
        strictly inside the unit ball (``‖x_i‖ < 1``).
    nodes:
        Node labels in the row order of ``X``. Defaults to
        ``list(range(X.shape[0]))``.

    Notes
    -----
    - :meth:`fit` is a no-op that returns ``self`` — the embedding is
      already "fit" by construction; ``G`` is accepted (and ignored) only
      so the call signature matches the ABC.
    - :meth:`structural_similarity` and :meth:`decode` raise
      ``NotImplementedError``: a raw coordinate matrix carries no
      encoder/decoder pair, so this class is a pure *geometry source*, not
      a trainable method. :meth:`is_generative` correspondingly stays at
      the base-class default of ``False``, so a
      :class:`~hypegrl.generation.fermi_dirac.FermiDiracGenerator` wrapping
      it always falls back to its generic global ``r``/``t`` model rather
      than attempting to call the nonexistent ``decode()``.
    - :meth:`embeddings_representation` stays at the base-class default of
      ``None`` (no ``Representation`` was built), so downstream distance
      computations use the Poincaré-ball manifold on :meth:`embeddings`
      directly — exact for ``‖x‖ ≲ tanh(6) ≈ 1 − 5e-6`` (radius ≲ 12) and
      increasingly lossy closer to the boundary. If you need exactness at
      larger radii, sample in another chart and convert with
      :mod:`hypegrl.manifolds.poincare` / :mod:`hypegrl.manifolds.lorentz`
      before wrapping.

    Examples
    --------
    Sample points uniformly at random on the Poincaré disk and turn them
    into a graph via :class:`~hypegrl.generation.fermi_dirac.FermiDiracGenerator`:

    >>> import numpy as np
    >>> from hypegrl.embedders.precomputed import PrecomputedEmbedder
    >>> from hypegrl.generation.fermi_dirac import FermiDiracGenerator
    >>> rng = np.random.default_rng(0)
    >>> n = 100
    >>> angle = rng.uniform(0, 2 * np.pi, n)
    >>> radius = np.sqrt(rng.uniform(0, 1, n)) * 0.9  # uniform in area, r<0.9
    >>> X = np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)
    >>> emb = PrecomputedEmbedder(X)
    >>> gen = FermiDiracGenerator(emb, r=1.0, t=0.2, random_state=0)
    >>> G = gen.sample(n_graphs=1)[0]
    >>> G.number_of_nodes()
    100
    """

    def __init__(self, X, nodes: Optional[list] = None):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D (N, d) array; got shape {X.shape}.")
        norms = np.linalg.norm(X, axis=1)
        if np.any(norms >= 1.0):
            raise ValueError(
                "X must lie strictly inside the Poincaré ball (every row "
                f"must have ‖x_i‖ < 1); got a max norm of {norms.max():.6f}."
            )
        if nodes is not None and len(nodes) != X.shape[0]:
            raise ValueError(
                f"len(nodes)={len(nodes)} does not match X.shape[0]={X.shape[0]}."
            )
        self._X = X
        self._nodes = list(nodes) if nodes is not None else list(range(X.shape[0]))

    def fit(
        self,
        G: Optional[nx.Graph] = None,
        unknown_edges: Optional[list] = None,
        X_init=None,
    ) -> "PrecomputedEmbedder":
        """No-op: the embedding was supplied at construction. Returns ``self``."""
        return self

    def embeddings(self) -> np.ndarray:
        """Return the ``(N, d)`` Poincaré-ball coordinates given at construction."""
        return self._X

    def structural_similarity(self, G: nx.Graph) -> np.ndarray:
        raise NotImplementedError(
            "PrecomputedEmbedder wraps a raw coordinate matrix and has no "
            "encoder target (s(A)) — it is a geometry source, not a "
            "trainable method."
        )

    def decode(self, X) -> np.ndarray:
        raise NotImplementedError(
            "PrecomputedEmbedder has no decoder. To generate a graph from "
            "its geometry, wrap it in a FermiDiracGenerator (or another "
            "GraphGenerator) rather than calling decode() directly."
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(N={self._X.shape[0]}, d={self._X.shape[1]})"
