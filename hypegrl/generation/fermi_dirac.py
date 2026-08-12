"""Fermi-Dirac generator for hyperbolic embedding methods."""
import warnings
from typing import Optional

import networkx as nx
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from hypegrl.embedders.precomputed import PrecomputedEmbedder
from hypegrl.generation.base import GraphGenerator
from hypegrl.representations import PolarRepresentation


class FermiDiracGenerator(GraphGenerator):
    r"""
    Samples graphs under the Fermi-Dirac connection-probability model

    .. math::

        P_{ij} = \frac{1}{\exp\big((d_{ij} - r)/t\big) + 1}

    where :math:`d_{ij}` is the hyperbolic distance between nodes ``i`` and
    ``j`` under the embedder's fitted embedding, ``r`` is the connection
    radius and ``t > 0`` the temperature controlling how sharply the
    probability drops off past ``r``.

    This is the same decoder family used internally by
    :class:`~hypegrl.embedders.hypermap.HyperMapEmbedder`,
    :class:`~hypegrl.embedders.dmercator.DMercatorEmbedder`, and
    :class:`~hypegrl.embedders.poincare_embeddings.PoincareEmbeddingsEmbedder`
    (``loss="fermi_dirac"``) — but exposed here as a **generic, decoupled
    generator** that works with *any* fitted
    :class:`~hypegrl.embedders.base.HyperbolicEmbedder`, including methods
    that have no native generative direction (Poincaré Maps, Hydra, Hydra+,
    Lorentz Embeddings — all ``is_generative() == False``). Wrapping such an
    embedder in :class:`FermiDiracGenerator` turns its geometry into a
    generative model post hoc.

    Two modes, chosen automatically:

    - **Reuse the embedder's own decoder** (default, when ``r`` and ``t``
      are both left ``None``): if ``embedder.is_generative()`` is ``True``,
      :meth:`sample` calls ``embedder.decode(X)`` directly. This picks up
      whatever Fermi-Dirac-family decoder the embedder already carries —
      including HyperMap's *per-node* threshold :math:`R_i` (finer than the
      single global ``r`` used here) and D-Mercator's ``β``/``R̂`` — rather
      than approximating it with a single global ``r``, ``t``.
    - **Generic global model** (whenever ``r`` or ``t`` is supplied, or the
      embedder is not natively generative): pairwise hyperbolic distances
      are read off the embedding through a
      :class:`~hypegrl.representations.Representation` — the embedder's own
      when it has one, else the polar chart built from its ball coordinates,
      so the distances are exact at every radius either way — and combined
      with a single global ``r``, ``t`` in the formula above. Call
      :meth:`fit` to estimate ``r``, ``t`` by maximum likelihood against an
      observed graph, or set them directly at construction.

    Parameters
    ----------
    embedder:
        A fitted :class:`~hypegrl.embedders.base.HyperbolicEmbedder` (or any
        object exposing the same ``nodes()`` / ``embeddings()`` /
        ``embeddings_representation()`` / ``is_generative()`` /
        ``decode()`` interface). To generate from an embedding that was
        never fit by a hypeGRL method at all — e.g. points sampled directly
        on the Poincaré disk — wrap the raw coordinates in
        :class:`~hypegrl.embedders.precomputed.PrecomputedEmbedder`, or use
        the :meth:`from_embeddings` shortcut below.
    r:
        Fermi-Dirac connection radius. ``None`` (default) defers to the
        embedder's own decoder when it is natively generative, else to a
        value set by :meth:`fit`, else the median pairwise distance (with a
        warning).
    t:
        Fermi-Dirac temperature, ``t > 0``. Same fallback order as ``r``,
        defaulting to ``1.0`` in the uncalibrated case.
    random_state:
        Seed for reproducible sampling.

    Examples
    --------
    Reusing a natively generative embedder's own decoder:

    >>> import networkx as nx
    >>> from hypegrl.embedders.hypermap import HyperMapEmbedder
    >>> from hypegrl.generation.fermi_dirac import FermiDiracGenerator
    >>> G = nx.karate_club_graph()
    >>> emb = HyperMapEmbedder(d=2, n_steps=0).fit(G)
    >>> gen = FermiDiracGenerator(emb)          # reuses emb.decode()
    >>> samples = gen.sample(n_graphs=3)
    >>> len(samples)
    3

    Turning a non-generative embedder into a generative one:

    >>> from hypegrl.embedders.poincare_maps import PoincareMapsEmbedder
    >>> emb = PoincareMapsEmbedder(d=2, n_steps=200).fit(G)
    >>> gen = FermiDiracGenerator(emb).fit(G)   # MLE-calibrate r, t against G
    >>> samples = gen.sample(n_graphs=3)

    Generating from a raw coordinate matrix that was never fit by a
    hypeGRL embedder at all — e.g. points sampled uniformly on the
    Poincaré disk — via :meth:`from_embeddings`:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> angle = rng.uniform(0, 2 * np.pi, n)
    >>> radius = np.sqrt(rng.uniform(0, 1, n)) * 0.9
    >>> X = np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)
    >>> gen = FermiDiracGenerator.from_embeddings(X, r=1.0, t=0.2, random_state=0)
    >>> G_sampled = gen.sample(n_graphs=1)[0]

    See Also
    --------
    hypegrl.embedders.precomputed.PrecomputedEmbedder :
        The adapter :meth:`from_embeddings` builds under the hood. Use it
        directly if you want the wrapped object itself (e.g. to pass to
        other tools that expect a
        :class:`~hypegrl.embedders.base.HyperbolicEmbedder`).
    """

    def __init__(
        self,
        embedder,
        r: Optional[float] = None,
        t: Optional[float] = None,
        random_state: Optional[int] = None,
    ):
        super().__init__(embedder)
        self.r = r
        self.t = t
        self.random_state = random_state

    @classmethod
    def from_embeddings(
        cls,
        X,
        nodes: Optional[list] = None,
        r: Optional[float] = None,
        t: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> "FermiDiracGenerator":
        """
        Build a generator directly from a raw ``(N, d)`` Poincaré-ball
        coordinate matrix, without needing a fitted
        :class:`~hypegrl.embedders.base.HyperbolicEmbedder` — e.g. points
        sampled uniformly at random on the Poincaré disk, hand-crafted
        coordinates, or an embedding produced outside hypeGRL.

        Equivalent to::

            FermiDiracGenerator(
                PrecomputedEmbedder(X, nodes=nodes), r=r, t=t,
                random_state=random_state,
            )

        Since :class:`~hypegrl.embedders.precomputed.PrecomputedEmbedder`
        is never natively generative, the returned generator always uses
        the generic global ``r``/``t`` model (see the class docstring) —
        pass ``r``/``t`` explicitly, or call :meth:`fit` against an
        observed graph afterwards to calibrate them by maximum likelihood.

        Parameters
        ----------
        X:
            ``(N, d)`` array of Poincaré-ball coordinates (every row must
            satisfy ``‖x_i‖ < 1``).
        nodes:
            Node labels in the row order of ``X``. Defaults to
            ``range(X.shape[0])``.
        r, t, random_state:
            Forwarded to the constructor.

        Returns
        -------
        FermiDiracGenerator
        """
        return cls(
            PrecomputedEmbedder(X, nodes=nodes),
            r=r,
            t=t,
            random_state=random_state,
        )

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def fit(self, G: nx.Graph) -> "FermiDiracGenerator":
        """
        Estimate the global ``r``, ``t`` by maximum likelihood against an
        observed graph, under the embedder's current (already fitted)
        embedding.

        Fits a two-parameter Bernoulli MLE of the edge indicator on the
        pairwise hyperbolic distances — the joint generalisation of
        :meth:`~hypegrl.embedders.hypermap.HyperMapEmbedder.estimate_temperature`'s
        one-parameter slope fit, here also solving for the radius ``r``
        rather than taking it as given.

        Only meaningful for the generic global model (see the class
        docstring): if the embedder is natively generative and ``r``/``t``
        are left ``None``, :meth:`sample` bypasses ``r``/``t`` entirely and
        this calibration has no effect.

        Parameters
        ----------
        G:
            Observed graph. Only pairs of nodes that both appear in
            ``embedder.nodes()`` contribute to the fit.

        Returns
        -------
        self, with ``self.r`` and ``self.t`` set to the MLE estimates.
        """
        D = self._distance_matrix()
        A = self._adjacency_in_embedder_order(G)

        iu, ju = np.triu_indices(D.shape[0], k=1)
        d = D[iu, ju]
        a = A[iu, ju]

        def neg_log_likelihood(params: np.ndarray) -> float:
            r, log_t = params
            t = np.exp(log_t)
            z = (r - d) / t
            log_p = -np.logaddexp(0.0, -z)      # log sigmoid(z)
            log_1m_p = -np.logaddexp(0.0, z)    # log sigmoid(-z) = log(1-p)
            return float(-(a * log_p + (1.0 - a) * log_1m_p).sum())

        r0 = float(np.median(d))
        result = minimize(neg_log_likelihood, x0=[r0, 0.0], method="Nelder-Mead")
        self.r = float(result.x[0])
        self.t = float(np.exp(result.x[1]))
        return self

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def probabilities(self) -> np.ndarray:
        """
        Return the ``(N, N)`` matrix of Fermi-Dirac connection probabilities
        (rows/columns in ``embedder.nodes()`` order) without sampling.

        Useful for inspection, link-prediction scoring, or as the basis for
        a custom sampling scheme.

        Returns
        -------
        ``(N, N)`` NumPy array with values in ``[0, 1]`` and a zero
        diagonal (no self-loops).
        """
        embedder = self.embedder
        if self.r is None and self.t is None and embedder.is_generative():
            # Every generative embedder is gradient-based and so carries a
            # fitted Representation; decoding it gives the exact geometry.
            P = np.asarray(
                embedder.decode(embedder.embeddings_representation()),
                dtype=np.float64,
            )
        else:
            if self.r is None or self.t is None:
                warnings.warn(
                    "FermiDiracGenerator: r and/or t were not set and the "
                    "embedder is not natively generative (or an override "
                    "was only partially given); defaulting the missing "
                    "value(s) to r=median pairwise distance, t=1.0. Call "
                    ".fit(G) to calibrate r, t against an observed graph, "
                    "or set them explicitly at construction.",
                    stacklevel=2,
                )
            D = self._distance_matrix()
            r = self.r if self.r is not None else float(
                np.median(D[np.triu_indices(D.shape[0], k=1)])
            )
            t = self.t if self.t is not None else 1.0
            # expit, not 1/(exp(.)+1): the exponential overflows for a sharp
            # cutoff (small t) or a large distance, both ordinary here.
            P = expit((r - D) / t)

        P = np.clip(P, 0.0, 1.0)
        np.fill_diagonal(P, 0.0)
        return P

    def sample(self, n_graphs: int = 1) -> list[nx.Graph]:
        """
        Sample ``n_graphs`` graphs by independently drawing each pair
        ``(i, j)`` as an edge with probability :meth:`probabilities` ``[i, j]``.

        Parameters
        ----------
        n_graphs:
            Number of independent graphs to sample.

        Returns
        -------
        List of NetworkX graphs, each containing every node from
        ``embedder.nodes()`` (isolated nodes are included, not dropped).
        """
        nodes = self.embedder.nodes()
        if nodes is None:
            raise RuntimeError(
                "Call embedder.fit() before FermiDiracGenerator.sample()."
            )
        P = self.probabilities()

        N = P.shape[0]
        iu, ju = np.triu_indices(N, k=1)
        probs = P[iu, ju]

        rng = np.random.default_rng(self.random_state)
        graphs = []
        for _ in range(n_graphs):
            keep = rng.random(probs.shape[0]) < probs
            sampled = nx.Graph()
            sampled.add_nodes_from(nodes)
            sampled.add_edges_from(
                (nodes[i], nodes[j])
                for i, j, k in zip(iu, ju, keep)
                if k
            )
            graphs.append(sampled)
        return graphs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _distance_matrix(self) -> np.ndarray:
        """
        ``(N, N)`` exact pairwise hyperbolic distance under the embedder's
        current fit, always measured through a ``Representation``.

        Embedders that build one (every gradient method) supply it directly.
        Those that do not — :class:`~hypegrl.embedders.precomputed.PrecomputedEmbedder`,
        :class:`~hypegrl.embedders.hydra.HydraEmbedder` — expose only
        ``embeddings()``, whose Poincaré-ball coordinates are re-charted into
        the polar representation here. Measuring in the ball instead would cap
        every distance at ``2·artanh(1 − 1e-7) = 16.811243``, silently
        flattening the far end of the probability matrix for a large-radius
        embedding.
        """
        embedder = self.embedder
        rep = embedder.embeddings_representation()
        if rep is None:
            rep = PolarRepresentation.from_ball(embedder.embeddings())
        return rep.dist().detach().cpu().numpy()

    def _adjacency_in_embedder_order(self, G: nx.Graph) -> np.ndarray:
        """Binary ``(N, N)`` adjacency of ``G``, in ``embedder.nodes()`` order."""
        nodes = self.embedder.nodes()
        if nodes is None:
            raise RuntimeError(
                "Call embedder.fit() before FermiDiracGenerator.fit()."
            )
        order = {node: i for i, node in enumerate(nodes)}
        N = len(nodes)
        A = np.zeros((N, N), dtype=np.float64)
        for u, v in G.edges():
            if u not in order or v not in order:
                continue
            i, j = order[u], order[v]
            A[i, j] = A[j, i] = 1.0
        return A

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(embedder={self.embedder!r}, "
            f"r={self.r!r}, t={self.t!r})"
        )
