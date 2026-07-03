"""
Lorentz Embeddings embedder (Nickel & Kiela, 2018).

Implements the encoder-decoder instantiation of:
    Nickel & Kiela, "Learning Continuous Hierarchies in the Lorentz Model of
    Hyperbolic Geometry", ICML 2018.

The method reuses the soft-ranking objective of Poincaré Embeddings (Nickel &
Kiela, 2017) but embeds on the **Lorentz / hyperboloid model** instead of the
Poincaré ball. The contribution is geometric and numerical, not a new loss:
the Lorentz distance ``d_L(x, y) = arcosh(-<x, y>_L)`` (paper Eq. 5) has no
``(1 - ||x||^2)`` denominator, avoiding the numerical instability of the
Poincaré distance near the boundary, and admits an exact closed-form
exponential map (Eq. 9) so optimisation follows true geodesics.

Geometry
--------
Points live on the upper sheet of the two-sheeted hyperboloid
``H^d = {x in R^{d+1} : <x, x>_L = -1, x_0 > 0}`` with the Lorentzian scalar
product ``<x, y>_L = -x_0 y_0 + sum_i x_i y_i`` (Eq. 2-3). An intrinsic
dimension ``d`` therefore uses ``d + 1`` ambient coordinates. ``geoopt.Lorentz``
supplies the exact ``expmap`` (Eq. 9), so ``RiemannianAdam`` over it *is* the
paper's exact-geodesic optimiser. Embeddings are initialised near the origin
per Eq. 6: ``x' ~ U(-init_scale, init_scale)`` with ``x_0 = sqrt(1 + ||x'||^2)``.

For compatibility with the rest of hypeGRL (disk plotters, downstream code),
:meth:`embeddings` returns the **Poincaré-ball image** ``(N, d)`` via the
isometry ``p(x_0, x') = x' / (x_0 + 1)`` (Eq. 11) — exactly how the paper
visualises its embeddings. The raw hyperboloid coordinates ``(N, d+1)`` are
available through :meth:`hyperboloid_embeddings`.

Ranking loss and the similarity generalisation
-----------------------------------------------
The decoder is the raw pairwise Lorentz distance and the structural similarity
is the (possibly weighted) adjacency ``K = s(A)``. The loss is the soft-ranking
negative log-likelihood (paper Eq. 12)

.. math::

    \\mathcal{L} = -\\sum_{(i, j)} \\log
        \\frac{e^{-d_L(u_i, u_j)}}
             {\\sum_{z \\in N(i, j)} e^{-d_L(u_i, u_z)}},

with the neighbourhood set generalised from "non-neighbours" to *less-similar
nodes*:

.. math::

    N(i, j) = \\{z : K_{iz} < K_{ij}\\} \\cup \\{j\\}.

For an unweighted graph (``K_{ij} in {0, 1}``) this reduces exactly to the
Poincaré-Embeddings negative set (non-edges of ``i``). For a weighted graph the
similarity magnitude enters *only* through which nodes are negatives, as in the
paper. ``N(i, j)`` is subsampled to ``n_negatives`` draws per positive pair
(the paper subsamples following Jean et al., 2015).

Unknown edges
-------------
Unknown edges slot into the joint optimisation framework via ``joint_optimize``.
The membership test ``K_{iz} < K_{ij}`` is a hard threshold on ``K`` — a step
function with no gradient — so an unknown weight ``a_Omega`` that entered the
loss *only* by deciding set membership would receive no training signal (the
same obstruction that stops the shortest-path target of the Hydra embedders
from routing unknown-edge gradients). Two design choices keep imputation on the
differentiable side of that line:

- **Unknown pairs are excluded from every** ``N(i, j)``. Negatives are drawn
  strictly from *known* structure (``(i, z) not in Omega``), so the combinatorial
  set membership never depends on ``a_Omega``. Known non-edges (``K = 0``) and
  known lower-weight edges remain eligible; unknown pairs appear only as
  positives.
- **Positives are weighted by** ``K_{ij}`` **when — and only when — unknown
  edges are present.** This routes a smooth gradient to ``a_Omega`` (an unknown
  edge's imputed weight scales its own positive term), mirroring how the
  Poincaré-Embeddings ranking loss weights positives by ``A_{ij}``.

Differences from the paper (be explicit about extrapolation)
------------------------------------------------------------
The ``Omega = {}`` path is a faithful reference implementation of the paper's
objective; the departures are confined to regimes the paper does not define:

- **The** ``K_{ij}`` **weighting of positives is hypeGRL's device, gated on**
  ``Omega``. The paper's Eq. 12 is an *unweighted* sum — the similarity
  magnitude enters purely through ``N(i, j)``. With no unknown edges we
  reproduce that unweighted sum exactly (binary *and* weighted graphs); the
  weighting appears only when ``Omega`` is non-empty, i.e. only in the
  unknown-edge case the paper never covers. It is a no-op on binary graphs
  regardless (``K_{ij} = 1``).
- **Optimiser:** we use geoopt ``RiemannianAdam``; the paper's Algorithm 1 is
  exact-geodesic Riemannian *SGD*. The geometry is still exact — geoopt's
  Lorentz ``expmap`` follows the true geodesic (Eq. 9) — but the update rule is
  Adam, matching the rest of the library (and our Poincaré-Embeddings embedder).
- **Defaults** (``init_scale = 1e-3`` follows the paper's ``U(-0.001, 0.001)``;
  ``n_negatives = 50`` is hypeGRL's default, the paper subsamples without a
  stated count for these experiments).
"""

from __future__ import annotations

import warnings
from typing import Optional

import geoopt
import networkx as nx
import numpy as np
import torch

from hypegrl.embedders.base import HyperbolicEmbedder
from hypegrl.inference.joint_optimizer import joint_optimize
from hypegrl.inference.riemannian_optimizer import riemannian_optimize
from hypegrl.manifolds.lorentz import LORENTZ
from hypegrl.manifolds.poincare import lorentz_to_poincare, poincare_to_lorentz

# ---------------------------------------------------------------------------
# Distance decoder
# ---------------------------------------------------------------------------

def lorentz_distance_matrix(
    X: torch.Tensor,
    manifold: geoopt.Manifold = LORENTZ,
) -> torch.Tensor:
    """
    Pairwise Lorentz distance matrix ``d_L(x_i, x_j) = arcosh(-<x_i, x_j>_L)``.

    Parameters
    ----------
    X:
        ``(N, d+1)`` points on the hyperboloid (first coordinate timelike).
    manifold:
        Manifold on which the embeddings live (defaults to the shared
        ``geoopt.Lorentz`` instance).

    Returns
    -------
    ``(N, N)`` distance tensor (the ranking decoder output).
    """
    return manifold.dist(X.unsqueeze(1), X.unsqueeze(0))


# ---------------------------------------------------------------------------
# Ranking loss (Nickel & Kiela, 2018 — similarity generalisation)
# ---------------------------------------------------------------------------

def sample_similarity_negatives(
    A: torch.Tensor,
    i_idx: torch.Tensor,
    thresh: torch.Tensor,
    unknown_mask: torch.Tensor,
    n_negatives: int,
) -> torch.Tensor:
    """
    Sample ``n_negatives`` negatives for every positive pair.

    For positive ``(i, j)`` with threshold ``K_{ij}`` the candidate set is the
    *less-similar, known* neighbourhood ``{z != i : K_{iz} < K_{ij}
    and (i, z) not in Omega}`` (the ``N(i, j)`` of paper Eq. 12, minus ``j``,
    restricted to known pairs). Unknown pairs are excluded so ``a_Omega`` never
    enters the loss through the non-differentiable membership test. Sampling is
    with replacement, so ``n_negatives`` may exceed the number of distinct
    candidates.

    Two fallbacks keep the sampler well-defined on degenerate rows: a positive
    whose source node has no less-similar known node falls back to any known
    non-self node, and — only if the node's entire row is unknown — to any
    non-self node.

    Parameters
    ----------
    A:
        ``(N, N)`` similarity / adjacency matrix (detached; gradients are not
        needed for index sampling).
    i_idx:
        ``(P,)`` source-node index of each positive pair.
    thresh:
        ``(P,)`` similarity ``K_{ij}`` of each positive pair.
    unknown_mask:
        ``(N, N)`` boolean mask, ``True`` at unknown (``Omega``) positions.
    n_negatives:
        Number of negatives to draw per positive pair.

    Returns
    -------
    ``(P, n_negatives)`` long tensor of sampled column indices.
    """
    N = A.shape[0]
    not_self = ~torch.eye(N, dtype=torch.bool, device=A.device)
    row_not_self = not_self[i_idx]                       # (P, N)
    known = ~unknown_mask[i_idx]                          # (P, N)

    # Primary candidates: strictly less similar, known, not self.
    cand = (A[i_idx] < thresh.unsqueeze(1)) & known & row_not_self

    # Fallback 1: any known non-self node (ignore the threshold).
    empty = ~cand.any(dim=1)
    if empty.any():
        cand[empty] = (known & row_not_self)[empty]

    # Fallback 2: any non-self node (node whose whole row is unknown).
    empty = ~cand.any(dim=1)
    if empty.any():
        cand[empty] = row_not_self[empty]

    return torch.multinomial(cand.to(torch.float64), n_negatives, replacement=True)


def lorentz_ranking_nll(
    X: torch.Tensor,
    A: torch.Tensor,
    n_negatives: int,
    unknown_mask: torch.Tensor,
    weighted: bool,
    manifold: geoopt.Manifold = LORENTZ,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Soft-ranking negative log-likelihood in the Lorentz model (paper Eq. 12,
    similarity generalisation).

    For each positive pair ``(i, j)`` (an entry ``K_{ij} > 0``) the loss
    encourages ``d_L(u_i, u_j)`` to be small relative to the distances to
    ``n_negatives`` sampled *less-similar, known* nodes. When ``weighted`` is
    ``True`` each positive is scaled by ``K_{ij}``; when ``False`` every
    positive contributes equally, reproducing the paper's unweighted sum.

    Parameters
    ----------
    X:
        ``(N, d+1)`` hyperboloid embeddings.
    A:
        ``(N, N)`` similarity / adjacency matrix (may contain imputed unknown
        entries in the joint-optimisation path).
    n_negatives:
        Number of negatives sampled per positive pair (re-sampled every call).
    unknown_mask:
        ``(N, N)`` boolean mask, ``True`` at unknown (``Omega``) positions.
        All ``False`` when there are no unknown edges.
    weighted:
        Weight positives by ``K_{ij}`` (``True``) or sum unweighted (``False``).
        The embedder sets this to ``len(unknown_edges) > 0`` so the paper's
        unweighted objective is used whenever ``Omega`` is empty.
    manifold:
        Embedding manifold.
    eps:
        Numerical floor for the log argument.

    Returns
    -------
    Scalar loss tensor.
    """
    D = lorentz_distance_matrix(X, manifold)              # (N, N)

    i_idx, j_idx = torch.nonzero(A, as_tuple=True)        # (P,), (P,)
    if i_idx.numel() == 0:
        # No edges: nothing to rank. Keep X in the graph for a zero gradient.
        return D.sum() * 0.0

    thresh = A[i_idx, j_idx]                              # (P,) = K_ij

    neg_idx = sample_similarity_negatives(
        A.detach(), i_idx, thresh.detach(), unknown_mask, n_negatives
    )                                                     # (P, n_neg)

    d_pos = D[i_idx, j_idx]                               # (P,)
    d_neg = D[i_idx.unsqueeze(1), neg_idx]               # (P, n_neg)

    # denom_{ij} = e^{-d(i,j)} (the positive) + sum over sampled negatives.
    denom = torch.exp(-d_pos) + torch.exp(-d_neg).sum(dim=1)   # (P,)
    log_prob = -d_pos - torch.log(denom + eps)                # (P,)

    w = thresh if weighted else torch.ones_like(thresh)
    return -(w * log_prob).sum()


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class LorentzEmbeddingsEmbedder(HyperbolicEmbedder):
    """
    Lorentz Embeddings (Nickel & Kiela, 2018) on the hyperboloid.

    Optimises the soft-ranking objective of Poincaré Embeddings in the Lorentz
    model (exact-geodesic Riemannian optimisation), generalised to real-valued
    similarity so weighted graphs are handled natively. See the module
    docstring for the geometry, the ``N(i, j)`` similarity generalisation, and
    the unknown-edge handling.

    Parameters
    ----------
    d:
        Intrinsic embedding dimension (``2`` for visualisation, higher for
        downstream tasks). Points use ``d + 1`` ambient hyperboloid coordinates.
    n_negatives:
        Number of negatives sampled per positive pair per step. Default ``50``
        is hypeGRL's; the paper subsamples ``N(i, j)`` without a stated count.
    lr_X:
        Learning rate for Riemannian Adam on the embeddings.
    lr_a:
        Learning rate for Adam on the unknown edge weights.
    n_steps:
        Number of gradient steps.
    regularize_a:
        L2 regularisation on unknown edge weights.
    grad_clip:
        Maximum gradient norm. Set to ``0`` to disable clipping.
    log_every:
        Print loss every this many steps. ``0`` suppresses output.
    device:
        Torch device string.
    init_scale:
        Half-width of the uniform initialisation ``U(-init_scale, init_scale)``
        of the spatial coordinates ``x'`` (with ``x_0 = sqrt(1 + ||x'||^2)``,
        Eq. 6), used when ``X_init`` is not supplied. Default ``1e-3`` follows
        the paper's ``U(-0.001, 0.001)``.
    random_state:
        Seed for reproducible initialisation and negative sampling.

    Examples
    --------
    >>> import networkx as nx
    >>> from hypegrl.embedders.lorentz_embeddings import LorentzEmbeddingsEmbedder
    >>> G = nx.karate_club_graph()
    >>> emb = LorentzEmbeddingsEmbedder(d=2, n_steps=200, log_every=0)
    >>> emb.fit(G)
    LorentzEmbeddingsEmbedder(d=2)
    >>> emb.embeddings().shape          # Poincaré-ball image
    (34, 2)
    >>> emb.hyperboloid_embeddings().shape
    (34, 3)
    """

    def __init__(
        self,
        d: int = 2,
        n_negatives: int = 50,
        lr_X: float = 1e-2,
        lr_a: float = 1e-2,
        n_steps: int = 500,
        regularize_a: float = 0.0,
        grad_clip: float = 10.0,
        log_every: int = 50,
        device: str = "cpu",
        init_scale: float = 1e-3,     # paper: U(-0.001, 0.001)
        random_state: Optional[int] = None,
    ):
        self.d            = d
        self.n_negatives  = n_negatives
        self.lr_X         = lr_X
        self.lr_a         = lr_a
        self.n_steps      = n_steps
        self.regularize_a = regularize_a
        self.grad_clip    = grad_clip
        self.log_every    = log_every
        self.device       = device
        self.init_scale   = init_scale
        self.random_state = random_state

        self._X: Optional[np.ndarray]              = None   # Poincaré ball (N, d)
        self._X_hyper: Optional[np.ndarray]        = None   # hyperboloid (N, d+1)
        self._a_omega: Optional[np.ndarray]        = None
        self._loss_history: Optional[list[float]]  = None
        self._unknown_edges: list[tuple[int, int]] = []
        self._unknown_mask: Optional[torch.Tensor] = None
        self._G: Optional[nx.Graph]                = None

    # ------------------------------------------------------------------
    # HyperbolicEmbedder interface
    # ------------------------------------------------------------------

    def fit(
        self,
        G: nx.Graph,
        unknown_edges: Optional[list[tuple[int, int]]] = None,
        X_init: Optional[np.ndarray] = None,
        a_omega_init: Optional[np.ndarray] = None,
    ) -> "LorentzEmbeddingsEmbedder":
        """
        Fit Lorentz Embeddings, optionally with unknown edges.

        Parameters
        ----------
        G:
            Input graph. Edge weights are used when present (real-valued
            similarity); unweighted graphs give a binary target.
        unknown_edges:
            Edges treated as unknown; their weights are jointly optimised with
            the embeddings. When empty, the paper's unweighted Eq. 12 is used.
        X_init:
            ``(N, d+1)`` initial embeddings *on the hyperboloid*. Defaults to
            the Eq. 6 near-origin init. Note this is the hyperboloid
            representation, not the Poincaré-ball one returned by
            :meth:`embeddings`; use :meth:`hyperboloid_embeddings` to warm-start.
        a_omega_init:
            Initial estimates for unknown edge weights in ``(0, 1)``.

        Returns
        -------
        self
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)

        N = G.number_of_nodes()
        unknown_edges = unknown_edges or []

        if X_init is None:
            xr = np.random.uniform(-self.init_scale, self.init_scale, size=(N, self.d))
            x0 = np.sqrt(1.0 + (xr ** 2).sum(axis=1, keepdims=True))   # Eq. 6
            X_init = np.concatenate([x0, xr], axis=1)                  # (N, d+1)

        self._unknown_edges = unknown_edges
        self._unknown_mask = self._build_unknown_mask(N, unknown_edges)

        if not unknown_edges:
            # Fixed graph: structural similarity is the (constant) adjacency.
            result = riemannian_optimize(
                X_init    = X_init,
                s_A       = self.structural_similarity(G),
                loss_fn   = self.distance,
                manifold  = LORENTZ,
                lr        = self.lr_X,
                n_steps   = self.n_steps,
                grad_clip = self.grad_clip,
                log_every = self.log_every,
                device    = self.device,
            )
            result["a_omega"] = np.array([])
        else:
            result = joint_optimize(
                G             = G,
                loss_fn       = self.distance,
                X_init        = X_init,
                manifold      = LORENTZ,
                unknown_edges = unknown_edges,
                a_omega_init  = a_omega_init,
                lr_X          = self.lr_X,
                lr_a          = self.lr_a,
                n_steps       = self.n_steps,
                regularize_a  = self.regularize_a,
                grad_clip     = self.grad_clip,
                log_every     = self.log_every,
                device        = self.device,
                verbose       = self.log_every > 0,
            )

        self._X_hyper      = result["X"]
        self._X            = lorentz_to_poincare(self._X_hyper)
        self._a_omega      = result["a_omega"]
        self._loss_history = result["loss_history"]
        self._G            = G
        self._nodes        = list(G.nodes())   # rows follow G.nodes() (no reorder)
        return self

    def distance(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Soft-ranking loss in the Lorentz model.

        Positives are weighted by ``K_{ij}`` only when unknown edges are
        present (so ``Omega = {}`` reproduces the paper's unweighted Eq. 12);
        negatives are drawn from known, less-similar nodes.

        Parameters
        ----------
        X : torch.Tensor
            ``(N, d+1)`` hyperboloid embeddings.
        A : torch.Tensor
            ``(N, N)`` similarity / adjacency matrix (may contain imputed
            unknown entries).

        Returns
        -------
        Scalar loss tensor.
        """
        weighted = len(self._unknown_edges) > 0
        return lorentz_ranking_nll(
            X, A, self.n_negatives, self._unknown_mask, weighted, LORENTZ
        )

    def embeddings(self) -> np.ndarray:
        """
        Return the ``(N, d)`` Poincaré-ball image of the embeddings.

        The optimisation runs on the hyperboloid; this maps the result into the
        Poincaré ball via the isometry ``p(x_0, x') = x' / (x_0 + 1)`` (Eq. 11)
        so it drops into the disk plotters and the rest of hypeGRL. For the raw
        ``(N, d+1)`` hyperboloid coordinates use :meth:`hyperboloid_embeddings`.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if self._X is None:
            raise RuntimeError("Call fit() before embeddings().")
        return self._X

    def hyperboloid_embeddings(self) -> np.ndarray:
        """
        Return the raw ``(N, d+1)`` hyperboloid coordinates.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if self._X_hyper is None:
            raise RuntimeError("Call fit() before hyperboloid_embeddings().")
        return self._X_hyper

    def structural_similarity(self, G: nx.Graph) -> np.ndarray:
        """
        Return the (possibly weighted) adjacency matrix ``K = s(A)``.

        Edge weights are used when present, so ``K`` is the real-valued
        similarity of the paper's Eq. 12; unweighted graphs give a binary ``K``.
        The ranking loss derives its positive pairs and the ``K_{iz} < K_{ij}``
        negative candidates directly from this matrix.
        """
        return nx.to_numpy_array(G, dtype=np.float64)

    def decode(self, X: np.ndarray) -> np.ndarray:
        """
        Decoder output: the pairwise Lorentz distance matrix.

        Parameters
        ----------
        X:
            ``(N, d)`` Poincaré-ball embeddings (the output of
            :meth:`embeddings`). Mapped back to the hyperboloid before the
            distance is computed; the two models are isometric, so the distances
            are identical either way.

        Returns
        -------
        ``(N, N)`` NumPy array of hyperbolic distances.
        """
        H = torch.tensor(poincare_to_lorentz(X), dtype=torch.float64)
        return lorentz_distance_matrix(H).detach().numpy()

    # ------------------------------------------------------------------
    # Capability flags
    # ------------------------------------------------------------------

    def is_gradient_based(self) -> bool:
        return True

    def is_generative(self) -> bool:
        # The decoder is a distance, not an edge probability.
        return False

    def supports_update(self) -> bool:
        return True

    def supports_node_update(self) -> bool:
        return True

    def update(
        self,
        added_edges    = None,
        removed_edges  = None,
        revealed_edges = None,
        added_nodes    = None,
        removed_nodes  = None,
        node_edges     = None,
    ) -> "LorentzEmbeddingsEmbedder":
        """
        Warm-started refit after graph changes.

        Currently performs a full warm-started refit from the current
        hyperboloid embeddings; true incremental updates are not yet
        implemented.

        Parameters
        ----------
        added_edges:
            New edges; treated as unknown and added to ``Omega``.
        removed_edges:
            Edges to remove.
        revealed_edges:
            Dict ``{(i, j): weight}`` of previously unknown edges whose true
            weights are now known.
        added_nodes:
            New node IDs (embedded via warm-started refit).
        removed_nodes:
            Node IDs to remove.
        node_edges:
            Edges connecting new nodes to the existing graph.
        """
        if self._G is None:
            raise RuntimeError("Call fit() before update().")

        if added_nodes or removed_nodes:
            warnings.warn(
                "Full out-of-sample node extension is not yet implemented. "
                "Falling back to warm-started full refit.",
                stacklevel=2,
            )

        G_new = self._G.copy()

        if removed_nodes:
            self._check_connectivity_after_removal(G_new, nodes=removed_nodes)
            G_new.remove_nodes_from(removed_nodes)

        if removed_edges:
            self._check_connectivity_after_removal(G_new, edges=removed_edges)
            G_new.remove_edges_from(removed_edges)

        if added_edges:
            G_new.add_edges_from(added_edges)

        if revealed_edges:
            for (i, j), w in revealed_edges.items():
                G_new[i][j]["weight"] = w

        new_unknown = list(self._unknown_edges)

        if revealed_edges:
            revealed_set = {(min(i, j), max(i, j)) for i, j in revealed_edges}
            new_unknown = [
                (m, n) for (m, n) in new_unknown
                if (min(m, n), max(m, n)) not in revealed_set
            ]

        if added_edges:
            known = {(min(m, n), max(m, n)) for m, n in new_unknown}
            for e in added_edges:
                key = (min(e[0], e[1]), max(e[0], e[1]))
                if key not in known:
                    new_unknown.append(key)

        X_init = self._X_hyper
        if added_nodes and X_init is not None:
            n_new = len(added_nodes)
            xr = np.random.uniform(
                -self.init_scale, self.init_scale, size=(n_new, self.d)
            )
            x0 = np.sqrt(1.0 + (xr ** 2).sum(axis=1, keepdims=True))
            X_init = np.vstack([X_init, np.concatenate([x0, xr], axis=1)])

        return self.fit(G_new, unknown_edges=new_unknown, X_init=X_init)

    # ------------------------------------------------------------------
    # Extra accessors
    # ------------------------------------------------------------------

    @property
    def imputed_weights(self) -> Optional[np.ndarray]:
        """Imputed unknown edge weights after fitting."""
        return self._a_omega

    @property
    def loss_history(self) -> Optional[list[float]]:
        """Loss value at each optimisation step."""
        return self._loss_history

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_unknown_mask(
        self, N: int, unknown_edges: list[tuple[int, int]]
    ) -> torch.Tensor:
        """``(N, N)`` boolean mask, ``True`` at unknown (``Omega``) positions."""
        mask = torch.zeros(
            N, N, dtype=torch.bool, device=torch.device(self.device)
        )
        for (m, n) in unknown_edges:
            mask[m, n] = True
            mask[n, m] = True
        return mask

    @staticmethod
    def _check_connectivity_after_removal(
        G: nx.Graph,
        nodes: Optional[list] = None,
        edges: Optional[list] = None,
    ) -> None:
        G_test = G.copy()
        if nodes:
            G_test.remove_nodes_from(nodes)
        if edges:
            G_test.remove_edges_from(edges)
        if not nx.is_connected(G_test):
            raise ValueError(
                "The requested removal would disconnect the graph."
            )

    def __repr__(self) -> str:
        return f"LorentzEmbeddingsEmbedder(d={self.d})"
