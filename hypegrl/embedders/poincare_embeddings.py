"""
Poincaré Embeddings embedder (Nickel & Kiela, 2017).

Implements the encoder-decoder instantiation of:
    Nickel & Kiela, "Poincaré Embeddings for Learning Hierarchical
    Representations", NeurIPS 2017.

Two decoders / losses are provided, both living on the Poincaré ball
(``POINCARE_BALL``):

**Ranking loss (default, the original objective).**
The decoder is the raw pairwise hyperbolic distance and the structural
similarity is the set of observed relations (the adjacency matrix). The
loss is the soft-ranking negative log-likelihood

.. math::

    \\mathcal{L} = -\\sum_{(i,j)\\in s(A)} \\log
        \\frac{e^{-d_P(x_i,x_j)}}
             {\\sum_{k\\in \\mathrm{Neg}(i,j)} e^{-d_P(x_i,x_k)}},

with :math:`\\mathrm{Neg}(i,j) = \\{k: A_{ik}=0\\}\\cup\\{j\\}`. For
scalability the negative set is approximated by the positive together with
``n_negatives`` uniformly sampled non-neighbours, re-sampled every step
(stochastic, as in the original paper).

**Fermi-Dirac loss (the paper's network-embedding objective).**
The decoder is the Fermi-Dirac edge probability
``P_ij = 1/(exp((d_P - r)/t) + 1)`` (paper eq. 6, Section 4.2) and the loss
is the Bernoulli cross-entropy against the adjacency matrix. The paper uses
this for the collaboration-network experiments; the ranking loss is used for
the WordNet taxonomy experiments (Section 4.1).

Unknown edges
-------------
Unknown edges slot into the joint optimisation framework via
``joint_optimize``. In the ranking loss each positive pair is weighted by
its (possibly imputed) adjacency entry ``A_ij``, so gradients flow to the
unknown edge weights; sampled negatives are drawn only from *known*
non-edges (``A_ik == 0``), never from imputed pairs.

Differences from the reference implementation
---------------------------------------------
The objective and distance match the paper exactly; the optimisation does
not, by design:

- **Optimiser:** we use geoopt ``RiemannianAdam``; the reference uses a
  custom ``RiemannianSGD`` (natural gradient, ``lr=1000``). Our ``lr_X``
  default is therefore on a different scale (Adam, ~1e-2).
- **No burn-in:** the reference runs an initial burn-in phase (reduced lr +
  degree-dampened negative sampling) to settle the angular layout. We do not.
- **Fermi-Dirac uses *all* pairs:** ``fermi_dirac_nll`` sums the Bernoulli
  cross-entropy over every ``i < j`` (exact, O(N^2)); the paper negatively
  samples it as in the ranking loss. Fine at the graph sizes hypeGRL targets.
- **Defaults** (``init_scale=1e-4``, ``n_negatives=50``) follow the reference
  *code* (facebookresearch/poincare-embeddings), which differ from the
  *paper* (1e-3 / 10). That repo only ships the ranking loss; the Fermi-Dirac
  objective lives in the paper, not its code.
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
from hypegrl.manifolds.poincare import POINCARE_BALL

# ---------------------------------------------------------------------------
# Distance decoder
# ---------------------------------------------------------------------------

def poincare_distance_matrix(
    X: torch.Tensor,
    manifold: geoopt.Manifold = POINCARE_BALL,
) -> torch.Tensor:
    """
    Pairwise hyperbolic distance matrix ``d_P(x_i, x_j)``.

    Parameters
    ----------
    X:
        ``(N, d)`` Poincaré ball embeddings.
    manifold:
        Manifold on which the embeddings live (defaults to the shared
        unit-curvature Poincaré ball).

    Returns
    -------
    ``(N, N)`` distance tensor (the ranking decoder output).
    """
    return manifold.dist(X.unsqueeze(1), X.unsqueeze(0))


# ---------------------------------------------------------------------------
# Ranking loss (Nickel & Kiela, 2017)
# ---------------------------------------------------------------------------

def sample_negatives(
    A: torch.Tensor,
    n_negatives: int,
) -> torch.Tensor:
    """
    Sample ``n_negatives`` non-neighbour indices for every node.

    For row ``i`` candidates are the known non-edges ``{k != i : A_ik == 0}``.
    Imputed unknown edges (``A_ik > 0``) are excluded so the model never
    pushes apart pairs whose relation is merely unobserved. Rows with no
    available non-edge (fully connected node) fall back to sampling from all
    other nodes. Sampling is with replacement, so ``n_negatives`` may exceed
    the number of distinct candidates.

    Parameters
    ----------
    A:
        ``(N, N)`` adjacency matrix (detached; gradients are not needed for
        index sampling).
    n_negatives:
        Number of negatives to draw per node.

    Returns
    -------
    ``(N, n_negatives)`` long tensor of sampled column indices.
    """
    # Uniform over non-neighbours, sampled once per node and shared across its
    # positives. The reference rejects neighbours the same way, but samples
    # fresh per positive pair and degree-dampens during burn-in (not done here).
    N = A.shape[0]
    not_self = ~torch.eye(N, dtype=torch.bool, device=A.device)
    probs = ((A == 0) & not_self).to(torch.float64)

    empty = probs.sum(dim=1) == 0
    if empty.any():
        probs[empty] = not_self[empty].to(torch.float64)

    return torch.multinomial(probs, n_negatives, replacement=True)


def ranking_nll(
    X: torch.Tensor,
    A: torch.Tensor,
    n_negatives: int = 10,
    manifold: geoopt.Manifold = POINCARE_BALL,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Soft-ranking negative log-likelihood of Nickel & Kiela (2017).

    For each observed pair ``(i, j)`` the loss encourages ``d_P(x_i, x_j)``
    to be small relative to the distances to sampled negatives. Each positive
    pair is weighted by ``A_ij``: this is ``1`` for ordinary edges and equals
    the imputed weight for unknown edges, routing gradients to the joint
    optimiser's free variables.

    Parameters
    ----------
    X:
        ``(N, d)`` Poincaré ball embeddings.
    A:
        ``(N, N)`` adjacency matrix (may contain imputed unknown entries).
    n_negatives:
        Number of negatives sampled per node (re-sampled on every call).
    manifold:
        Embedding manifold.
    eps:
        Numerical floor for the log argument.

    Returns
    -------
    Scalar loss tensor.
    """
    D = poincare_distance_matrix(X, manifold)              # (N, N)

    neg_idx = sample_negatives(A.detach(), n_negatives)    # (N, n_neg)
    d_neg = torch.gather(D, 1, neg_idx)                    # (N, n_neg)
    exp_neg_sum = torch.exp(-d_neg).sum(dim=1)             # (N,)

    # denom_{ij} = e^{-d_ij} (the positive) + sum over node i's negatives
    denom = torch.exp(-D) + exp_neg_sum.unsqueeze(1)       # (N, N)
    log_prob = -D - torch.log(denom + eps)                # (N, N)

    return -(A * log_prob).sum()


# ---------------------------------------------------------------------------
# Fermi-Dirac decoder / loss (alternative probabilistic interpretation)
# ---------------------------------------------------------------------------

def fermi_dirac_decoder(
    X: torch.Tensor,
    r: float,
    t: float,
    manifold: geoopt.Manifold = POINCARE_BALL,
) -> torch.Tensor:
    """
    Fermi-Dirac edge-probability decoder.

    .. math::

        \\hat A_{ij} = \\frac{1}{\\exp\\!\\big((d_P(x_i,x_j) - r)/t\\big) + 1}.

    Parameters
    ----------
    X:
        ``(N, d)`` Poincaré ball embeddings.
    r:
        Characteristic connection radius (``r > 0``).
    t:
        Temperature controlling the sharpness of the transition (``t > 0``).
    manifold:
        Embedding manifold.

    Returns
    -------
    ``(N, N)`` matrix of connection probabilities in ``(0, 1)``.
    """
    D = poincare_distance_matrix(X, manifold)
    return torch.sigmoid((r - D) / t)


def fermi_dirac_nll(
    X: torch.Tensor,
    A: torch.Tensor,
    r: float,
    t: float,
    manifold: geoopt.Manifold = POINCARE_BALL,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Bernoulli cross-entropy between the adjacency and the Fermi-Dirac decoder.

    .. math::

        \\mathcal{L} = -\\sum_{i<j}
            \\big[A_{ij}\\log\\hat A_{ij}
                  + (1-A_{ij})\\log(1-\\hat A_{ij})\\big].

    Parameters
    ----------
    X:
        ``(N, d)`` Poincaré ball embeddings.
    A:
        ``(N, N)`` adjacency matrix (may contain imputed unknown entries).
    r, t:
        Fermi-Dirac radius and temperature.
    manifold:
        Embedding manifold.
    eps:
        Numerical floor for the log arguments.

    Returns
    -------
    Scalar loss tensor.
    """
    P = fermi_dirac_decoder(X, r, t, manifold)
    P = torch.clamp(P, eps, 1.0 - eps)

    # Exact: sum over every i < j. The paper negatively samples this term
    # instead; we can afford the full O(N^2) sum at hypeGRL's graph sizes.
    N = X.shape[0]
    mask = torch.triu(
        torch.ones(N, N, dtype=torch.bool, device=X.device), diagonal=1
    )
    nll = -(A[mask] * torch.log(P[mask])
            + (1.0 - A[mask]) * torch.log(1.0 - P[mask]))
    return nll.sum()


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class PoincareEmbeddingsEmbedder(HyperbolicEmbedder):
    """
    Poincaré Embeddings (Nickel & Kiela, 2017) on the Poincaré ball.

    Parameters
    ----------
    d:
        Embedding dimension (2 for visualisation, higher for downstream tasks).
    loss:
        ``"ranking"`` (default) for the original soft-ranking NLL with
        negative sampling, or ``"fermi_dirac"`` for the Bernoulli
        cross-entropy under the Fermi-Dirac edge-probability decoder.
    n_negatives:
        Number of negatives sampled per node per step (``"ranking"`` only).
        Default ``50`` follows the reference code (the paper uses ``10``).
    r:
        Fermi-Dirac connection radius (``"fermi_dirac"`` only).
    t:
        Fermi-Dirac temperature (``"fermi_dirac"`` only).
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
        used when ``X_init`` is not supplied (near-origin start, as in the
        original). Default ``1e-4`` follows the reference code (the paper
        uses ``1e-3``).
    random_state:
        Seed for reproducible initialisation and negative sampling.

    Examples
    --------
    >>> import networkx as nx
    >>> from hypegrl.embedders.poincare_embeddings import PoincareEmbeddingsEmbedder
    >>> G = nx.karate_club_graph()
    >>> emb = PoincareEmbeddingsEmbedder(d=2, n_steps=200, log_every=0)
    >>> emb.fit(G)
    PoincareEmbeddingsEmbedder(d=2, loss='ranking')
    >>> emb.embeddings().shape
    (34, 2)
    """

    def __init__(
        self,
        d: int = 2,
        loss: str = "ranking",
        n_negatives: int = 50,        # reference-code default (paper uses 10)
        r: float = 2.0,
        t: float = 1.0,
        lr_X: float = 1e-2,
        lr_a: float = 1e-2,
        n_steps: int = 500,
        regularize_a: float = 0.0,
        grad_clip: float = 10.0,
        log_every: int = 50,
        device: str = "cpu",
        init_scale: float = 1e-4,     # reference-code default (paper uses 1e-3)
        random_state: Optional[int] = None,
    ):
        if loss not in ("ranking", "fermi_dirac"):
            raise ValueError(
                f"loss must be 'ranking' or 'fermi_dirac', got {loss!r}."
            )

        self.d            = d
        self.loss         = loss
        self.n_negatives  = n_negatives
        self.r            = r
        self.t            = t
        self.lr_X         = lr_X
        self.lr_a         = lr_a
        self.n_steps      = n_steps
        self.regularize_a = regularize_a
        self.grad_clip    = grad_clip
        self.log_every    = log_every
        self.device       = device
        self.init_scale   = init_scale
        self.random_state = random_state

        self._X: Optional[np.ndarray]               = None
        self._a_omega: Optional[np.ndarray]         = None
        self._loss_history: Optional[list[float]]   = None
        self._unknown_edges: list[tuple[int, int]]  = []
        self._G: Optional[nx.Graph]                 = None

    # ------------------------------------------------------------------
    # HyperbolicEmbedder interface
    # ------------------------------------------------------------------

    def fit(
        self,
        G: nx.Graph,
        unknown_edges: Optional[list[tuple[int, int]]] = None,
        X_init: Optional[np.ndarray] = None,
        a_omega_init: Optional[np.ndarray] = None,
    ) -> "PoincareEmbeddingsEmbedder":
        """
        Fit Poincaré Embeddings, optionally with unknown edges.

        Parameters
        ----------
        G:
            Input graph.
        unknown_edges:
            Edges treated as unknown; their weights are jointly optimised
            with the embeddings.
        X_init:
            ``(N, d)`` initial embeddings inside the Poincaré ball. Defaults
            to ``U(-init_scale, init_scale)`` near the origin.
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
            X_init = np.random.uniform(
                -self.init_scale, self.init_scale, size=(N, self.d)
            )

        if not unknown_edges:
            # Fixed graph: structural similarity is the (constant) adjacency.
            s_A = self.structural_similarity(G)

            def loss_fn(X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
                return self.distance(X, A)

            result = riemannian_optimize(
                X_init    = X_init,
                s_A       = s_A,
                loss_fn   = loss_fn,
                manifold  = POINCARE_BALL,
                lr        = self.lr_X,
                n_steps   = self.n_steps,
                grad_clip = self.grad_clip,
                log_every = self.log_every,
                device    = self.device,
            )
            self._a_omega = np.array([])

        else:
            result = joint_optimize(
                G             = G,
                loss_fn       = self.distance,
                X_init        = X_init,
                manifold      = POINCARE_BALL,
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
            self._a_omega = result["a_omega"]

        self._X             = result["X"]
        self._loss_history  = result["loss_history"]
        self._unknown_edges = unknown_edges
        self._G             = G
        return self

    def distance(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Encoder-decoder loss selected by ``self.loss``.

        Parameters
        ----------
        X : torch.Tensor
            ``(N, d)`` embeddings on the Poincaré ball.
        A : torch.Tensor
            ``(N, N)`` adjacency matrix (may contain imputed unknown entries).

        Returns
        -------
        Scalar loss tensor.
        """
        if self.loss == "ranking":
            return ranking_nll(X, A, self.n_negatives, POINCARE_BALL)
        return fermi_dirac_nll(X, A, self.r, self.t, POINCARE_BALL)

    def embeddings(self) -> np.ndarray:
        """
        Return the ``(N, d)`` embedding matrix.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if self._X is None:
            raise RuntimeError("Call fit() before embeddings().")
        return self._X

    def structural_similarity(self, G: nx.Graph) -> np.ndarray:
        """
        Return the adjacency matrix ``s(A) = A``.

        The observed relations are the edges of ``G``; the ranking loss
        derives its positive pairs (and the candidate negatives) directly
        from this matrix.
        """
        return nx.to_numpy_array(G, dtype=np.float64)

    def decode(self, X: np.ndarray) -> np.ndarray:
        """
        Decoder output induced by embeddings ``X``.

        For ``loss="ranking"`` this is the pairwise hyperbolic distance
        matrix; for ``loss="fermi_dirac"`` it is the matrix of Fermi-Dirac
        connection probabilities.

        Parameters
        ----------
        X:
            ``(N, d)`` Poincaré ball embeddings.

        Returns
        -------
        ``(N, N)`` NumPy array.
        """
        X_t = torch.tensor(X, dtype=torch.float64)
        if self.loss == "ranking":
            return poincare_distance_matrix(X_t).detach().numpy()
        return fermi_dirac_decoder(X_t, self.r, self.t).detach().numpy()

    # ------------------------------------------------------------------
    # Capability flags
    # ------------------------------------------------------------------

    def is_gradient_based(self) -> bool:
        return True

    def is_generative(self) -> bool:
        # Generative via the Fermi-Dirac edge-probability decoder.
        return self.loss == "fermi_dirac"

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
    ) -> "PoincareEmbeddingsEmbedder":
        """
        Warm-started refit after graph changes.

        Currently performs a full warm-started refit from the current
        embeddings; true incremental updates are not yet implemented.

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

        X_init = self._X
        if added_nodes and X_init is not None:
            n_new = len(added_nodes)
            new_rows = np.random.uniform(
                -self.init_scale, self.init_scale, size=(n_new, self.d)
            )
            X_init = np.vstack([X_init, new_rows])

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
        return (
            f"PoincareEmbeddingsEmbedder(d={self.d}, loss={self.loss!r})"
        )
