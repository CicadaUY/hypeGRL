Poincaré Embeddings
===================

**Geometry:** Poincaré ball

**Reference:** Nickel & Kiela, *Poincaré Embeddings for Learning Hierarchical Representations*, NeurIPS 2017

Overview
--------

In this method, embeddings are learned in the Poincaré ball model. This model is conformal, allowing Riemannian gradients to be computed simply as Euclidean gradients rescaled by a position-dependent factor induced by the metric. Embeddings are kept within the open unit ball throughout, so the representation stays valid.

Unlike methods such as Hydra, which reconstruct a graph-derived distance matrix, Poincaré Embeddings operate directly on observed relations between nodes, minimizing a loss defined in terms of hyperbolic distances. The original formulation of Nickel and Kiela learns embeddings by maximizing the likelihood of observed node pairs under a ranking objective — bringing related nodes closer in hyperbolic space while pushing unrelated nodes apart through negative sampling — optimized with Riemannian stochastic gradient descent. This implementation keeps the objectives but optimizes with Riemannian Adam (see `Relation to reference implementations`_).

The paper actually uses *two* objectives for different data. The ranking objective above is used for the taxonomy-embedding experiments (Section 4.1). For network data (Section 4.2), it instead models the probability of an edge as a decreasing function of hyperbolic distance via a Fermi--Dirac distribution (following hyperbolic random graph models), trained with a cross-entropy loss. Both objectives are exposed here through the ``loss`` argument; the second gives a natural probabilistic encoder--decoder interpretation.

:class:`~hypegrl.embedders.poincare_embeddings.PoincareEmbeddingsEmbedder` is a *single-stage* gradient embedder: unlike the init-then-refine methods (HyperMap, D-Mercator), it optimizes node positions directly on the Poincaré ball from a near-origin initialisation, with an optional burn-in phase. Both objectives, partially observed graphs, and the burn-in are described below.

Encoder-decoder instantiation
-----------------------------

The original Poincaré Embeddings objective can be expressed in the encoder--decoder framework by taking

.. math::

   s(\mathbf A)
   =
   \{(i,j): A_{ij}=1\}

to be the set of observed node relations.

The decoder computes pairwise hyperbolic distances

.. math::

   \hat A_{ij}
   =
   d_P(\mathbf x_i,\mathbf x_j)
   =
   \operatorname{arcosh}\!\left(
   1+
   \frac{2\|\mathbf x_i-\mathbf x_j\|^2}
   {(1-\|\mathbf x_i\|^2)(1-\|\mathbf x_j\|^2)}
   \right).

The loss is then given by the negative log-likelihood (eq. 5 of the paper)

.. math::

   d\!\left(s(\mathbf A),\hat{\mathbf A}\right)
   =
   -
   \sum_{(i,j)\in s(\mathbf A)}
   \log
   \frac{
      e^{-\hat A_{ij}}
   }{
      \sum_{k\in\mathrm{Neg}(i,j)}
      e^{-\hat A_{ik}}
   },

where :math:`\mathrm{Neg}(i,j) = \{k: A_{ik}=0\} \cup \{j\}` is the set of nonobserved relations for node :math:`i`. To obtain a scalable optimization procedure, this set is approximated by retaining the positive example together with a fixed number of randomly sampled negatives.

Fermi--Dirac decoder (network embeddings)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the network experiments (Section 4.2 of the paper), distances are instead interpreted through a Fermi--Dirac edge probability model, motivated by hyperbolic random graph models. In this case, the decoder becomes (eq. 6 of the paper)

.. math::

   \hat A_{ij}
   =
   \frac{1}
   {\exp\!\left(
      \frac{d_P(\mathbf x_i,\mathbf x_j)-r}{t}
   \right)+1},

where :math:`r>0` and :math:`t>0` are hyperparameters controlling the characteristic connection radius and the sharpness of the transition, respectively.

Under this interpretation,

.. math::

   s(\mathbf A)=\mathbf A,

and a natural loss is the Bernoulli cross-entropy

.. math::

   d\!\left(s(\mathbf A),\hat{\mathbf A}\right)
   =
   -\sum_{i,j}
   \Bigl[
      A_{ij}\log \hat A_{ij}
      +
      (1-A_{ij})\log(1-\hat A_{ij})
   \Bigr].

How the embedder works
----------------------

:class:`~hypegrl.embedders.poincare_embeddings.PoincareEmbeddingsEmbedder`
optimizes node positions directly on the Poincaré ball (``geoopt.PoincareBall``)
with Riemannian Adam, starting from a near-origin uniform initialisation
(``init_scale``) unless ``X_init`` is supplied.
:meth:`~hypegrl.embedders.poincare_embeddings.PoincareEmbeddingsEmbedder.fit`
runs in up to two phases:

1. **Burn-in** (optional, ``burnin`` steps, on by default): a short phase at a
   reduced learning rate (``lr * burnin_lr_multiplier``) that, for the ranking
   loss, samples negatives :math:`\propto \deg^{\text{sample\_dampening}}`
   instead of uniformly, settling the angular layout before the main run. Set
   ``burnin=0`` to skip it.
2. **Main phase** (``n_steps`` steps): full learning rate and uniform negative
   sampling, warm-started from the burn-in result.

When ``unknown_edges`` are supplied, fitting routes through the shared joint
optimiser :func:`~hypegrl.inference.joint_optimizer.joint_optimize` (also used
by :class:`~hypegrl.embedders.poincare_maps.PoincareMapsEmbedder` and
:class:`~hypegrl.embedders.hypermap.HyperMapEmbedder`); otherwise it uses the
fixed-graph optimiser
:func:`~hypegrl.inference.riemannian_optimizer.riemannian_optimize`.

The matrix returned by
:meth:`~hypegrl.embedders.poincare_embeddings.PoincareEmbeddingsEmbedder.decode`
depends on ``loss``:

- ``"ranking"`` → the pairwise **hyperbolic distance** matrix (a *dissimilarity*,
  not bounded to :math:`[0, 1]`);
- ``"fermi_dirac"`` → the matrix of **connection probabilities** in
  :math:`(0, 1)`.

The capability flags follow suit:
:meth:`~hypegrl.embedders.poincare_embeddings.PoincareEmbeddingsEmbedder.is_generative`
is ``True`` only for ``"fermi_dirac"`` (the only mode with a probabilistic
decoder), while ``supports_update`` and ``supports_node_update`` are
warm-started refits.

Implementation notes
--------------------

**Optimiser and manifold.** Embeddings are wrapped as
``geoopt.ManifoldParameter`` on the Poincaré ball and optimized with Riemannian
Adam; the retraction keeps every point inside the open unit ball, so no explicit
projection step is needed. The full pairwise distance matrix is formed each step
(dense, :math:`O(N^2)`), which is why the embedder targets the moderate graph
sizes typical of hypeGRL rather than the very large taxonomies of the original
paper.

**Partially observed graphs.** Unknown edges are a first-class feature here (the
library's central capability). Edges passed via ``fit(..., unknown_edges=...)``
are treated as free variables and jointly optimized through
:func:`~hypegrl.inference.joint_optimizer.joint_optimize`. In the ranking loss
each positive pair is weighted by its (possibly imputed) adjacency entry
:math:`A_{ij}`, so gradients reach the imputed weights; negatives are drawn only
from *known* non-edges (:math:`A_{ik}=0`), so unobserved pairs are never pushed
apart. Unlike :class:`~hypegrl.embedders.dmercator.DMercatorEmbedder` (which
zero-imputes and warns), this is fully supported
(``test_pe_with_unknown_edges``, ``test_pe_burnin_with_unknown_edges``).

**Burn-in and negative sampling.** Negatives for the ranking loss are sampled
once per node and shared across that node's positives, re-sampled every step.
During burn-in they are drawn :math:`\propto \deg^{0.75}` (``sample_dampening``);
otherwise uniformly. ``burnin`` defaults to the reference-code value of ``20``;
``test_pe_burnin_default_matches_reference`` and ``test_pe_burnin_zero_disables``
guard the parameter, and ``test_pe_ranking_loss_decreases`` checks that the loss
falls over a run.

**Decoder outputs.** ``decode`` returns a hyperbolic-distance matrix under
``"ranking"`` and a Fermi--Dirac probability matrix under ``"fermi_dirac"``
(``test_pe_decode_ranking_is_distance``,
``test_pe_decode_fermi_dirac_is_probability``); only the latter mode reports
:meth:`~hypegrl.embedders.poincare_embeddings.PoincareEmbeddingsEmbedder.is_generative`
as ``True`` (``test_pe_capability_flags``).

Relation to reference implementations
-------------------------------------

The distance and both objectives match the paper exactly; the optimisation
differs from the authors' reference implementation
(``facebookresearch/poincare-embeddings``) in a few deliberate ways:

- **Optimiser.** We optimise with geoopt ``RiemannianAdam``. The reference uses
  a custom ``RiemannianSGD`` (natural-gradient update with ``lr=1000``), so our
  ``lr_X`` default is on a different scale (Adam, ~\ ``1e-2``).
- **Execution model and negative sampling.** We take full-batch gradient steps
  over a dense :math:`(N, N)` distance matrix and sample negatives once per
  node, shared across that node's positives. The reference does minibatch SGD
  over an edge list and samples fresh negatives per positive pair. (Burn-in
  itself now matches the reference — on by default, see
  `Implementation notes`_ — modulo our full-batch "step" standing in for the
  reference's epoch.)
- **Fermi--Dirac is computed over all pairs.** ``fermi_dirac_nll`` sums the
  cross-entropy over every :math:`i<j` (exact, :math:`O(N^2)`); the paper
  negatively samples this term as well.
- **Defaults follow the reference code** (``init_scale=1e-4``,
  ``n_negatives=50``, ``burnin=20``), which differ from the values quoted in the
  *paper* (``1e-3``, ``10``, and a 10-epoch burn-in). Note also that the
  authors' public repository only ships the ranking loss; the Fermi--Dirac
  objective appears in the paper but not in that code.

Comparison with gensim's ``PoincareModel``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``gensim.models.poincare.PoincareModel`` is a third faithful implementation of
the ranking objective. Its distance, softmax-ranking loss and reduced-lr
burn-in match ours, and it shares the same structural differences as the
reference code above (RSGD natural gradient rather than geoopt
``RiemannianAdam``; minibatch SGD over an edge list rather than full-batch;
fresh negatives per positive pair rather than per node). Beyond those, a few
behaviours are specific to gensim:

- **Negative sampling is degree-proportional in every phase.** gensim draws
  negatives :math:`\propto \deg` (linear) throughout training, whereas we
  sample uniformly except during burn-in (where we use
  :math:`\propto \deg^{0.75}`); gensim's burn-in only lowers the learning rate.
- **L2 regularisation on embeddings.** gensim's loss adds
  ``regularization_coeff * ||v||^2`` (default ``1.0``), pulling embeddings
  toward the origin; we apply no embedding regularisation (``regularize_a``
  only affects imputed unknown-edge weights).
- **Ranking only, binary, no unknown edges.** gensim implements no
  Fermi--Dirac decoder, consumes a binary edge list (the project wrapper
  symmetrises an adjacency into ``(i, j)``/``(j, i)`` pairs), and has no notion
  of weighted or unknown edges — unlike our ``A_ij``-weighted positives and
  joint unknown-edge optimisation.

API reference
-------------
.. autoclass:: hypegrl.embedders.poincare_embeddings.PoincareEmbeddingsEmbedder
   :members:
