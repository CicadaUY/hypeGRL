Poincaré Embeddings
===================

**Geometry:** Poincaré ball

**Reference:** Nickel & Kiela, *Poincaré Embeddings for Learning Hierarchical Representations*, NeurIPS 2017

Overview
--------

In this method, embeddings are learned in the Poincaré ball model. This model is conformal, allowing Riemannian gradients to be computed simply as Euclidean gradients rescaled by a position-dependent factor induced by the metric. After each update, embeddings are projected back into the unit ball to ensure validity of the representation.

Learning is performed using Riemannian stochastic gradient descent, minimizing a loss function defined in terms of hyperbolic distances. Unlike methods such as Hydra, which seek to reconstruct a graph-derived distance matrix, Poincaré Embeddings operate directly on observed relations between nodes. The original formulation of Nickel and Kiela learns embeddings by maximizing the likelihood of observed node pairs under a ranking objective, bringing related nodes closer in hyperbolic space while pushing unrelated nodes apart through negative sampling.

The paper actually uses *two* objectives for different data. The ranking objective above is used for the taxonomy-embedding experiments (Section 4.1). For network data (Section 4.2), it instead models the probability of an edge as a decreasing function of hyperbolic distance via a Fermi--Dirac distribution (following hyperbolic random graph models), trained with a cross-entropy loss. Both objectives are exposed here through the ``loss`` argument; the second gives a natural probabilistic encoder--decoder interpretation.

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

The loss is then given by the negative log-likelihood

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

For the network experiments (Section 4.2 of the paper), distances are instead interpreted through a Fermi--Dirac edge probability model, motivated by hyperbolic random graph models. In this case, the decoder becomes

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

Implementation details
----------------------

Both objectives are selected through the ``loss`` argument of
:class:`~hypegrl.embedders.poincare_embeddings.PoincareEmbeddingsEmbedder`
(``"ranking"`` for eq. (5), ``"fermi_dirac"`` for the network objective). The
distance and the objectives match the paper exactly; the optimisation differs
from the authors' reference implementation in a few deliberate ways:

- **Optimiser.** We optimise with geoopt ``RiemannianAdam``. The reference uses
  a custom ``RiemannianSGD`` (natural-gradient update with ``lr=1000``); our
  ``lr_X`` default is therefore on a different scale (Adam, ~\ ``1e-2``).
- **Burn-in.** Like the reference, an initial burn-in phase runs by default —
  reduced learning rate together with degree-dampened (:math:`\propto
  \deg^{0.75}`) negative sampling — to settle the angular layout before the
  main run. It is controlled by the ``burnin`` constructor argument (with
  ``burnin_lr_multiplier`` and ``sample_dampening``); set ``burnin=0`` to
  disable it.
- **Negative sampling.** For the ranking loss we draw ``n_negatives`` negatives
  per node, re-sampled each step, sharing them across that node's positives;
  the reference samples fresh negatives per positive pair. For unknown edges,
  negatives are drawn only from *known* non-edges so imputed pairs are never
  pushed apart.
- **Fermi--Dirac is computed over all pairs.** ``fermi_dirac_nll`` sums the
  cross-entropy over every :math:`i<j` (exact, :math:`O(N^2)`); the paper
  negatively samples this term as well.
- **Defaults follow the reference *code*** (``init_scale=1e-4``,
  ``n_negatives=50``), which differ from the values quoted in the *paper*
  (``1e-3`` and ``10``). Note also that the authors' public repository
  (``facebookresearch/poincare-embeddings``) only ships the ranking loss; the
  Fermi--Dirac objective appears in the paper but not in that code.

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
