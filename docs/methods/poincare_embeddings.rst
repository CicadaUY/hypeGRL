Poincaré Embeddings
===================

**Geometry:** Poincaré ball

**Reference:** Nickel & Kiela, *Poincaré Embeddings for Learning Hierarchical Representations*, NeurIPS 2017

Overview
--------

In this method, embeddings are learned in the Poincaré ball model. This model is conformal, allowing Riemannian gradients to be computed simply as Euclidean gradients rescaled by a position-dependent factor induced by the metric. After each update, embeddings are projected back into the unit ball to ensure validity of the representation.

Learning is performed using Riemannian stochastic gradient descent, minimizing a loss function defined in terms of hyperbolic distances. Unlike methods such as Hydra, which seek to reconstruct a graph-derived distance matrix, Poincaré Embeddings operate directly on observed relations between nodes. The original formulation of Nickel and Kiela learns embeddings by maximizing the likelihood of observed node pairs under a ranking objective, bringing related nodes closer in hyperbolic space while pushing unrelated nodes apart through negative sampling.

For network data, a closely related interpretation arises from hyperbolic random graph models, where the probability of an edge is modeled as a decreasing function of hyperbolic distance via a Fermi--Dirac distribution. This probabilistic decoder is not the objective used in the original paper, but provides a natural encoder--decoder interpretation of the learned embeddings.

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

Alternative probabilistic decoder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A commonly used alternative, motivated by hyperbolic random graph models, is to interpret distances through a Fermi--Dirac edge probability model. In this case, the decoder becomes

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

API reference
-------------
.. autoclass:: hypegrl.embedders.poincare_embeddings.PoincareEmbeddingsEmbedder
   :members:
