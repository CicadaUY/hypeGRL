Poincaré Embeddings
===================

**Geometry:** Poincaré disk 

**Reference:** Nickel & Kiela, *Poincaré Embeddings for Learning Hierarchical Representations*, NeurIPS 2017

Overview
--------
In this method, embeddings are learned in the Poincaré ball model. This model is conformal, allowing Riemannian gradients to be computed simply as Euclidean gradients rescaled by a position-dependent factor induced by the metric. After each update, embeddings are projected back into the unit ball to ensure validity of the representation.
Learning is thus performed using Riemannian stochastic gradient descent, minimizing an application-dependent loss function defined in terms of hyperbolic distances. In the case of network embeddings, the probability of observing an edge between two nodes is modeled using a Fermi--Dirac distribution that decreases with their hyperbolic distance, and training proceeds by minimizing a cross-entropy loss with negative sampling. 

Encoder-decoder instantiation
------------------------------
.. math::
   \hat{A}_{ij} = \frac{e^{-d_H(x_i,x_j)/\gamma}}{\sum_k e^{-d_H(x_i,x_k)/\gamma}}
   \quad
   s(A) = (I+L)^{-1}
   \quad
   d = \sum_i \mathrm{SymKL}

Unknown edges
-------------
...

API reference
-------------
.. autoclass:: hypegrl.embedders.poincare_maps.PoincareMapsEmbedder
   :members:
