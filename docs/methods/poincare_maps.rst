Poincaré Maps
=============

**Geometry:** Poincaré disk 

**Reference:** Klimovskaia et al., *Poincaré Maps for Analyzing Complex Hierarchies*, Nature Communications 2020

Overview
--------
Given a graph, and in particular its Laplacian matrix :math:`\mathbf{L}`, the Relative Forest Accessibility (RFA) matrix is computed as :math:`(\mathbf{I} + \mathbf{L} )^{-1}`. This matrix is doubly stochastic and can be interpreted as defining probability distributions over nodes.

The embedding objective in Poincaré Maps is formulated by defining similarities between hyperbolic embeddings as normalized negative exponentials of the hyperbolic distance, and minimizing the symmetric Kullback--Leibler divergence between these similarities and the node-wise probability distributions induced by the RFA matrix. The resulting optimization problem is solved using Riemannian stochastic gradient descent in the Poincaré ball.

Encoder-decoder instantiation
------------------------------
.. math::
   \hat{A}_{ij} = \frac{e^{-d_H(x_i,x_j)/\gamma}}{\sum_k e^{-d_H(x_i,x_k)/\gamma}}
   \quad
   s(\mathbf{A}) = (\mathbf{I}+\mathbf{L})^{-1}
   \quad
   d = \sum_i \mathrm{SymKL}

Unknown edges
-------------
...

API reference
-------------
.. autoclass:: hypegrl.embedders.poincare_maps.PoincareMapsEmbedder
   :members:
