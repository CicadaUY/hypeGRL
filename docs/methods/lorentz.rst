Lorentz Embeddings
==================
**Geometry:** Hyperboloid 

**Reference:** Nickel & Kiela, *Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry*, ICML 2018

Overview
--------
Proposed as an alternative to :doc:`Poincaré Embeddings <poincare_embeddings>`, this approach performs representation learning in the Lorentz model instead, which the authors argue provides improved numerical stability for Riemannian optimization, particularly near the boundary of the Poincaré ball. 
Similarly to :doc:`Hydra <hydra>`, this method assumes the availability of pairwise similarity information between nodes.
Embeddings are learned so that more similar nodes are placed closer in hyperbolic space, using a softmax-based ranking loss that attracts similar pairs while pushing apart dissimilar ones. In the graph setting, direct neighbors are treated as the most similar nodes, while randomly sampled non-adjacent nodes serve as dissimilar examples.

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
...

