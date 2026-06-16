Mercator and D-Mercator
==========

**Geometry:** Spherical 

**Reference:** Jankowski et al., *The D-Mercator method for the multidimensional hyperbolic embedding of real networks*, Nature Communications 2023

Overview
--------
These methods operate in the native (polar) representation and assume the same underlying generative model as Poincaré Embeddings, in which the probability of connection also follows a Fermi--Dirac distribution that decreases with hyperbolic distance. Such models are known to generate networks that simultaneously exhibit strong clustering, small-world properties, and heterogeneous degree distributions. 

In this setup radial coordinates can be interpreted as latent "popularity" variables controlling expected node degrees, while angular coordinates encode "similarity" between nodes. Mercator first computes an initial embedding by applying a model-corrected Laplacian Eigenmaps procedure to recover an approximate angular ordering of the nodes. This initial configuration is subsequently refined by maximizing the likelihood of the observed network under the assumed hyperbolic generative model. D-Mercator extends this framework by allowing similarity to be represented using :math:`D` angular dimensions.

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
