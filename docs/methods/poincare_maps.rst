Poincaré Maps
=============

**Geometry:** Poincaré disk  
**Reference:** Klimovskaia et al., Nature Communications 2020

Overview
--------
...

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
