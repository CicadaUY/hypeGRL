Embedders
=========

The base class — the interface every embedder implements (including the
``nodes()`` accessor for the ``embeddings()`` row order). Each concrete method
(Poincaré Maps, Poincaré Embeddings, Lorentz Embeddings, HyperMap, D-Mercator,
Hydra) is documented on its own page under **Embedding methods**.

.. automodule:: hypegrl.embedders.base
   :members:
   :show-inheritance:

Precomputed embeddings
----------------------

Wraps a coordinate matrix that was not produced by fitting one of the
library's methods, so it can be passed anywhere a fitted embedder is expected.

.. automodule:: hypegrl.embedders.precomputed
   :members:
   :show-inheritance:
