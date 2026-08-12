Embedders
=========

The base class — the interface every embedder implements (including the
``nodes()`` accessor for the ``embeddings()`` row order).

The methods
-----------

Each concrete method has its own page under **Embedding methods**, carrying the
model it assumes and the encoder-decoder pair it instantiates alongside the
class reference. The names below link there.

.. autosummary::

   hypegrl.embedders.poincare_maps.PoincareMapsEmbedder
   hypegrl.embedders.poincare_embeddings.PoincareEmbeddingsEmbedder
   hypegrl.embedders.lorentz_embeddings.LorentzEmbeddingsEmbedder
   hypegrl.embedders.hydra.HydraEmbedder
   hypegrl.embedders.hydra_plus.HydraPlusEmbedder
   hypegrl.embedders.hypermap.HyperMapEmbedder
   hypegrl.embedders.dmercator.DMercatorEmbedder

Every one of them accepts ``unknown_edges`` and a ``representation=`` chart
where the method supports it, and returns its geometry through the common
interface below.

The interface
-------------

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
