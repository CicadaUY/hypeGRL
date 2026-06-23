**A few things that need to be done**
 - Some methods are still missing: Poincaré Embeddings, Lorentz Embeddings and D-Mercator.
 - Streaming (node or link deletion/addition) is a relatively straightforward thing to implement, but yet not implemented. In particular, how should we initialize the new nodes' embedddings is not so evident.
 - The unknown edges framework is already implemented, but it's terribly slow. Alternative training is needed. 
 - The tutorial part of the readthedocs site needs some further work.
 - All optimization should be performed in the Hyperboloid model by default.
 - Other visualizations should be available, not only the hyperbolic disk.
 - Include methods for random sampling on the hyperbolic space and different ways of generating graphs from these samples.
 - Isometries operations, such as translations (most importantly: ``centering'' around a node)


