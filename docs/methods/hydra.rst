Hydra and Hydra+
================
**Geometry:** Hyperboloid 

**Reference:** Keller-Ressel & Nargang., *Hydra: A Method for Strain-Minimizing Hyperbolic Embedding*, J. Complex Networks 2021

Overview
--------
Hydra assumes that pairwise dissimilarities between nodes of a graph are available in the form of matrix :math:`\mathbf{D}\in\mathbb{R}^{N\times N}`; e.g., given by shortest-path distances. The objective of the Hydra method is to find embeddings :math:`\mathbf{x}_i \in \mathbb{H}^n` (:math:`i=1,\ldots,N`) such that the resulting hyperbolic distances approximate the observed distances :math:`D_{i,j}`, by minimizing :math:`\sum_{i,j}|d_H(\mathbf{x}_i,\mathbf{x}_j)-D_{i,j}|^2`.

Directly minimizing this objective leads to a high-dimensional, non-convex optimization problem that is difficult to solve efficiently. Instead, this problem is reformulated by applying the hyperbolic cosine to the target distances, leading to the surrogate problem :math:`\min \sum_{i,j}\|-\langle\mathbf{u}_i,\mathbf{u}_j\rangle-\operatorname{cosh}(D_{i,j})\|^2`, which is efficiently solved using spectral decompositions.

The Hydra+ variant subsequently refines this solution by using it as an initialization for gradient-based optimization of the original cost function.


Encoder-decoder instantiation
------------------------------
.. math::
   \hat{A}_{ij} = -d_H(\mathbf{x}_i,\mathbf{x}_j)
   \quad
   s(\mathbf{A}) = \text{shortest path distance matrix}

.. math::
   d(s(\mathbf{A}),\hat{\mathbf{A}}) = ||\cosh{(s(\mathbf{A}))}-\cosh{(\hat{\mathbf{A}})}||_F^2

Unknown edges
-------------
...

API reference
-------------
.. autoclass:: hypegrl.embedders.hydra.HydraEmbedder
   :members:


