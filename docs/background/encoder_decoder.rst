Encoder-Decoder framework
=========================

Consider a graph :math:`G=(V,E)`. Graph Representation Learning (GRL) deals with the problem of obtaining a low-dimensional representation :math:`\mathbf{x}_i` of each node in :math:`V=\{1,\ldots,N\}`. Although typically :math:`\mathbf{x}_i\in\mathbb{R}^d`, other geometries besides Euclidean are possible and we will assume that :math:`\mathbf{x}_i` refers to the corresponding coordinates in whatever space the embeddings live in. In any case, the problem of finding the set of vectors :math:`\mathbf{x}_1,\ldots,\mathbf{x}_N` is virtually always expressed through the so-called encoder-decoder framework [#chami]_:

.. math::

    \underset{\boldsymbol{\theta}_E,\boldsymbol{\theta}_D\in \boldsymbol{\Theta}}{\min}\,\mathcal{L}(\mathbf{A},\boldsymbol{\theta}_E,\boldsymbol{\theta}_D)
    =
    \underset{\boldsymbol{\theta}_E,\boldsymbol{\theta}_D\in \boldsymbol{\Theta}}{\min}\,
    d\left(s(\mathbf{A}),
    \text{Dec}\left(\text{Enc}\left(\mathbf{A},\boldsymbol{\theta}_E\right),\boldsymbol{\theta}_D\right)\right)

where

- :math:`\text{Enc}\left(\mathbf{A},\boldsymbol{\theta}_E\right)` is a function, parametric in :math:`\boldsymbol{\theta}_E`, that maps the adjacency matrix :math:`\mathbf{A}` to the set of vectors :math:`\{\mathbf{x}_i\}` (which we will stack in a matrix :math:`\mathbf{X}\in\mathbb{R}^{N\times d}`). In the case of shallow embeddings, such as :doc:`ASE <../methods/ase>` or :doc:`Poincaré Maps <../methods/poincare_maps>`, the output is simply :math:`\mathbf{X}=\boldsymbol{\theta}_E`, but this framework is general enough to accommodate parametric models such as GNNs.

- :math:`\text{Dec}\left(\mathbf{X},\boldsymbol{\theta}_D\right)` is another function that takes a set of embeddings and produces a (dis)similarity matrix :math:`\hat{\mathbf{A}}` between the node's embeddings. This is where the chosen underlying geometry plays a significant role, as these (dis)similarities are typically distance-based.

- :math:`s(\mathbf{A})` takes the adjacency matrix and produces a structural (dis)similarity matrix between nodes in the original graph (e.g., shortest path distance between each pair of nodes).

- Finally, :math:`d(s(\mathbf{A}),\hat{\mathbf{A}})` measures the distance between both (dis)similarity matrices.

Once the optimal :math:`\boldsymbol{\theta}_E^*` and :math:`\boldsymbol{\theta}_D^*` are found, the embeddings are the output of the encoder :math:`\mathbf{X}^*=\text{Enc}\left(\mathbf{A},\boldsymbol{\theta}_E^*\right)`.

.. [#chami] Chami et al. *Machine Learning on Graphs: A Model and Comprehensive Taxonomy*, JMLR 2022.
