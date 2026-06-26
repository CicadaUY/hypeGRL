HyperMap
========

**Geometry:** Poincaré ball (H\ :sup:`2`/S\ :sup:`1` model, lifted to :math:`d` dimensions)

**Reference:** Papadopoulos, Aldecoa & Krioukov, *Network Geometry Inference using Common Neighbors*, Physical Review E 2015; building on the S\ :sup:`1`/H\ :sup:`2` model of Krioukov et al., *Hyperbolic Geometry of Complex Networks*, Phys. Rev. E 2010.

Overview
--------
HyperMap embeds a graph into the S\ :sup:`1`/H\ :sup:`2` geometric network model:
every node :math:`i` is given a **radial** coordinate :math:`r_i` (a "popularity"
coordinate fixed by its degree) and an **angular** coordinate :math:`\theta_i` (a
"similarity" coordinate). Two nodes connect with a Fermi--Dirac probability that
decreases with their hyperbolic distance:

.. math::
   p_{ij} = \frac{1}{1 + e^{\frac{\zeta}{2T}\left(x_{ij} - R_i\right)}},
   \qquad
   x_{ij} = d_{\mathbb{H}}(\mathbf{x}_i, \mathbf{x}_j),

where :math:`x_{ij}` is the hyperbolic distance, :math:`\zeta` is the curvature,
:math:`T\in(0,1)` is the temperature controlling clustering, and :math:`R_i` is a
per-node threshold determined analytically from the node's degree. High-degree
("popular") nodes sit near the centre at small radius; low-degree nodes sit near
the boundary. The angular layout encodes similarity, so that nodes likely to
share an edge are placed close together.

:class:`~hypegrl.embedders.hypermap.HyperMapEmbedder` follows the library's
two-stage shape — like :class:`~hypegrl.embedders.dmercator.DMercatorEmbedder`:
a faithful reimplementation of the *original* HyperMap greedy method provides the
encoder/initialisation, and a Riemannian gradient step then refines the embedding
on the Poincaré ball. The original method is two-dimensional; here it is lifted to
an arbitrary embedding dimension :math:`d`.

Encoder-decoder instantiation
-----------------------------
.. math::
   s(\mathbf{A}) = \mathbf{A} \ \text{(binary adjacency)}
   \qquad
   \hat{A}_{ij} = p_{ij}
   = \frac{1}{1 + e^{\frac{\zeta}{2T}\left(x_{ij} - R_i\right)}}

.. math::
   d\!\left(s(\mathbf{A}),\hat{\mathbf{A}}\right)
   = -\sum_{i<j} \Big[ A_{ij}\ln p_{ij} + (1-A_{ij})\ln(1-p_{ij}) \Big]

HyperMap uses the adjacency matrix directly as its structural similarity
(:math:`s(\mathbf{A}) = \mathbf{A}`), and the decoder is the Fermi--Dirac
connection probability on the exact hyperbolic distance. The threshold
:math:`R_i` is the per-node radius assigned during initialisation (the radius of
the earlier/higher-degree node of a pair); the loss is the Fermi--Dirac negative
log-likelihood (binary cross-entropy).

Initialization — the original HyperMap method
---------------------------------------------
The encoder is a faithful reimplementation of the published greedy algorithm, in
``hypegrl/embedders/_hypermap_init.py`` (mirroring how ``_dmercator_init.py``
supplies D-Mercator's warm start). Nodes are processed in **degree-descending**
order, and angles are placed one node at a time:

1. **Radii** are assigned analytically from each node's rank in the
   degree-sorted order and then frozen — they are not learned during
   initialisation.
2. **Phase 1 — common-neighbours likelihood** (high-degree nodes): each node is
   placed at the angle that maximises a Gaussian log-likelihood on the *number of
   common neighbours* it shares with already-placed nodes, with the expected
   count obtained by 48-point Gauss--Legendre integration.
3. **Phase 2 — Fermi--Dirac MLE** (remaining, lower-degree nodes): each node is
   placed at the angle that maximises the Fermi--Dirac log-likelihood over its
   observed and non-observed edges to already-placed nodes.
4. **Correction sweeps** at degree thresholds :math:`\{10, 20, 40, 60\}`: the
   Phase-2 angular placement is re-run for all Phase-2 nodes to refine the layout.

The result lives in H\ :sup:`2` polar coordinates :math:`(r_i, \theta_i)` and is
converted to the Poincaré ball by the caller — directly for :math:`d=2`
(:func:`~hypegrl.manifolds.poincare.polar_to_poincare`), or for :math:`d>2` by
embedding the 2-D layout in the first two dimensions with the extra
hyperspherical angles set to zero
(:func:`~hypegrl.manifolds.poincare.hyperspherical_to_poincare`); the gradient
step then moves nodes freely in the full :math:`d`-dimensional space.

Gradient refinement
-------------------
:meth:`~hypegrl.embedders.hypermap.HyperMapEmbedder.fit` wraps the warm start as a
``geoopt.ManifoldParameter`` on the Poincaré ball and refines it with Riemannian
Adam, minimising the Fermi--Dirac NLL above. The global parameters
(:math:`\gamma`, :math:`T`, :math:`\zeta`, and the per-node thresholds
:math:`R_i`) are held fixed, as in the original method; only the per-node
positions move.

By default (``fix_radii=True``) the radial coordinates are held at their
analytically assigned values — only the angles are refined, matching the original
HyperMap. A radius-preserving retraction snaps each row's norm back to its target
after every step. Setting ``fix_radii=False`` frees the radii, letting the
optimiser move nodes radially as well. Setting ``n_steps=0`` returns the pure
greedy initialisation without refinement.

Implementation notes
--------------------
**Degree-based node ordering.** The greedy initialisation sorts nodes by degree,
and the row order of
:meth:`~hypegrl.embedders.hypermap.HyperMapEmbedder.embeddings` follows that order
— exposed as
:attr:`~hypegrl.embedders.hypermap.HyperMapEmbedder.nodes_sorted`, *not* the
original NetworkX node order. Always map rows back through ``nodes_sorted`` before
associating an embedding with a node.

**Initialisation is always 2-D, then lifted.** The greedy phases run in
H\ :sup:`2` polar coordinates regardless of the target dimension; for :math:`d>2`
the extra hyperspherical angles start at zero and only move during the gradient
refinement. The warm-start path is reused on subsequent ``fit`` calls when
``X_init`` is supplied — ``test_hypermap_x_init_equivalent_to_default`` checks
that passing the cached warm start reproduces the default run.

**Power-law exponent.** The model needs the degree-distribution exponent
:math:`\gamma`; if it is not supplied it is estimated from the graph (via the
maximum-likelihood estimator of Clauset et al., ``k_min=3``). The temperature
:math:`T` and curvature :math:`\zeta` are fixed hyperparameters.

**Unknown edges.** Supported. Edges passed via ``fit(..., unknown_edges=...)`` are
treated as free variables and jointly optimised through the shared joint optimiser
:func:`~hypegrl.inference.joint_optimizer.joint_optimize` (also used by
:class:`~hypegrl.embedders.poincare_maps.PoincareMapsEmbedder` and
:class:`~hypegrl.embedders.poincare_embeddings.PoincareEmbeddingsEmbedder`), with
the Fermi--Dirac NLL as the loss.

API reference
-------------
.. autoclass:: hypegrl.embedders.hypermap.HyperMapEmbedder
   :members:
