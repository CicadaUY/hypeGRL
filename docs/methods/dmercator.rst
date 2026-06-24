Mercator and D-Mercator
=======================

**Geometry:** Spherical similarity (S\ :sup:`D`) + hyperbolic popularity (Poincaré ball)

**Reference:** Jankowski, Allard, Boguñá & Serrano, *The D-Mercator method for the multidimensional hyperbolic embedding of real networks*, Nature Communications 2023; García-Pérez et al., *Mercator*, New J. Phys. 2019.

Overview
--------
D-Mercator embeds a graph into the :math:`\mathbb{S}^D` geometric network model:
every node :math:`i` gets a hidden degree :math:`\kappa_i` (a "popularity"
coordinate) and an angular position :math:`\mathbf{v}_i \in \mathbb{S}^D` (a
"similarity" coordinate). Two nodes connect with a Fermi--Dirac probability that
decreases with their angular separation, scaled by their popularities:

.. math::
   \Delta\theta_{ij} = \arccos(\mathbf{v}_i\cdot\mathbf{v}_j),
   \qquad
   \chi_{ij} = \frac{R\,\Delta\theta_{ij}}{(\mu\,\kappa_i\kappa_j)^{1/D}},
   \qquad
   p_{ij} = \frac{1}{1+\chi_{ij}^{\beta}} .

Here :math:`R` (sphere radius) and :math:`\mu` are fixed by the network size and
mean degree, and :math:`\beta>D` (the inverse temperature) controls clustering.
The model simultaneously reproduces strong clustering, the small-world property,
and heterogeneous degrees. The single angular dimension of the original Mercator
(:math:`D=1`, a circle) is generalised to :math:`D` similarity dimensions.

The :math:`\mathbb{S}^D` model maps to hyperbolic space :math:`\mathbb{H}^{D+1}`
by a radial transform (Eq. 7): a node with hidden degree :math:`\kappa_i` is
placed at radius

.. math::
   r_i = \hat{R} - \frac{2}{D}\ln\frac{\kappa_i}{\kappa_{\min}},
   \qquad
   \hat{R} = 2\ln\frac{2R}{(\mu\,\kappa_{\min}^2)^{1/D}},

so popular (high-:math:`\kappa`) nodes sit near the centre and peripheral nodes
near the boundary. In this picture the connection probability becomes a
Fermi--Dirac on the *hyperbolic distance*,
:math:`p_{ij} = 1/(1+e^{(\beta/2)(d_H(\mathbf{x}_i,\mathbf{x}_j)-\hat{R})})`,
which coincides with the :math:`\mathbb{S}^D` form above because
:math:`d_H \approx r_i + r_j + 2\ln(\Delta\theta_{ij}/2)`.

:class:`~hypegrl.embedders.dmercator.DMercatorEmbedder` follows the library's
two-stage shape: a faithful reimplementation of the *original* D-Mercator method
provides the encoder/initialisation, and a Riemannian gradient step then refines
the embedding directly in hyperbolic space.

Encoder-decoder instantiation
------------------------------
.. math::
   s(\mathbf{A}) = \mathbf{A} \ \text{(binary adjacency)}
   \qquad
   \hat{A}_{ij} = p_{ij} = \frac{1}{1+e^{(\beta/2)\left(d_H(\mathbf{x}_i,\mathbf{x}_j)-\hat{R}\right)}}

.. math::
   d\!\left(s(\mathbf{A}),\hat{\mathbf{A}}\right)
   = -\sum_{i<j} \Big[ A_{ij}\ln p_{ij} + (1-A_{ij})\ln(1-p_{ij}) \Big]

D-Mercator is a **binary** model: edge weights are ignored when forming
:math:`s(\mathbf{A})`. The loss is the Fermi--Dirac negative log-likelihood
(binary cross-entropy) on the *exact* hyperbolic distances.

Initialization — the original D-Mercator method
-----------------------------------------------
The encoder is a faithful reimplementation of the published method, in
``hypegrl/embedders/_dmercator_init.py``. It mirrors how
``_hypermap_init.py`` supplies HyperMap's greedy initialisation, and proceeds in
four stages:

1. **Hidden degrees and inverse temperature** (:math:`\kappa`, :math:`\beta`).
   :math:`\kappa` is tuned per degree class so each node's *expected* model
   degree matches its observed degree; :math:`\beta` is found by bracket-then-
   bisect so the model's expected mean local clustering matches the empirical
   one (a Monte-Carlo estimate). The two are coupled, so the :math:`\kappa`
   loop is re-run for every :math:`\beta`.
2. **Angular positions** via :math:`\mathbb{S}^D`-corrected Laplacian Eigenmaps:
   a weighted-Laplacian eigenproblem whose target distances are the model's
   expected angular distances for connected pairs.
3. **Likelihood maximisation**: nodes are visited in onion-decomposition order
   and each is moved to the most likely of several candidate positions sampled
   around its hidden-degree-weighted neighbour mean.
4. **Final hidden-degree readjustment** using the now-known positions.

The exact algorithm — including the numerical-integration details and the
pitfalls reconciled against the upstream C++ code — is recorded verbatim in
:doc:`dmercator_pseudocode`. That page is the specification the init module
tracks; its section numbers are cited from the code comments.

Hyperbolic refinement
----------------------
Stage 6 of the init (the radial map) turns the warm start into a hyperbolic
embedding :math:`\mathbf{x}_i = \tanh(r_i/2)\,\mathbf{v}_i` on the Poincaré ball.
:meth:`~hypegrl.embedders.dmercator.DMercatorEmbedder.fit` then refines it with
Riemannian Adam, minimising the Fermi--Dirac NLL above. The global parameters
:math:`\beta` and :math:`\hat{R}` are held fixed (as HyperMap fixes its radius
cutoff); only the per-node positions move. Because a position carries *both* an
angle and a radius, the hidden degree :math:`\kappa_i` — recovered from the
refined radius via the inverse of Eq. 7 — is learned jointly with the angle,
rather than frozen at its initial value.

Setting ``n_steps=0`` returns the pure original-method initialisation (a useful
baseline, and the hook used by the ``X_init`` equivalence tests).

Implementation notes
--------------------
**Why the Poincaré ball and not the hyperboloid.** Both are exact charts of
:math:`\mathbb{H}^{D+1}`, and ``geoopt.PoincareBall.dist`` is the exact
hyperbolic distance — the choice is purely numerical. Low-degree leaf nodes
belong at large radius (:math:`r \approx \hat{R}`, which reaches 16–20 even on
small graphs). On the **hyperboloid** that means a timelike coordinate
:math:`\cosh(r)\approx 10^5`–:math:`10^7`; an optimisation step that pushes two
such nodes together loses precision in the Lorentzian inner product and sends
the gradient to ``NaN`` — fatal in the :math:`D=1`/Mercator, leaf-heavy regime
(it failed on most seeds of a Barabási–Albert tree). The **Poincaré ball** keeps
coordinates in :math:`(-1,1)` and stays well-conditioned at those radii, so it
is the manifold used for the refinement. The regression test
``test_dmercator_robust_on_leaf_heavy_graph_d1`` guards this.

**The D=1 reduction to Mercator.** Setting ``d=2`` gives similarity dimension
:math:`D=1` — the original Mercator model, where :math:`\mathbb{S}^D` collapses
to a circle. The cleanest closed-form sanity check is the sphere radius, which
must satisfy :math:`R(N,1)=N/(2\pi)`. This is verified in the tests and is the
quickest way to localise a bug in the :math:`\Gamma`/:math:`\pi` global-constant
machinery (see §7 of :doc:`dmercator_pseudocode`).

**Unknown edges.** Not yet supported: ``fit(..., unknown_edges=...)`` warns and
zero-imputes. Joint optimisation of missing entries (as the gradient-based
embedders do via the joint optimiser) is future work.

.. toctree::
   :hidden:

   dmercator_pseudocode

API reference
-------------
.. autoclass:: hypegrl.embedders.dmercator.DMercatorEmbedder
   :members:
