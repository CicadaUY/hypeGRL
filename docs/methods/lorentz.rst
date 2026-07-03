Lorentz Embeddings
==================

**Geometry:** Lorentz / hyperboloid model

**Reference:** Nickel & Kiela, *Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry*, ICML 2018

Overview
--------

Lorentz Embeddings reuse the soft-ranking objective of
:doc:`Poincaré Embeddings <poincare_embeddings>` but learn the representation in
the **Lorentz (hyperboloid) model** of hyperbolic space rather than the Poincaré
ball. The contribution is geometric and numerical, not a new loss: the Lorentz
distance

.. math::

   d_{\mathcal L}(\mathbf x, \mathbf y)
   = \operatorname{arcosh}\!\bigl(-\langle \mathbf x, \mathbf y\rangle_{\mathcal L}\bigr),
   \qquad
   \langle \mathbf x, \mathbf y\rangle_{\mathcal L}
   = -x_0 y_0 + \sum_{i\ge 1} x_i y_i,

has no :math:`(1-\|\mathbf x\|^2)` denominator, so it avoids the numerical
instability the Poincaré distance suffers near the boundary — precisely where the
leaves of a hierarchy want to live — and it admits an exact closed-form
exponential map, so optimisation follows true geodesics.

Like :doc:`Hydra <hydra>`, the method assumes pairwise **similarity** information:
more similar nodes are placed closer in hyperbolic space. For a plain graph the
similarity is the (weighted) adjacency, and *generality* emerges as the norm of
the embedding — general nodes sit near the origin, specific ones near the
boundary.

Encoder-decoder instantiation
-----------------------------

Points live on the upper sheet of the two-sheeted hyperboloid

.. math::

   \mathcal H^d
   = \{\mathbf x \in \mathbb R^{d+1} :
       \langle \mathbf x, \mathbf x\rangle_{\mathcal L} = -1,\ x_0 > 0\},

so an intrinsic dimension :math:`d` uses :math:`d+1` ambient coordinates, with
:math:`x_0 = \sqrt{1 + \|\mathbf x_{1:}\|^2}`.

The structural similarity :math:`s(\mathbf A) = \mathbf K` is the (possibly
weighted) adjacency, and the decoder is the pairwise Lorentz distance,

.. math::

   \hat A_{ij} = d_{\mathcal L}(\mathbf x_i, \mathbf x_j).

The loss is the soft-ranking negative log-likelihood (eq. 12 of the paper)

.. math::

   d\!\left(s(\mathbf A), \hat{\mathbf A}\right)
   = -\sum_{(i,j)}
       \log
       \frac{e^{-d_{\mathcal L}(\mathbf x_i, \mathbf x_j)}}
            {\sum_{z \in N(i,j)} e^{-d_{\mathcal L}(\mathbf x_i, \mathbf x_z)}},

where the neighbourhood set generalises "non-neighbours" to the **less-similar
nodes**

.. math::

   N(i,j) = \{z : K_{iz} < K_{ij}\} \cup \{j\}.

For an unweighted graph (:math:`K_{ij}\in\{0,1\}`) this collapses exactly to the
Poincaré-Embeddings negative set (the non-edges of :math:`i`); for a weighted
graph the similarity magnitude enters *only* through which nodes are negatives,
as in the paper. The set :math:`N(i,j)` is subsampled to a fixed number of
negatives per positive pair (``n_negatives``).

How the embedder works
----------------------

:class:`~hypegrl.embedders.lorentz_embeddings.LorentzEmbeddingsEmbedder`
optimises node positions directly on the hyperboloid with Riemannian Adam,
starting from the paper's near-origin initialisation (eq. 6):
:math:`\mathbf x_{1:} \sim \mathcal U(-\texttt{init\_scale}, \texttt{init\_scale})`
with :math:`x_0 = \sqrt{1 + \|\mathbf x_{1:}\|^2}`, unless ``X_init`` is supplied.

**Poincaré-ball image.** The optimisation runs on the hyperboloid, but
:meth:`~hypegrl.embedders.lorentz_embeddings.LorentzEmbeddingsEmbedder.embeddings`
returns the :math:`(N, d)` **Poincaré-ball image** via the isometry (eq. 11)

.. math::

   p(x_0, \mathbf x_{1:}) = \frac{\mathbf x_{1:}}{x_0 + 1},

exactly as the paper visualises its results — so the embedder is a drop-in for
the disk plotters and the rest of the library. The raw :math:`(N, d+1)`
hyperboloid coordinates are available through
:meth:`~hypegrl.embedders.lorentz_embeddings.LorentzEmbeddingsEmbedder.hyperboloid_embeddings`,
and ``X_init`` / warm-started
:meth:`~hypegrl.embedders.lorentz_embeddings.LorentzEmbeddingsEmbedder.update`
use the hyperboloid representation, not the ball.

When ``unknown_edges`` are supplied, fitting routes through the shared joint
optimiser :func:`~hypegrl.inference.joint_optimizer.joint_optimize` (also used by
:class:`~hypegrl.embedders.poincare_maps.PoincareMapsEmbedder` and
:class:`~hypegrl.embedders.poincare_embeddings.PoincareEmbeddingsEmbedder`);
otherwise it uses the fixed-graph optimiser
:func:`~hypegrl.inference.riemannian_optimizer.riemannian_optimize`. The decoder
returns the Lorentz distance matrix, and — since that is a *dissimilarity*, not
an edge probability — the method is ranking-only:
:meth:`~hypegrl.embedders.lorentz_embeddings.LorentzEmbeddingsEmbedder.is_generative`
is ``False``.

Numerical stability
-------------------

The exponential map scales coordinates by :math:`\cosh(\|\mathbf v\|_{\mathcal L})`,
so a single high-learning-rate step toward the boundary can overflow :math:`x_0`
and turn every distance into ``NaN`` — the well-known failure of naive
hyperboloid optimisation on leaf-heavy graphs. The embedder therefore optimises
on :class:`~hypegrl.manifolds.lorentz.StableLorentz`, a ``geoopt.Lorentz``
subclass that clamps each point's spatial norm :math:`\|\mathbf x_{1:}\|` to
``max_norm`` after every retraction (mirroring the reference implementation's
``set_dim0`` renorm). This is what lets the embedder use an aggressive learning
rate — needed for leaves to reach the near-boundary radii that encode
generality — without diverging.

``max_norm`` is more than a safety valve: it also **regularises spread**, well
below the ``float64`` overflow ceiling (:math:`\|\mathbf x_{1:}\|\approx10^{154}`).
A sweep over trees, karate and Les Misérables found a usable window of roughly
:math:`10^2`–:math:`10^4`: clamping *below* the natural radius hurts
reconstruction, while very large values (:math:`\gtrsim 10^6`) let the
optimisation run away and degrade quality. The default :math:`10^3` (Poincaré
radius :math:`\approx 0.999`) improves on the reference's :math:`10^2` on denser
graphs while staying clear of the runaway regime; it is exposed as the
``max_norm`` constructor argument because the optimum is graph-dependent.

Partially observed graphs
-------------------------

Unknown edges are a first-class feature here (the library's central capability),
but the ranking loss needs care: the membership test :math:`K_{iz} < K_{ij}` is a
non-differentiable hard threshold, so an imputed weight that entered the loss
only by deciding set membership would receive no gradient (the same obstruction
that stops :doc:`Hydra <hydra>`'s shortest-path target from routing unknown-edge
gradients). Two choices keep imputation differentiable:

- **Unknown pairs are excluded from every** :math:`N(i,j)`. Negatives are drawn
  strictly from *known* structure, so the combinatorial set membership never
  depends on the imputed weights :math:`a_\Omega`; unknown pairs appear only as
  positives.
- **Positives are weighted by** :math:`K_{ij}` **only when** :math:`\Omega` **is
  non-empty**. This routes a smooth gradient to :math:`a_\Omega` (an unknown
  edge's imputed weight scales its own positive term), mirroring the
  :math:`A_{ij}`-weighting in the Poincaré-Embeddings ranking loss.

A consequence worth noting: with no unknown edges the loss is the paper's
*unweighted* eq. 12 exactly — on both binary and weighted graphs. The
:math:`K_{ij}`-weighting is hypeGRL's device, confined to the unknown-edge regime
the paper never covers (and a no-op on binary graphs, where :math:`K_{ij}=1`).

Relation to reference implementations
-------------------------------------

The distance, objective and initialisation match the paper; the implementation
was cross-checked against the widely used theSage21 PyTorch port
(``theSage21/lorentz-embeddings``), which confirms the core design and differs in
a few deliberate ways:

- **Confirmed.** The reference's negative set (``pairwise[i,x] < min``) and its
  unweighted-positive loss validate our :math:`N(i,j)` similarity generalisation
  and the :math:`\Omega`-gated weighting.
- **Optimiser.** We optimise with geoopt ``RiemannianAdam``; the paper's
  Algorithm 1 (and the reference) use exact-geodesic Riemannian *SGD*. The
  geometry is still exact — the Lorentz ``expmap`` follows the true geodesic — but
  the update rule is Adam, matching the rest of the library. We do not replicate
  the reference's learning-rate burn-in (no measurable effect under Adam).
- **Execution model.** The reference treats the matrix as a *directed* DAG and
  samples one positive per anchor per epoch (minibatch SGD); we sum symmetrically
  over all edges full-batch, which is closer to the paper's symmetric eq. 12 and
  appropriate for the undirected graphs hypeGRL targets.
- **Defaults are tuned, not paper values.** ``lr_X=0.3`` / ``n_steps=2000`` are
  chosen so leaves reach large radius (a timid rate leaves the embedding
  under-spread and reconstruction several AUC points worse); ``max_norm=1e3`` is
  the sweep-tuned clamp discussed above (the reference uses ``1e2``).

API reference
-------------
.. autoclass:: hypegrl.embedders.lorentz_embeddings.LorentzEmbeddingsEmbedder
   :members:
