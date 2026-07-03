**A few things that need to be done**
 - Some methods are still missing: Poincaré Embeddings, Lorentz Embeddings and D-Mercator.
 - Streaming (node or link deletion/addition) is a relatively straightforward thing to implement, but yet not implemented. In particular, how should we initialize the new nodes' embedddings is not so evident.
 - The unknown edges framework is already implemented, but it's terribly slow. Alternative training is needed. 
 - The tutorial part of the readthedocs site needs some further work.
 - All optimization should be performed in the Hyperboloid model by default.
 - Other visualizations should be available, not only the hyperbolic disk.
 - Include methods for random sampling on the hyperbolic space and different ways of generating graphs from these samples.
 - Isometries operations, such as translations (most importantly: ``centering'' around a node)
 - Implement the generative methods for those models that do support it (e.g. D-Mercator). 
 - Finish the investigation into why HyperMap embeds a balanced tree "nicely" while D-Mercator does not, and whether the Fermi-Dirac latent-geometry model can *generatively recover* a tree at all. Findings so far: D-Mercator's clustering-matching drives β to its floor (β = D + 0.01) because a tree's c̄ = 0 is unreachable, corrupting κ/radii (degree–radius corr −0.25 vs HyperMap's −0.86); forcing a large β does NOT recover the tree (AUC 0.71→0.81, generated clustering ≈ 1.0), and sharpening the threshold on HyperMap's own coordinates *raises* generated clustering (0.24 → 0.40), moving away from the tree's zero. Tentative conclusion: a tree's zero clustering fights the model's quasi-transitivity (proximity ⇒ triangles), so large β cannot recover it; HyperMap merely wins on *layout/edge-ranking* (AUC ~0.99, clean radial hierarchy), not generative fidelity. Still to nail: the MAP/threshold-graph reconstruction (connect iff d_H < R̂, β-independent) — precision/recall and clustering vs the tree — to confirm the tree isn't threshold-recoverable from these coordinates.


