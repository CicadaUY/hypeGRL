# hypeGRL: Hyperbolic Graph Representation Learning

Graph Representation Learning in hyperbolic (and Euclidean) spaces,
with principled support for unknown (partially observed) edges, and
planned support for streaming graph updates.

## Overview

`hypeGRL` provides a unified framework for embedding graphs into
low-dimensional spaces — mostly hyperbolic but also some Euclidean — through the
encoder-decoder formalism. In addition to constituting a unified framework for several (previously dispersed) embedding methods, a key contribution of `hypeGRL`is the treatment of
**partially observed graphs**: rather than imputing unknown edges with
zeros (which introduces bias), the framework jointly optimizes node
embeddings and unknown adjacency entries, enforcing that the learned
representations are insensitive to the unobserved edges.

## Embedding methods

`hypeGRL` is a work in progress. Implemented methods (✅) and planned ones:

| Method | Geometry | Gradient-based | Generative | Unknown edges | Updates | Status |
|---|---|---|---|---|---|---|
| Poincaré Maps | Poincaré disk | Yes | No | Yes | warm-start refit | ✅ |
| Poincaré Embeddings | Poincaré ball | Yes | FD loss only | Yes | warm-start refit | ✅ |
| Lorentz Embeddings | Hyperboloid | Yes | No | Yes | warm-start refit | ✅ |
| HyperMap | Poincaré ball | Yes | Yes | Yes | warm-start refit | ✅ |
| Hydra | Poincaré disk | No (closed-form) | No | No (zero-impute) | refit | ✅ |
| Hydra+ | Poincaré disk | Yes | No | No (zero-impute) | refit | ✅ |
| D-Mercator | S^D + Poincaré ball | Yes | Yes | Planned | refit | ✅ |
| ASE / RDPG | Euclidean | Yes | Yes | — | — | Planned |

*Updates*: methods currently re-fit on graph changes (warm-started where
gradient-based). True incremental updates — Woodbury forest-matrix updates and
out-of-sample node extension — are planned.


## Installation

```bash
pip install git+https://github.com/cicadaUY/hypeGRL
```

You may also clone the repo and create your own hyperbolic method. The library is designed to be modular an easily extendible. 


For development:

```bash
git clone https://github.com/cicadaUY/hypeGRL
cd hypeGRL
pip install -e ".[dev]"
```

This way, instead of copying files to your site-packages, pip creates a link to your local folder. Any code you change in that folder is instantly used by Python.

## Quick start

```python
import networkx as nx
from hypegrl.embedders.poincare_maps import PoincareMapsEmbedder

G = nx.karate_club_graph()
unknown_edges = list(G.edges())[:5]

embedder = PoincareMapsEmbedder(d=2, n_steps=300)
embedder.fit(G, unknown_edges=unknown_edges)

X = embedder.embeddings()
print(X.shape)  # (34, 2)
```

Or, for example, Poincaré Embeddings (Nickel & Kiela) with the ranking loss
and negative sampling:

```python
from hypegrl.embedders.poincare_embeddings import PoincareEmbeddingsEmbedder

embedder = PoincareEmbeddingsEmbedder(d=2, n_steps=300)  # loss="ranking" by default
embedder.fit(G)

X = embedder.embeddings()
print(X.shape)  # (34, 2)
```

## Evaluation

`hypegrl.evaluation` provides dataset-agnostic tools for benchmarking any
embedder under a common protocol — link prediction and distance-based node
classification — reusing scikit-learn for the underlying metrics.

```python
from hypegrl.evaluation import (
    link_prediction_split, training_graph,
    pairwise_distance_matrix, candidate_scores, f1_at_k,
)

# Hide 10% of edges, embed the remaining graph, rank the held-out edges.
split = link_prediction_split(G, q=0.9, seed=0)
emb = PoincareMapsEmbedder(d=2).fit(training_graph(G, split))
D = pairwise_distance_matrix(emb.embeddings())               # geodesic distances
scores, is_positive = candidate_scores(split, D, nodes=emb.nodes())
print(f1_at_k(scores, is_positive, higher_is_link=False))    # rank by distance
```

Node classification with a hyperbolic-distance KNN:

```python
from hypegrl.evaluation import hyperbolic_knn_classification

result = hyperbolic_knn_classification(emb.embeddings(), labels, k=5, seed=0)
print(result["accuracy"], result["f1"])
```

The scripts under `experiments/` reproduce our benchmark tables on top of these
utilities. They also serve as a worked, advanced-usage example of the library
beyond the tutorials — see [`experiments/README.md`](experiments/README.md).

## Documentation

Full documentation at [hypegrl.readthedocs.io](https://hypegrl.readthedocs.io).

## References

- Klimovskaia et al., *Poincaré Maps for Analyzing Complex Hierarchies*, Nature Communications 2020.
- Nickel & Kiela, *Poincaré Embeddings for Learning Hierarchical Representations*, NeurIPS 2017.
- Nickel & Kiela, *Learning Continuous Hierarchies in the Lorentz Model*, ICML 2018.
- Papadopoulos, Aldecoa & Krioukov, *Network Geometry Inference using Common Neighbors* (HyperMap), Physical Review E 2015.
- Keller et al., *Hydra: A Method for Strain-Minimizing Hyperbolic Embedding*, J. Complex Networks 2021.
- García-Pérez et al., *Mercator: uncovering faithful hyperbolic embeddings of complex networks*, New J. Phys. 2019.
- Jankowski, Allard, Boguñá & Serrano, *The D-Mercator method for the multidimensional hyperbolic embedding of real networks*, Nature Communications 2023.
- Scheinerman & Tucker, *Modeling Graphs Using Dot Product Representations*, Computational Statistics 2010.
- Fiori et al., *Gradient-Based Spectral Embeddings of Random Dot Product Graphs*, IEEE TSIPN 2024.
