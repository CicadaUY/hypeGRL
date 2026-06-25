# hypeGRL: Hyperbolic Graph Representation Learning

Graph Representation Learning in hyperbolic (and Euclidean) spaces,
with principled support for unknown edges and streaming graph updates.

## Overview

`hypeGRL` provides a unified framework for embedding graphs into
low-dimensional spaces — mostly hyperbolic but also some Euclidean — through the
encoder-decoder formalism. In addition to constituting a unified framework for several (previously dispersed) embedding methods, a key contribution of `hypeGRL`is the treatment of
**partially observed graphs**: rather than imputing unknown edges with
zeros (which introduces bias), the framework jointly optimizes node
embeddings and unknown adjacency entries, enforcing that the learned
representations are insensitive to the unobserved edges.

## Embedding methods 

This is a work-in-progress, but the objective is to have these methods:

| Method | Geometry | Gradient-based | Generative | Edge updates | Node updates |
|---|---|---|---|---|---|
| ASE / RDPG | Euclidean | Yes | Yes | Yes | Out-of-sample |
| Poincaré Maps | Poincaré disk | Yes | Partial | Yes | Out-of-sample |
| Poincaré Embeddings | Poincaré disk | Yes | Yes | Yes | Out-of-sample |
| Lorentz Embeddings | Hyperboloid | Yes | Yes | Yes | Out-of-sample |
| Hydra+ | Hyperboloid | No | No | Full refit | Full refit |
| d-Mercator | Spherical | Yes | Yes | Full refit | Full refit |


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

## Documentation

Full documentation at [hypegrl.readthedocs.io](https://hypegrl.readthedocs.io).

## References

- Klimovskaia et al., *Poincaré Maps for Analyzing Complex Hierarchies*, Nature Communications 2020.
- Nickel & Kiela, *Poincaré Embeddings for Learning Hierarchical Representations*, NeurIPS 2017.
- Nickel & Kiela, *Learning Continuous Hierarchies in the Lorentz Model*, ICML 2018.
- Keller et al., *Hydra: A Method for Strain-Minimizing Hyperbolic Embedding*, J. Complex Networks 2021.
- Scheinerman & Tucker, *Modeling Graphs Using Dot Product Representations*, Computational Statistics 2010.
- Fiori et al., *Gradient-Based Spectral Embeddings of Random Dot Product Graphs*, IEEE TSIPN 2024.
