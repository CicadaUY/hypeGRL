"""
hypeGRL
==============
Graph Representation Learning in hyperbolic (and some Euclidean) spaces,
with principled support for unknown edges and streaming graph updates.
"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("hypegrl")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"

from hypegrl.embedders.base import HyperbolicEmbedder
from hypegrl.embedders.poincare_maps import PoincareMapsEmbedder
from hypegrl.generation.base import GraphGenerator
from hypegrl.visualization.disk import plot_poincare_graph

__all__ = [
    "HyperbolicEmbedder",
    "PoincareMapsEmbedder",
    "GraphGenerator",
    "plot_poincare_graph",
    "__version__",
]
