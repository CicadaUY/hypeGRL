"""Abstract base class for graph generators."""
from abc import ABC, abstractmethod
import numpy as np
import networkx as nx


class GraphGenerator(ABC):
    """
    Common interface for generative models that sample graphs from embeddings.

    Parameters
    ----------
    embedder:
        A fitted :class:`~hypegrl.embedders.base.HyperbolicEmbedder`
        whose embeddings will be used to sample graphs.
    """

    def __init__(self, embedder):
        self.embedder = embedder

    @abstractmethod
    def sample(self, n_graphs: int = 1) -> list[nx.Graph]:
        """
        Sample one or more graphs from the generative model.

        Parameters
        ----------
        n_graphs:
            Number of independent graphs to sample.

        Returns
        -------
        List of NetworkX graphs.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(embedder={self.embedder!r})"
