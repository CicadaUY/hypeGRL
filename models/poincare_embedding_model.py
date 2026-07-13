import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from gensim.models.poincare import PoincareModel
from tqdm import trange

from models.base_hyperbolic_model import BaseHyperbolicModel
from utils.geometric_conversions import poincare_to_hyperboloid

# Enable logging for gensim
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

logger = logging.getLogger(__name__)
logging.getLogger("gensim").setLevel(logging.WARNING)


class PoincareEmbeddingModel(BaseHyperbolicModel):
    def __init__(self, config: Dict):
        self.dim = config.get("dim", 10)
        self.epochs = config.get("epochs", 50)
        self.batch_size = config.get("batch_size", 10)
        self.lr = config.get("lr", 0.3)
        self.negs = config.get("negs", 50)
        self.burnin = config.get("burnin", 20)
        self.workers = config.get("train_threads", 1)

    @property
    def native_space(self) -> str:
        """Get the native embedding space for this model."""
        return "poincare"

    def train(
        self,
        edge_list: Optional[List[tuple]] = None,
        adjacency_matrix: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
        model_path: str = "saved_models/model.bin",
    ):
        if edge_list is None and adjacency_matrix is None:
            raise ValueError("You must provide either edge_list or adjacency_matrix.")

        if adjacency_matrix is not None:
            edge_list = []
            for i in range(adjacency_matrix.shape[0]):
                for j in range(adjacency_matrix.shape[1]):
                    if adjacency_matrix[i, j] == 1:
                        edge_list.append((str(i), str(j)))

        # Train model
        self.model = PoincareModel(
            train_data=edge_list, size=self.dim, burn_in=self.burnin, alpha=self.lr, negative=self.negs, workers=self.workers
        )
        for epoch in trange(self.epochs, desc="Training PoincareModel"):
            self.model.train(epochs=1, batch_size=self.batch_size)
        logger.info("Training completed.")

        self.model.save(model_path)
        logger.info(f"Model saved in {model_path}.")

    def get_embedding(self, node_id: str, model_path: Optional[str] = None) -> np.ndarray:
        if model_path:
            self.model = PoincareModel.load(model_path)

        if node_id not in self.model.kv:
            raise KeyError(f"Node id '{node_id}' does not exist in the model.")
        return self.model.kv.get_vector(node_id)

    def get_all_embeddings(self, model_path: Optional[str] = None) -> np.ndarray:
        if model_path:
            self.model = PoincareModel.load(model_path)
        vectors = []
        node_ids = sorted(self.model.kv.index_to_key, key=lambda k: int(k))
        for k in node_ids:
            vectors.append(self.model.kv.get_vector(k))
        return np.array(vectors)

    def most_similar(self, node_id: str, topn: int = 5, model_path: Optional[str] = None) -> List[Tuple[str, float]]:
        if model_path:
            self.model = PoincareModel.load(model_path)
        if node_id not in self.model.kv:
            raise KeyError(f"Node id '{node_id}' does not exist in the model.")
        return self.model.kv.most_similar(node_id, topn=topn)

    def to_hyperboloid(self, model_path: Optional[str] = None) -> np.ndarray:
        """Convert Poincaré embeddings to hyperboloid coordinates."""
        poincare_embeddings = self.get_all_embeddings(model_path)
        return poincare_to_hyperboloid(poincare_embeddings)

    def to_poincare(self, model_path: Optional[str] = None) -> np.ndarray:
        """Return embeddings in Poincaré coordinates (already in Poincaré space)."""
        return self.get_all_embeddings(model_path)
