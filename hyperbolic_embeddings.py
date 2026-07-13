import logging
from typing import Dict, List, Optional, Tuple, Type

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from models.base_hyperbolic_model import BaseHyperbolicModel
from models.dmercator_embedding_model import DMercatorModel
from models.hydra_embedding_model import HydraModel
from models.hydra_plus_embedding_model import HydraPlusModel
from models.hypermap_embedding_model import HypermapEmbeddingModel
from models.lorentz_embedding_model import LorentzEmbeddingsModel
from models.poincare_embedding_model import PoincareEmbeddingModel
from models.poincare_maps_model import PoincareMapsModel
from utils.geometric_conversions import (
    HyperbolicConversions,
    compute_distances,
    convert_coordinates,
    validate_embeddings,
)

# Configure logging
logger = logging.getLogger(__name__)


class HyperbolicEmbeddings:

    MODEL_REGISTRY: Dict[str, Type["BaseHyperbolicModel"]] = {
        "poincare_embeddings": PoincareEmbeddingModel,
        "lorentz": LorentzEmbeddingsModel,
        "poincare_maps": PoincareMapsModel,
        "dmercator": DMercatorModel,
        "hydra": HydraModel,
        "hypermap": HypermapEmbeddingModel,
        "hydra_plus": HydraPlusModel,
    }

    SUPPORTED_SPACES = HyperbolicConversions.SUPPORTED_SPACES

    def __init__(self, embedding_type: str, config: Dict):
        """
        Initialize HyperbolicEmbeddings with specified model type and configuration.

        Parameters:
        - embedding_type: Type of hyperbolic embedding model to use
        - config: Configuration dictionary for the model
        """
        logger.info(f"Initializing HyperbolicEmbeddings with type: {embedding_type}")

        if not isinstance(embedding_type, str):
            raise TypeError("embedding_type must be a string")
        if not isinstance(config, dict):
            raise TypeError("config must be a dictionary")

        self.embedding_type = embedding_type
        self.model = self._load_model(embedding_type, config)
        logger.info(f"Successfully initialized {embedding_type} model")

    def _load_model(self, embedding_type: str, config: Dict) -> "BaseHyperbolicModel":
        """Load the specified hyperbolic embedding model."""
        logger.debug(f"Loading model: {embedding_type}")
        model_class = self.MODEL_REGISTRY.get(embedding_type)
        if not model_class:
            valid_keys = "', '".join(self.MODEL_REGISTRY.keys())
            logger.error(f"Unsupported model type: '{embedding_type}'. Available: {valid_keys}")
            raise ValueError(f"Unsupported model type: '{embedding_type}'. " f"Choose from: '{valid_keys}'")
        logger.debug(f"Model class loaded: {model_class.__name__}")
        return model_class(config)

    def train(
        self,
        edge_list: Optional[List[tuple]] = None,
        adjacency_matrix: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
        model_path: str = "saved_models/model.bin",
    ):
        """Train the hyperbolic embedding model."""
        logger.info(f"Starting training for {self.embedding_type} model")
        logger.debug(f"Model path: {model_path}")

        if edge_list is not None:
            logger.info(f"Training with edge list: {len(edge_list)} edges")
        elif adjacency_matrix is not None:
            logger.info(f"Training with adjacency matrix: {adjacency_matrix.shape}")
        else:
            logger.error("No training data provided")
            raise ValueError("Either edge_list or adjacency_matrix must be provided")

        self.model.train(edge_list, adjacency_matrix, features, model_path)
        logger.info(f"Training completed for {self.embedding_type} model")

    def get_node_embedding(self, node_id: str, model_path: Optional[str] = None):
        """Get embedding for a specific node."""
        logger.debug(f"Getting embedding for node: {node_id}")
        if not isinstance(node_id, str):
            logger.error(f"Invalid node_id type: {type(node_id)}")
            raise TypeError("node_id must be a string")
        embedding = self.model.get_embedding(node_id, model_path)
        logger.debug(f"Retrieved embedding for node {node_id}: shape {embedding.shape}")
        return embedding

    def get_all_embeddings(self, model_path: Optional[str] = None):
        """Get embeddings for all nodes."""
        logger.info("Retrieving all embeddings")
        embeddings = self.model.get_all_embeddings(model_path)
        logger.info(f"Retrieved embeddings: shape {embeddings.shape}")
        return embeddings

    def convert_coordinates(self, embeddings: np.ndarray, from_space: str, to_space: str) -> np.ndarray:
        """
        Convert embeddings between different hyperbolic spaces.

        Parameters:
        - embeddings: Input embeddings array
        - from_space: Source space ("poincare", "hyperboloid", "klein", "hemisphere", "half_plane", "spherical")
        - to_space: Target space ("poincare", "hyperboloid", "klein", "hemisphere", "half_plane", "spherical")

        Returns:
        - Converted embeddings
        """
        logger.info(f"Converting coordinates: {from_space} → {to_space}")
        logger.debug(f"Input embeddings shape: {embeddings.shape}")

        converted_embeddings = convert_coordinates(embeddings, from_space, to_space)

        logger.debug(f"Output embeddings shape: {converted_embeddings.shape}")
        return converted_embeddings

    def validate_embeddings(self, embeddings: np.ndarray, space: str) -> bool:
        """
        Validate that embeddings satisfy the constraints of the given space.

        Parameters:
        - embeddings: Embeddings to validate
        - space: Space to validate against

        Returns:
        - True if valid, raises ValueError if invalid
        """
        logger.debug(f"Validating embeddings for {space} space (shape: {embeddings.shape})")

        try:
            result = validate_embeddings(embeddings, space)
            logger.info(f"Validation passed for {space} space")
            return result
        except ValueError as e:
            logger.error(f"Validation failed for {space} space: {e}")

    def plot_geodesic_arc(self, p1, p2, ax, linewidth: float = 0.5, alpha: float = 0.6, color: str = "gray", linestyle: str = "--"):
        """
        Plot a hyperbolic geodesic between p1 and p2 on the Poincaré disk.
        """
        z1, z2 = complex(*p1), complex(*p2)

        # Check if it's a straight line through origin
        if np.isclose(np.cross(p1, p2), 0):
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=linewidth, alpha=alpha, linestyle=linestyle, color=color)
            return

        denom = np.imag(np.conj(z1) * z2 - z1 * np.conj(z2))
        if np.isclose(denom, 0):
            return

        center = (z1 * abs(z2) ** 2 - z2 * abs(z1) ** 2 + z1 - z2) / denom * 1j
        center = complex(center)
        radius = abs(z1 - center)

        theta1 = np.angle(z1 - center)
        theta2 = np.angle(z2 - center)

        # Ensure we go the shorter arc
        if theta2 < theta1:
            theta1, theta2 = theta2, theta1
        if theta2 - theta1 > np.pi:
            theta1, theta2 = theta2, theta1 + 2 * np.pi

        angles = np.linspace(theta1, theta2, 100)
        arc = center + radius * np.exp(1j * angles)
        ax.plot(arc.real, arc.imag, linewidth=linewidth, alpha=alpha, linestyle=linestyle, color=color)

    def plot_embeddings(
        self,
        embeddings: np.ndarray,
        embedding_space: str = "hyperboloid",
        output_space: str = "poincare",
        labels: Optional[List[str]] = None,
        edge_list: Optional[List[Tuple]] = None,
        save_path: Optional[str] = None,
        plot_title: Optional[str] = None,
        plot_geodesic: bool = True,
        figsize: Tuple[int, int] = (10, 10),
        point_size: int = 100,
        show_node_labels: bool = True,
        node_label_size: int = 8,
        edge_alpha: float = 0.5,
        edge_width: float = 0.5,
        colormap: str = "tab10",
    ):
        """
        Plot embeddings in Poincaré or spherical spaces.

        Parameters:
        - embeddings: Numpy array of embeddings (N x D)
        - embedding_space: Space of the embeddings return by the method
        - output_space: Space to plot in ("poincare" or "spherical")
        - labels: Optional list of labels for coloring points
        - edge_list: Optional list of edges to draw connections
        - save_path: Optional path to save the plot
        - plot_geodesic: Whether to plot geodesic arcs (only for Poincaré space)
        - figsize: Figure size as (width, height)
        - point_size: Size of scatter points
        - show_node_labels: Whether to show node indices as text
        - node_label_size: Font size for node labels
        - edge_alpha: Transparency of edges
        - edge_width: Width of edge lines
        - colormap: Matplotlib colormap name
        """
        # Input validation
        if embedding_space.lower() not in self.SUPPORTED_SPACES:
            logger.error(f"Invalid input_space: {embedding_space}. Must be one of {self.SUPPORTED_SPACES}")
            raise ValueError(f"input_space must be one of {self.SUPPORTED_SPACES}")

        # Only allow poincare and spherical as output spaces
        allowed_output_spaces = ["poincare", "spherical"]
        if output_space.lower() not in allowed_output_spaces:
            logger.error(f"Invalid output_space: {output_space}. Must be one of {allowed_output_spaces}")
            raise ValueError(f"output_space must be one of {allowed_output_spaces}")

        if embeddings.shape[1] > 2:
            logger.info(f"Embedding dimension is {embeddings.shape[1]}; plotting only the first 2 dimensions.")

        # Convert coordinates
        if embedding_space != output_space:
            logger.info(f"Converting embeddings for plotting: {embedding_space} → {output_space}")
            plot_embeddings = self.convert_coordinates(embeddings, embedding_space, output_space)
        else:
            plot_embeddings = embeddings

        # Validate embeddings
        self.validate_embeddings(plot_embeddings, output_space)

        x, y = plot_embeddings[:, 0], plot_embeddings[:, 1]

        # Create figure
        logger.debug(f"Creating plot with figsize: {figsize}")
        fig, ax = plt.subplots(figsize=figsize)
        if plot_title:
            ax.set_title(plot_title)
        else:
            ax.set_title(f"Hyperbolic Embeddings: {embedding_space.capitalize()} → {output_space.capitalize()}")

        if output_space.lower() == "poincare":
            # Plot edges
            if edge_list:
                logger.debug(f"Plotting {len(edge_list)} edges")
                for u, v in edge_list:
                    if u < len(x) and v < len(x):  # Check bounds
                        p1 = (x[u], y[u])
                        p2 = (x[v], y[v])
                        if plot_geodesic and output_space.lower() == "poincare":
                            self.plot_geodesic_arc(p1, p2, ax)
                        else:
                            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="gray", linewidth=edge_width, alpha=edge_alpha, zorder=1)

            # Plot points with labels
            if labels:
                labels = np.array(labels)
                unique_labels = sorted(set(labels))
                cmap = cm.get_cmap(colormap, len(unique_labels))
                label_to_color = {label: cmap(i) for i, label in enumerate(unique_labels)}

                for label in unique_labels:
                    indices = np.where(labels == label)[0]
                    ax.scatter(x[indices], y[indices], s=point_size, edgecolor="black", color=label_to_color[label], label=label, zorder=2)
                    ax.legend(loc="upper right", fontsize=8, frameon=True)
            else:
                ax.scatter(x, y, s=point_size, edgecolor="black", color="skyblue", zorder=2)

            # Add node labels
            if show_node_labels:
                for i in range(len(x)):
                    ax.text(x[i], y[i], str(i), fontsize=node_label_size, ha="center", va="center", zorder=3)

            # Draw Poincaré disk boundary
            ax.add_artist(plt.Circle((0, 0), 1, fill=False, color="black", linestyle="--"))
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_aspect("equal")
            ax.axis("off")

        elif output_space.lower() == "spherical":
            ax = plt.subplot(111, projection="polar")

            # Plot edges
            if edge_list:
                logger.debug(f"Plotting {len(edge_list)} edges")
                for u, v in edge_list:
                    if u < len(x) and v < len(x):  # Check bounds
                        p1 = (y[u], x[u])
                        p2 = (y[v], x[v])
                        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="gray", linewidth=edge_width, alpha=edge_alpha, zorder=1)

            # Plot points with labels
            if labels:
                labels = np.array(labels)
                unique_labels = sorted(set(labels))
                cmap = cm.get_cmap(colormap, len(unique_labels))
                label_to_color = {label: cmap(i) for i, label in enumerate(unique_labels)}

                for label in unique_labels:
                    indices = np.where(labels == label)[0]
                    ax.scatter(y[indices], x[indices], s=point_size, edgecolor="black", color=label_to_color[label], label=label, zorder=2)
                    ax.legend(loc="upper right", fontsize=8, frameon=True)
            else:
                ax.scatter(y, x, s=point_size, edgecolor="black", color="skyblue", zorder=2)

            # Add node labels
            if show_node_labels:
                for i in range(len(x)):
                    ax.text(y[i], x[i], str(i), fontsize=node_label_size, ha="center", va="center", zorder=3)

            # Draw spherical boundary with radius > max distance from origin
            max_radius = np.max(x)
            boundary_radius = max_radius * 1.1  # 10% margin
            ax.set_ylim(0, boundary_radius)  # or
            ax.set_rlim(0, boundary_radius)
            ax.set_aspect("equal")

        if save_path:
            logger.info(f"Saving plot to: {save_path}")
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            plt.show()
            logger.info(f"Plot saved successfully to {save_path}")
        else:
            logger.debug("Displaying plot")
            plt.show()

    def compute_distances(self, embeddings: np.ndarray, space: str = "hyperboloid") -> np.ndarray:
        """
        Compute pairwise hyperbolic distances between embeddings.

        Parameters:
        - embeddings: Embeddings array
        - space: Space of the embeddings

        Returns:
        - Distance matrix
        """
        logger.info(f"Computing distances for {space} space (embeddings shape: {embeddings.shape})")
        distances = compute_distances(embeddings, space)
        logger.info(f"Distance matrix computed: shape {distances.shape}")
        return distances

    def get_available_models(self) -> List[str]:
        """Get list of available embedding models."""
        models = list(self.MODEL_REGISTRY.keys())
        logger.debug(f"Available models: {models}")
        return models

    def get_model_info(self) -> Dict:
        """Get information about the current model."""
        info = {
            "embedding_type": self.embedding_type,
            "model_class": type(self.model).__name__,
            "available_models": self.get_available_models(),
            "supported_spaces": self.SUPPORTED_SPACES,
        }
        logger.debug(f"Model info: {info}")
        return info
