import argparse
import os

import networkx as nx
import numpy as np

from hyperbolic_embeddings import HyperbolicEmbeddings


def main():
    parser = argparse.ArgumentParser(description="Train and plot hyperbolic embeddings for the Karate Club graph.")
    parser.add_argument(
        "--embedding_type",
        type=str,
        required=True,
        choices=["poincare_embeddings", "lorentz", "dmercator", "hydra", "poincare_maps", "hypermap", "hydra_plus"],
        help="Type of embedding model to use.",
    )
    parser.add_argument("--model_dir", type=str, default="saved_models/tree_test", help="Directory to save the trained model.")
    parser.add_argument("--plot_dir", type=str, default="test/tree_test/plots", help="Directory to save the plots.")
    parser.add_argument(
        "--output_space",
        type=str,
        default="poincare",
        choices=["poincare", "hyperboloid", "klein", "hemisphere", "half_plane", "spherical"],
        help="Space to plot embeddings in.",
    )

    args = parser.parse_args()

    # Load Karate Club graph
    G = nx.karate_club_graph()
    edge_list = list(G.edges())
    A = nx.to_numpy_array(G)
    labels = [G.nodes[n]["club"] for n in sorted(G.nodes())]

    # Embedding configs (embedding_space now determined by model's native_space property)
    configurations = {
        "poincare_embeddings": {"dim": 2, "negs": 5, "epochs": 1000, "batch_size": 256, "dimension": 1},
        "lorentz": {"dim": 2, "epochs": 10000, "batch_size": 1024, "num_nodes": 31},
        "dmercator": {"dim": 1},
        "hydra": {"dim": 2},
        "poincare_maps": {"dim": 2, "epochs": 1000},
        "hypermap": {"dim": 3},
        "hydra_plus": {"dim": 2},
    }

    embedding_type = args.embedding_type

    config = configurations[embedding_type]
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    model_path = os.path.join(args.model_dir, f"{embedding_type}_embeddings.bin")
    plot_path = os.path.join(args.plot_dir, f"{embedding_type}_{args.output_space}_embeddings.pdf")

    print(f"Training {embedding_type} embeddings...")
    embedding_runner = HyperbolicEmbeddings(embedding_type=embedding_type, config=config)

    if embedding_type == "hydra" or embedding_type == "poincare_maps" or embedding_type == "lorentz":
        embedding_runner.train(adjacency_matrix=A, model_path=model_path)
    else:
        embedding_runner.train(edge_list=edge_list, model_path=model_path)

    # Get embeddings for plotting
    embeddings = embedding_runner.get_all_embeddings(model_path)

    print(f"Embeddings saved to {model_path}")

    # Get the native embedding space from the model's property
    native_embedding_space = embedding_runner.model.native_space

    # Print model information
    model_info = embedding_runner.get_model_info()
    print(f"\nModel Information:")
    print(f"  Embedding Type: {model_info['embedding_type']}")
    print(f"  Model Class: {model_info['model_class']}")
    print(f"  Native Embedding Space: {native_embedding_space}")
    print(f"  Output Space: {args.output_space}")
    print(f"  Embedding Shape: {embeddings.shape}")

    # Validate embeddings
    embedding_runner.validate_embeddings(embeddings, native_embedding_space)

    # Plot embeddings with new API
    embedding_runner.plot_embeddings(
        embeddings=embeddings,
        embedding_space=native_embedding_space,
        output_space=args.output_space,
        labels=labels,
        edge_list=edge_list,
        save_path=plot_path,
        plot_geodesic=(args.output_space == "poincare"),  # Only plot geodesics for Poincaré
        figsize=(10, 10),
        point_size=150,
        show_node_labels=True,
        node_label_size=10,
        edge_alpha=0.6,
        edge_width=1.0,
        colormap="viridis",
    )


if __name__ == "__main__":
    main()
