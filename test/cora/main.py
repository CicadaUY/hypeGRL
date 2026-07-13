import argparse
import json
import os
import pickle

import networkx as nx
import numpy as np

from hyperbolic_embeddings import HyperbolicEmbeddings


def main():
    parser = argparse.ArgumentParser(description="Train and plot hyperbolic embeddings for Cora graph.")
    parser.add_argument(
        "--embedding_type",
        type=str,
        required=True,
        choices=["poincare_embeddings", "lorentz", "dmercator", "hydra", "poincare_maps", "hypermap", "hydra_plus"],
        help="Type of embedding model to use.",
    )
    parser.add_argument("--model_dir", type=str, default="saved_models/cora", help="Directory to save the trained model.")
    parser.add_argument("--plot_dir", type=str, default="test/cora/plots", help="Directory to save the plots.")

    args = parser.parse_args()

    # Load CORA graph
    with open("./data/Cora/cora_graph.pkl", "rb") as f:
        edge_list = pickle.load(f)
    with open("./data/Cora/cora_graph.json", "r") as f:
        graph_data = json.load(f)

    # Build networkx graph from edge index
    G = nx.Graph()
    G.add_edges_from(edge_list)

    A = nx.to_numpy_array(G)

    labels = graph_data["y"]

    # Embedding configs
    configurations = {
        "poincare_embeddings": {"dim": 2, "negs": 5, "epochs": 1000, "batch_size": 256, "dimension": 1},
        "lorentz": {"dim": 2, "epochs": 50000, "batch_size": 256},
        "dmercator": {"dim": 1},
        "hydra": {"dim": 2},
        "poincare_maps": {"dim": 2, "epochs": 1000},
        "hypermap": {},
        "hydra_plus": {"dim": 2},
    }

    embedding_type = args.embedding_type

    config = configurations[embedding_type]
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    model_path = os.path.join(args.model_dir, f"{embedding_type}_embeddings.bin")
    plot_path = os.path.join(args.plot_dir, f"{embedding_type}_embeddings.pdf")

    print(f"Training {embedding_type} embeddings...")
    embedding_runner = HyperbolicEmbeddings(embedding_type=embedding_type, config=config)

    if embedding_type == "hydra" or embedding_type == "poincare_maps" or embedding_type == "lorentz":
        embedding_runner.train(adjacency_matrix=A, model_path=model_path)
    else:
        embedding_runner.train(edge_list=edge_list, model_path=model_path)

    embeddings = embedding_runner.get_all_embeddings(model_path)
    embedding_runner.plot_embeddings(labels=labels, edge_list=edge_list, save_path=plot_path)


if __name__ == "__main__":
    main()
