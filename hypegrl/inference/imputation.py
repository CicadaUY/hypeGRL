"""Methods to compute edge imputation of unknown entries. """
import geoopt
import torch
import networkx as nx
import numpy as np
import numpy.ma as ma

from typing import Optional


def compute_threshold_two_clusters(
        G: nx.Graph, 
        X: np.ndarray, 
        unknown_edges: list[tuple[int, int]],
        manifold: geoopt.Manifold
        ) -> dict:
    """
    Takes all the known entries in the adjacency matrix of graph ``G`` and computes the mean distance
    between pairs of nodes actually connected, and pairs of nodes not connected. For this, embeddings
    in manifold ``manifold`` are used, 
    and the distance between them is computed accordingly. Finally, the mean between the corresponding two 
    means is returned as the threshold to flag an edge. 

    Parameters
    ----------
    G : nx.Graph
        The graph.
    X : np.ndarray
        The embeddings of shape ``(N,d)``, with ``N`` the number of nodes in ``G``. A correspondence between the order of each embedding
        and nx.to_numpy_array is expected.
    unknown_edges : list[tuple[int, int]]
        List of ``(m, n)`` tuples for edges whose weights are unknown, where ``m`` refers to the node's index in the resulting adjacency matrix.
    manifold : geoopt.Manifold
        A ``geoopt.Manifold`` instance defining the embedding geometry.
        Used to compute the distance between embeddings in ``X``.

    Returns
    -------
    dict with keys:

    - ``D``: ``(N, N)`` Torch.tensor with the distance between embeddings.
    - ``distances``: list of the distances between nodes corresponding to the unknown edges.
    - ``threshold``: the computed threshold.

    """
    D = manifold.dist(torch.tensor(X).unsqueeze(1), torch.tensor(X).unsqueeze(0))
    distances = []
    for (m, n) in unknown_edges:
        distances.append(D[m,n])
    
    # TODO probably there's a more efficient way for this
    adjacency_matrix = nx.to_numpy_array(G)
    distances_edge = []
    distances_no_edge = []
    for m in np.arange(D.shape[0]):
        for n in np.arange(D.shape[0]):
            if m!=n:
                if (min(m, n), max(m, n)) not in unknown_edges: 
                    if (adjacency_matrix[m,n]==1):
                        distances_edge.append(D[m,n])
                    else:
                        distances_no_edge.append(D[m,n])    
    threshold = (np.mean(distances_edge) + np.mean(distances_no_edge))/2
    result = {}
    result['D'] = D
    result['distances'] = distances
    result['threshold'] = threshold
    return result

def compute_results_imputation_unweighted(
        G: nx.Graph, 
        unknown_edges: list[tuple[int, int]],
        a_omega: torch.Tensor,
        threshold: Optional[float] = 0.5
        ) -> dict:
    """
    Given a graph, a list of unknown edges and their computed imputed weights, 
    this method produces a dictionary for easily computing performance 
    indicators of the imputed values. The method is designed for unweighted graphs, 
    since we simply compare the imputed weight to a certain threshold to flag an edge. 

    Parameters
    ----------
    G : nx.Graph
        The input graph, that contains the ground truth for the unknown edges. We are assuming an undirected graph with nodes that are indexed starting from 0.
    unknown_edges : list[tuple[int, int]]
        List of ``(m, n)`` tuples for edges whose weights are unknown, where ``m`` refers to the node's index in the resulting adjacency matrix.
    a_omega : torch.Tensor
        A tensor including the values imputed to the edges' weight.
    threshold : Optional[float], optional
        The value to which compare the imputed value to flag an actual edge. The default is 0.5.

    Returns
    -------
    dict with keys:

    - ``actual_values``: an array with the actual weight (0 or 1) for each element in ``unknown_edges``. 
    - ``predicted_values``: an array with a 1 if we flagged an edge for the corresponding pair of nodes in ``unknown_edges`` or a 0. 
    - ``a_omegas``: the imputed weights. 

    """
    
    results = {'actual_values': [], 'predicted_values': [], 'a_omegas': []}
    for (m, n), val in zip(unknown_edges, a_omega):
        actual_value = 0
        if G.has_edge(m, n):
            actual_value = 1
            
        # print(f"  ({m},{n}): {val:.6f} (actual weight: {actual_value})")
        predicted_value = int(val>threshold)
        
        results['actual_values'].append(actual_value)
        results['predicted_values'].append(predicted_value)
        results['a_omegas'].append(val)
        
    return results

def compute_results_distances(
        G: nx.Graph, 
        unknown_edges: list[tuple[int, int]], 
        D: torch.Tensor, 
        threshold: float
        ) -> dict:
    """
    Given a graph, a list of unknown edges and the computed distance between the corresponding embeddings, 
    this method produces a dictionary for easily computing performance 
    indicators of these distances. The method is designed for unweighted graphs, 
    since we simply compare the distance to a certain threshold to flag an edge. 

    Parameters
    ----------
    G : nx.Graph
        The input graph, that contains the ground truth for the unknown edges. We are assuming an undirected graph with nodes that are indexed starting from 0.
    unknown_edges : list[tuple[int, int]]
        List of ``(m, n)`` tuples for edges whose weights are unknown, where ``m`` refers to the node's index in the resulting adjacency matrix.
    D : torch.Tensor
        A ``(N,N)`` tensor including the distances between all pairs of the nodes' embeddings.
    threshold : float
        The value to which compare the distance between nodes to flag an actual edge.

    Returns
    -------
    dict with keys:

    - ``actual_values``: an array with the actual weight (0 or 1) for each element in ``unknown_edges``. 
    - ``predicted_values``: an array with a 1 if we flagged an edge for the corresponding pair of nodes in ``unknown_edges`` or a 0. 
    - ``distances`` the distance between embeddings corresponding to pairs of nodes in the ``unknown_edges`` set. 

    """
    results = {}
    ground_truths = []
    distances = []
    for (m, n) in unknown_edges:
        actual_value = 0
        if G.has_edge(m, n):
            actual_value = 1
        ground_truths.append(actual_value)
        # print(f"  ({m},{n}): {val:.6f} (actual weight: {actual_value})")
        distances.append(D[m,n])

    predictions = (distances < threshold).astype(int)
        
    results['actual_values'] = np.array(ground_truths)
    results['predicted_values'] = np.array(predictions)
    results['distances'] = np.array(distances)

    return results

def compute_a_omega_init(
        G: nx.Graph, 
        unknown_edges: list[tuple[int, int]]
        ) -> np.array:
    """
    A simple estimation for the weight of the edge between each pair of nodes in ``unknown_edges``is computed.
    The idea is to simply use the mean per row and column in the adjacency matrix (naturally, ignoring the unknown entries).

    Parameters
    ----------
    G : nx.Graph
        The input graph. 
    unknown_edges : list[tuple[int, int]]
        List of ``(m, n)`` tuples for edges whose weights are unknown, where ``m`` refers to the node's index in the resulting adjacency matrix.

    Returns
    -------
    np.array
        The computed means for each element in ``unknown_edges``. 

    """
    A_np = nx.to_numpy_array(G)
    a_omega_init = np.full(len(unknown_edges), 0.5)
    # masked entries (mask=1) are ignored in numpy.ma
    mask = np.zeros_like(A_np)
    for k, (m, n) in enumerate(unknown_edges):
        mask[m,n] = 1
        mask[n,m] = 1
        
    masked_A = ma.masked_array(A_np, mask=mask)
    row_means = masked_A.mean(axis=1).data
    col_means = masked_A.mean(axis=0).data
    P = (row_means+np.atleast_2d(col_means).T)/2
    for k, (m, n) in enumerate(unknown_edges):
        a_omega_init[k] = P[m,n]
        # A[n, m] = P[n,m]
    return a_omega_init
