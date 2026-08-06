import networkx as nx
from geometry_diagnostics import diagnose, format_report

graphs = {
    'balanced_tree(3,5)': nx.balanced_tree(3, 5),
    'grid_2d(20x20)': nx.grid_2d_graph(20, 20),
    'ER(n=300,p=0.03)': nx.erdos_renyi_graph(300, 0.03, seed=0),
    'BA scale-free(n=300,m=3)': nx.barabasi_albert_graph(300, 3, seed=0),
    'WS small-world(n=300,k=6,p=0.05)': nx.watts_strogatz_graph(300, 6, 0.05, seed=0),
    'complete(n=40)': nx.complete_graph(40),
}
for name, G in graphs.items():
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    G = nx.convert_node_labels_to_integers(G)
    r = diagnose(G, n_hyperbolicity_samples=5000, n_ricci_edges=200, seed=0)
    print(format_report(name, r))
