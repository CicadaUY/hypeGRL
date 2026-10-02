"""Rebuild the vendored phylogenies in ``data/phylogeny/`` from the Open Tree of Life.

Each whole clade is resolved to its Open Tree taxonomy (OTT) id, its induced subtree
is fetched from the synthetic tree as newick, and the tree is written as an edge list
of integer labels in a canonical order. The two ``*_sub`` files are not whole clades:
the clades whose shape is most interesting are too large to embed, so each is a rooted
subtree cut locally from its parent clade by :func:`widest_subtree`.

Every step is deterministic, so for a fixed Open Tree synthesis release the output is
byte-identical to the vendored files. The files were fetched from release
``opentree16.1``; a later release may change the trees.

Run, from the repository root (needs network access and ``biopython``):
    python -m experiments.build_phylogenies                    # every file
    python -m experiments.build_phylogenies --only cichlidae fabaceae_sub
    python -m experiments.build_phylogenies --out /tmp/check   # compare, not overwrite
"""
from __future__ import annotations

import argparse
import io
import json
import urllib.request
from pathlib import Path

import networkx as nx
import numpy as np

API = "https://api.opentreeoflife.org/v3/"
PHYLOGENY = Path(__file__).resolve().parent / "data" / "phylogeny"

CLADES = [
    "Anseriformes", "Bromeliaceae", "Cactaceae", "Carnivora", "Caudata", "Cephalopoda",
    "Cetacea", "Chiroptera", "Cichlidae", "Falconiformes", "Gadiformes", "Lagomorpha",
    "Mantodea", "Marsupialia", "Perissodactyla", "Piciformes", "Pinaceae", "Primates",
    "Psittaciformes", "Rodentia", "Salmonidae", "Serpentes", "Strigiformes",
    "Syngnathidae", "Testudines",
]
# label -> parent clade; each is that clade's widest subtree in the size band
SUBTREES = {"fabaceae_sub": "Fabaceae", "poaceae_sub": "Poaceae"}
SUBTREE_SIZE = (600, 8000)


def _post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(API + path, data=json.dumps(payload).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)


def fetch_newick(clade: str) -> str:
    """The newick of ``clade``'s induced subtree of the Open Tree synthetic tree."""
    match = _post("tnrs/match_names", {"names": [clade]})["results"][0]["matches"][0]
    ott_id = match["taxon"]["ott_id"]
    subtree = _post("tree_of_life/subtree", {"ott_id": ott_id, "format": "newick"})
    return subtree["newick"]


def newick_to_graph(newick: str) -> nx.Graph:
    """The tree as an undirected graph on integer labels ``0..N-1``.

    Nodes are first keyed by their newick name, or ``_i<k>`` for an unnamed node with
    ``k`` its position in a depth-first walk, and then relabelled in sorted order of
    those keys. The labels therefore depend only on the newick, which is what makes the
    output reproducible.
    """
    from Bio import Phylo

    tree = Phylo.read(io.StringIO(newick), "newick")
    G = nx.Graph()
    counter = 0

    def walk(clade, parent):
        nonlocal counter
        name = clade.name or f"_i{counter}"
        counter += 1
        if parent is None:
            G.add_node(name)
        else:
            G.add_edge(parent, name)
        for child in clade.clades:
            walk(child, name)

    walk(tree.root, None)
    return nx.convert_node_labels_to_integers(G, ordering="sorted")


def _eccentricity_iqr(T: nx.Graph) -> tuple[float, int]:
    """(interquartile range of the node eccentricities, diameter) of a tree.

    On a tree every node's eccentricity is its distance to the farther end of any
    diameter path, so three breadth-first sweeps give all of them.
    """
    d = nx.single_source_shortest_path_length(T, next(iter(T)))
    a = max(d, key=d.get)
    da = nx.single_source_shortest_path_length(T, a)
    b = max(da, key=da.get)
    db = nx.single_source_shortest_path_length(T, b)
    ecc = np.array([max(da[v], db[v]) for v in T])
    return float(np.percentile(ecc, 75) - np.percentile(ecc, 25)), da[b]


def widest_subtree(G: nx.Graph, lo: int, hi: int) -> nx.Graph:
    """The rooted subtree of ``G`` with between ``lo`` and ``hi`` nodes whose nodes are
    spread most widely across depths, measured by the interquartile range of their
    eccentricities; ties go to the larger, then the deeper subtree, then the larger root
    label.

    ``G`` is rooted at one end of its diameter, so the candidate subtrees are the deep,
    extended parts of the tree. Relabelled ``0..n-1`` in sorted order.
    """
    d = nx.single_source_shortest_path_length(G, next(iter(G)))
    root = max(d, key=d.get)
    children = nx.bfs_successors(G, root)
    order = [root] + [w for _, ws in children for w in ws]
    parent = dict(nx.bfs_predecessors(G, root))
    size = {u: 1 for u in order}
    for u in reversed(order[1:]):
        size[parent[u]] += size[u]

    best = None
    for u in order:
        if not lo <= size[u] <= hi:
            continue
        sub = G.subgraph(nx.dfs_preorder_nodes(nx.bfs_tree(G, root), u))
        iqr, diameter = _eccentricity_iqr(sub)
        key = (iqr, sub.number_of_nodes(), diameter, u)
        if best is None or key > best[0]:
            best = (key, sub)
    if best is None:
        raise ValueError(f"no rooted subtree with {lo}..{hi} nodes")
    return nx.convert_node_labels_to_integers(best[1], ordering="sorted")


def write_edgelist(label: str, G: nx.Graph, out: Path) -> Path:
    """Write ``G`` with one header line and its edges in sorted order."""
    out.mkdir(parents=True, exist_ok=True)
    edges = sorted(tuple(sorted(e)) for e in G.edges())
    path = out / f"{label.lower()}.edgelist"
    path.write_text(f"# {label} clade from the Open Tree of Life; "
                    f"{G.number_of_nodes()} nodes, {len(edges)} edges; "
                    f"canonical sorted order\n"
                    + "\n".join(f"{u} {v}" for u, v in edges) + "\n")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", metavar="LABEL",
                    help="file labels to rebuild, e.g. cichlidae fabaceae_sub")
    ap.add_argument("--out", type=Path, default=PHYLOGENY)
    args = ap.parse_args()

    wanted = {lbl.lower() for lbl in args.only} if args.only else None
    for clade in CLADES:
        if wanted is None or clade.lower() in wanted:
            path = write_edgelist(clade, newick_to_graph(fetch_newick(clade)), args.out)
            print(f"wrote {path}", flush=True)
    for label, parent in SUBTREES.items():
        if wanted is None or label in wanted:
            G = newick_to_graph(fetch_newick(parent))
            path = write_edgelist(label, widest_subtree(G, *SUBTREE_SIZE), args.out)
            print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
