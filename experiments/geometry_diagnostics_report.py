"""Run :mod:`experiments.geometry_diagnostics` across every dataset in
:mod:`experiments.datasets`, plus a set of additional networks commonly used
in the hyperbolic-embedding (and, for the OGB entries, the broader graph-
embedding) literature (see "Additional literature datasets" below for the
rationale behind each), and print/save a comparison table.

Every loader is tried in turn; one that needs a dependency, local file, or
download unavailable in the current environment (``torch_geometric`` for
PolBlogs/Airports/citation networks, ``ogb`` for the OGB datasets, ``nltk``
+ the WordNet corpus, network access for OpenFlights/SNAP/HGCN) is reported
as skipped rather than aborting the whole run, so this is safe to run with
a partial install.

Run:
    python experiments/geometry_diagnostics_report.py
"""
import gzip
import sys
import warnings
from pathlib import Path
from typing import Callable, Optional

import networkx as nx

REPO = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, REPO)

from geometry_diagnostics import diagnose, format_report  # noqa: E402

RESULTS = Path(REPO) / "experiments" / "results"


def _patch_aiohttp_trust_env() -> None:
    """
    Make aiohttp (used internally by fsspec's ``HTTPFileSystem``, which
    recent ``torch_geometric`` versions route dataset downloads through --
    ``Planetoid``/``PPI``/``WordNet18RR``/``PolBlogs``/``Airports`` here --
    honour ``HTTP_PROXY``/``HTTPS_PROXY`` environment variables.

    ``aiohttp.ClientSession`` defaults to ``trust_env=False``: unlike
    ``requests``, ``urllib``, ``curl``, ``pip``, or ``git``, it silently
    ignores proxy env vars unless ``trust_env=True`` is passed explicitly.
    PyG's dataset classes don't expose a way to pass that through, so
    behind a proxy their downloads can hang and eventually raise
    ``fsspec.exceptions.FSTimeoutError`` even when the same proxy works
    fine for pip/git and for the plain-``urllib`` SNAP/Disease loaders
    elsewhere in this file (those aren't affected -- ``urllib`` reads proxy
    env vars by default).

    Patches ``aiohttp.ClientSession.__init__`` once, at import time, to
    default ``trust_env`` to ``True`` (an explicit ``trust_env=False`` from
    a caller would still be respected via ``setdefault``). A no-op if
    ``aiohttp`` isn't installed (e.g. ``torch_geometric`` itself isn't) or
    if it's already been patched (e.g. this module reloaded).
    """
    try:
        import aiohttp
    except ImportError:
        return

    if getattr(aiohttp.ClientSession.__init__, "_trust_env_patched", False):
        return

    _original_init = aiohttp.ClientSession.__init__

    def _patched_init(self, *args, **kwargs):
        kwargs.setdefault("trust_env", True)
        _original_init(self, *args, **kwargs)

    _patched_init._trust_env_patched = True
    aiohttp.ClientSession.__init__ = _patched_init


_patch_aiohttp_trust_env()


def _largest_component(G: nx.Graph) -> nx.Graph:
    """diagnose() requires connected input; every loader below is already
    documented as returning the largest connected component except
    balanced_tree_graph (always connected) and single_cell_graph (connected
    by construction, via its component-joining step) -- this is a no-op
    safety net for those, not a correction to any of them."""
    if nx.is_connected(G):
        return G
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()


# ----------------------------------------------------------------------
# Additional literature datasets
#
# Chosen to span the datasets most directly tied to specific hyperbolic-
# embedding papers, rather than just "more networks":
#
# - ``karate_club``: Zachary's karate club (networkx built-in, no
#   download). Small and not really a "hyperbolic" benchmark on its own
#   merits, but it's the one network that shows up as a toy illustrative
#   example across nearly the whole literature (Poincare embeddings,
#   HyperMap, ...), and it's a free, always-available control here.
# - ``wordnet/mammal``: the WordNet noun-hypernym subtree rooted at
#   "mammal.n.01" -- Nickel & Kiela (2017), "Poincare Embeddings for
#   Learning Hierarchical Representations", the paper that put hyperbolic
#   embeddings on the map; the mammal subtree specifically is their own
#   qualitative-figure example. The *full* noun hierarchy (~82k synsets,
#   used for their actual reconstruction table) is deliberately not used
#   here: this module's ``mean_hyperbolicity``/``ollivier_ricci_curvature``
#   pipeline builds dense (N, N) distance matrices and would need tens of
#   gigabytes at that N -- a subtree keeps the same "this is a literal
#   hierarchy" character at a size the pipeline can actually run on. See
#   ``wordnet_noun_subtree_graph`` to pick a different (larger or smaller)
#   root or raise ``max_nodes``.
# - ``citation/{Cora,CiteSeer,PubMed}``: the standard trio for hyperbolic
#   *graph neural network* papers -- Chami et al. (2019), "Hyperbolic
#   Graph Convolutional Neural Networks" (HGCN), report Gromov
#   delta-hyperbolicity for exactly these three and correlate it with how
#   much hyperbolic GCNs beat Euclidean ones, which is precisely the
#   question this module's ``diagnose()`` is trying to answer generically.
# - ``internet_as``: the Internet AS-level topology (SNAP AS-733,
#   2000-01-02 snapshot) -- the flagship real-network validation of
#   Boguna, Papadopoulos & Krioukov (2010), "Sustaining the Internet with
#   Hyperbolic Mapping", and the domain D-Mercator/HyperMap (both already
#   first-class embedders in this library) were built for.
# - ``arxiv_collab``: the Arxiv HEP-TH co-authorship network (SNAP
#   ca-HepTh) -- a canonical sparse-yet-clustered scientific collaboration
#   network of the kind used throughout the Krioukov/Boguna/Papadopoulos
#   line of work on hyperbolic random graph models; SNAP reports its own
#   average clustering coefficient at 0.47 despite a mean degree far below
#   its node count -- close to the PNAS sparse-and-clustered regime this
#   module checks for directly.
# - ``disease``: Chami et al.'s (2019) synthetic "Disease" tree, built by
#   simulating an SIR epidemic-spreading process over a random tree, with
#   node labels for infection status. HGCN's own purpose-built stress test
#   for literal hierarchical structure -- alongside Cora/PubMed/Airport
#   (already covered above), one of the three domains their paper reports
#   results on, and reportedly their widest hyperbolic-vs-Euclidean gap.
# - ``ppi``: one tissue's human protein-protein interaction network
#   (Zitnik & Leskovec, 2017) -- the third of HGCN's three real-network
#   domains, alongside citation networks and Airport.
# - ``wordnet18rr``: WN18RR, the WordNet lexical-relation knowledge graph
#   (Dettmers et al., 2018), edge direction and relation types dropped.
#   The standard benchmark for the *hyperbolic knowledge-graph embedding*
#   line of work (Balazevic et al. 2019 "MuRP"; Chami et al. 2020
#   "Low-Dimensional Hyperbolic Knowledge Graph Embeddings") -- distinct
#   from, and a useful contrast to, ``wordnet/mammal`` above: same source
#   lexicon, a genuinely different relation structure (synonymy, meronymy,
#   antonymy, ... instead of a single is-a hierarchy).
# - ``ogb/arxiv``: ogbn-arxiv (Hu et al., 2020, "Open Graph Benchmark"),
#   169,343 nodes. Not from the hyperbolic-embedding literature
#   specifically -- it's the closest thing graph ML has to a default
#   "medium-large" node-classification benchmark today, precisely *because*
#   Cora/CiteSeer/PubMed (2.7k-20k nodes) are considered too small to
#   separate methods reliably; several papers explicitly justify choosing
#   it on those grounds. A natural larger sibling to the citation/* trio
#   above, and worth checking whether the diagnosis (and any embedding
#   method's relative ranking) holds up at this scale.
# - ``ogb/collab``: ogbl-collab (Hu et al., 2020), 235,868-node author
#   collaboration network. From OGB's *link prediction* track specifically
#   -- link prediction is the task embedding papers (hyperbolic or not)
#   most often evaluate on, so this is closer to those papers' actual
#   evaluation setting than a node-classification network repurposed for
#   it.
# - ``ogb/ddi``: ogbl-ddi (Hu et al., 2020), a drug-drug interaction
#   network -- only 4,267 nodes but ~1.3M edges (mean degree ~500+): an
#   extremely dense contrast case, and (unlike the two above) small enough
#   to be an easy addition rather than a scale stress-test.
# ----------------------------------------------------------------------

_SNAP_URLS = {
    "internet_as": "https://snap.stanford.edu/data/as20000102.txt.gz",
    "arxiv_collab": "https://snap.stanford.edu/data/ca-HepTh.txt.gz",
}
_DISEASE_URL = (
    "https://raw.githubusercontent.com/HazyResearch/hgcn/master/"
    "data/disease_nc/disease_nc.edges.csv"
)


def _download(url: str, path: Path) -> Path:
    """Download ``url`` to ``path`` if not already cached; return ``path``."""
    import urllib.request

    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    return path


def _read_snap_edgelist_gz(path: Path) -> nx.Graph:
    """
    Parse a gzip'd SNAP edge-list file into an undirected graph.

    SNAP's plain edge-list files share one format: zero or more ``#``
    comment/header lines, then one ``FromNodeId<whitespace>ToNodeId`` pair
    per line. Both files used here (``as20000102.txt.gz``, ``ca-HepTh.txt.gz``)
    describe inherently undirected relationships (AS peering, coauthorship)
    even though SNAP's own directed-graph tooling stores each edge once in
    an arbitrary direction; ``nx.Graph`` (not ``nx.DiGraph``) is the correct
    read here, not a simplification.
    """
    G = nx.Graph()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            u, v = line.split()[:2]
            G.add_edge(int(u), int(v))
    return G


def internet_as_graph(cache_dir: Optional[str] = None) -> nx.Graph:
    """Internet AS-level topology, 2000-01-02 snapshot (SNAP AS-733). See
    the "Additional literature datasets" note above this section."""
    directory = Path(cache_dir) if cache_dir is not None else Path("./data/internet_as")
    path = _download(_SNAP_URLS["internet_as"], directory / "as20000102.txt.gz")
    return _read_snap_edgelist_gz(path)


def arxiv_collaboration_graph(cache_dir: Optional[str] = None) -> nx.Graph:
    """Arxiv HEP-TH co-authorship network (SNAP ca-HepTh). See the
    "Additional literature datasets" note above this section."""
    directory = Path(cache_dir) if cache_dir is not None else Path("./data/arxiv_collab")
    path = _download(_SNAP_URLS["arxiv_collab"], directory / "ca-HepTh.txt.gz")
    return _read_snap_edgelist_gz(path)


def wordnet_noun_subtree_graph(
    root: str = "mammal.n.01", max_nodes: int = 3000
) -> nx.Graph:
    """
    Undirected hypernym graph of the WordNet noun subtree rooted at
    ``root`` (default: the mammal subtree from Nickel & Kiela's own
    qualitative figures). See the "Additional literature datasets" note
    above this section for why a subtree rather than the full ~82k-synset
    noun hierarchy.

    Edges are *direct* hypernym relations (parent-child in the hierarchy),
    not the transitive closure Nickel & Kiela train their reconstruction
    task on -- the direct edges are the graph *structure*; the transitive
    closure is a denser training signal derived from it, and this module
    diagnoses structure.

    Requires ``nltk`` and its WordNet corpus (``nltk.download("wordnet")``,
    attempted automatically on first use if missing).

    Parameters
    ----------
    root:
        A WordNet synset name to root the subtree at, e.g. ``"mammal.n.01"``,
        ``"animal.n.01"`` (bigger), ``"furniture.n.01"`` (smaller).
    max_nodes:
        Safety cap: if the subtree has more synsets than this, it's
        truncated to a breadth-first ``max_nodes`` prefix from ``root``
        (keeps the graph's ``mean_hyperbolicity``/``ollivier_ricci_curvature``
        dense-distance-matrix cost bounded).
    """
    try:
        from nltk.corpus import wordnet as wn
        wn.ensure_loaded()
    except LookupError:
        import nltk
        nltk.download("wordnet")
        from nltk.corpus import wordnet as wn

    root_synset = wn.synset(root)

    # BFS over hyponyms (children) from root, capped at max_nodes, so a
    # truncation (if any) keeps a single connected subtree near the root
    # rather than an arbitrary scattered subset.
    G = nx.Graph()
    visited = {root_synset}
    queue = [root_synset]
    G.add_node(root_synset.name())
    while queue and len(visited) < max_nodes:
        node = queue.pop(0)
        for child in node.hyponyms():
            if child in visited:
                continue
            if len(visited) >= max_nodes:
                break
            visited.add(child)
            G.add_edge(node.name(), child.name())
            queue.append(child)
    return G


def citation_graph(name: str = "Cora", root: Optional[str] = None) -> nx.Graph:
    """
    Citation network (``"Cora"``, ``"CiteSeer"``, or ``"PubMed"``), edge
    direction dropped. See the "Additional literature datasets" note above
    this section -- these are the three datasets Chami et al.'s HGCN paper
    reports Gromov delta-hyperbolicity for.
    """
    from torch_geometric.datasets import Planetoid

    data = Planetoid(
        root=str(root) if root is not None else f"./data/citation_{name.lower()}",
        name=name,
    )[0]
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    G.add_edges_from(data.edge_index.numpy().T.tolist())
    return G


def karate_club_graph() -> nx.Graph:
    """Zachary's karate club (networkx built-in). See the "Additional
    literature datasets" note above this section."""
    return nx.karate_club_graph()


def disease_graph(cache_dir: Optional[str] = None) -> nx.Graph:
    """
    Chami et al.'s (2019, HGCN) synthetic "Disease" tree. See the
    "Additional literature datasets" note above this section.

    Downloads the edge list from HGCN's own GitHub repository
    (HazyResearch/hgcn) -- the same repo the citation-network loader above
    traces back to.

    Note
    ----
    The exact file layout of HGCN's ``data/disease_nc/`` directory
    (``disease_nc.edges.csv``, comma-separated node-id pairs) is inferred
    from that repository's well-known, widely-reused ``data_utils.py``
    loading code, not independently re-verified against a live download in
    this session -- if the format has since changed this will fail loudly
    (a parse error naming the bad line) rather than silently return
    something wrong. Worth confirming the first time you run it.
    """
    directory = Path(cache_dir) if cache_dir is not None else Path("./data/disease")
    path = _download(_DISEASE_URL, directory / "disease_nc.edges.csv")
    G = nx.Graph()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            u, v = line.split(",")[:2]
            G.add_edge(int(u), int(v))
    return G


def ppi_graph(cache_dir: Optional[str] = None, graph_index: int = 0) -> nx.Graph:
    """
    One human tissue's protein-protein interaction network from the PPI
    dataset (Zitnik & Leskovec, 2017), via PyTorch Geometric. See the
    "Additional literature datasets" note above this section.

    PPI ships as ~20 separate per-tissue graphs, not one connected network
    (different tissues' interactomes, not naturally a single graph);
    ``graph_index`` picks which one (default: the first of the training
    split).
    """
    from torch_geometric.datasets import PPI

    root = str(cache_dir) if cache_dir is not None else "./data/ppi"
    dataset = PPI(root=root, split="train")
    data = dataset[graph_index]
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    G.add_edges_from(data.edge_index.numpy().T.tolist())
    return G


def wordnet18rr_graph(cache_dir: Optional[str] = None) -> nx.Graph:
    """
    WN18RR: the WordNet lexical-relation knowledge graph (Dettmers et al.,
    2018), edge direction and the 11 relation types both dropped, via
    PyTorch Geometric. See the "Additional literature datasets" note above
    this section.
    """
    from torch_geometric.datasets import WordNet18RR

    root = str(cache_dir) if cache_dir is not None else "./data/wordnet18rr"
    data = WordNet18RR(root=root)[0]
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    G.add_edges_from(data.edge_index.numpy().T.tolist())
    return G


def _ogb_safe_globals():
    """
    Classes that need allow-listing to unpickle an OGB-cached
    ``torch_geometric`` ``Data`` object under PyTorch 2.6+'s
    ``weights_only=True`` default ``torch.load``.

    ``ogb``'s ``PygNodePropPredDataset``/``PygLinkPropPredDataset`` cache
    their processed graph as a single collated ``Data`` object, saved via
    plain ``torch.save``/pickle. Its ``_store`` is a ``GlobalStorage``, and
    the ``FeatureStore``/``GraphStore`` protocol ``Data`` implements
    attaches lists of ``DataTensorAttr``/``DataEdgeAttr`` -- all
    ``torch_geometric`` classes, none on ``torch``'s default safe-globals
    allowlist, so PyTorch 2.6's ``weights_only=True`` unpickler rejects them
    one at a time (``Unsupported global: GLOBAL
    torch_geometric.data.data.DataEdgeAttr``). Neither ``ogb`` nor (at the
    time of writing) ``torch_geometric`` have shipped a fix upstream.

    Allow-listing exactly these four classes -- rather than disabling
    ``weights_only`` altogether -- keeps the unpickler restricted to
    ``torch_geometric``'s own data-container types; it still refuses to
    execute arbitrary callables from the pickle.
    """
    from torch_geometric.data import Data
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage

    return [Data, DataEdgeAttr, DataTensorAttr, GlobalStorage]


def ogb_node_graph(name: str, cache_dir: Optional[str] = None) -> nx.Graph:
    """
    A node-property-prediction OGB dataset (e.g. ``"ogbn-arxiv"``), edge
    direction dropped, via the ``ogb`` package's PyG wrapper. See the
    "Additional literature datasets" note above this section.

    Requires the separate ``ogb`` package (``pip install ogb``; itself
    built on ``torch_geometric``, already a dependency here).
    """
    import torch
    from ogb.nodeproppred import PygNodePropPredDataset

    root = str(cache_dir) if cache_dir is not None else f"./data/{name}"
    with torch.serialization.safe_globals(_ogb_safe_globals()):
        data = PygNodePropPredDataset(name=name, root=root)[0]
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    G.add_edges_from(data.edge_index.numpy().T.tolist())
    return G


def ogb_link_graph(name: str, cache_dir: Optional[str] = None) -> nx.Graph:
    """
    A link-property-prediction OGB dataset (e.g. ``"ogbl-collab"``), via the
    ``ogb`` package's PyG wrapper. See the "Additional literature datasets"
    note above this section.

    Requires the separate ``ogb`` package (``pip install ogb``; itself
    built on ``torch_geometric``, already a dependency here).
    """
    import torch
    from ogb.linkproppred import PygLinkPropPredDataset

    root = str(cache_dir) if cache_dir is not None else f"./data/{name}"
    with torch.serialization.safe_globals(_ogb_safe_globals()):
        data = PygLinkPropPredDataset(name=name, root=root)[0]
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    G.add_edges_from(data.edge_index.numpy().T.tolist())
    return G


# ----------------------------------------------------------------------
# Dataset registry: name -> zero-arg loader returning nx.Graph (or
# (nx.Graph, labels), in which case only the graph is used here).
# ----------------------------------------------------------------------

def _dataset_loaders() -> dict:
    from experiments.datasets import (
        OFFICIAL_SETTINGS,
        airports_graph,
        balanced_tree_graph,
        openflights_graph,
        polblogs_graph,
        single_cell_graph,
    )

    loaders: dict[str, Callable[[], object]] = {
        "balanced_tree": lambda: balanced_tree_graph(),
        "karate_club": lambda: karate_club_graph(),
    }
    for name, cfg in OFFICIAL_SETTINGS.items():
        loaders[f"single_cell/{name}"] = (
            lambda name=name, cfg=cfg: single_cell_graph(
                name, k=cfg["k"], n_pca=cfg["n_pca"]
            )
        )
    loaders["polblogs"] = lambda: polblogs_graph()[0]
    for region in ("USA", "Brazil", "Europe"):
        loaders[f"airports/{region}"] = lambda region=region: airports_graph(region)[0]
    loaders["openflights"] = lambda: openflights_graph()
    loaders["wordnet/mammal"] = lambda: wordnet_noun_subtree_graph("mammal.n.01")
    for citation_name in ("Cora", "CiteSeer", "PubMed"):
        loaders[f"citation/{citation_name}"] = (
            lambda citation_name=citation_name: citation_graph(citation_name)
        )
    loaders["internet_as"] = lambda: internet_as_graph()
    loaders["arxiv_collab"] = lambda: arxiv_collaboration_graph()
    loaders["disease"] = lambda: disease_graph()
    loaders["ppi"] = lambda: ppi_graph()
    loaders["wordnet18rr"] = lambda: wordnet18rr_graph()
    loaders["ogb/arxiv"] = lambda: ogb_node_graph("ogbn-arxiv")
    loaders["ogb/collab"] = lambda: ogb_link_graph("ogbl-collab")
    loaders["ogb/ddi"] = lambda: ogb_link_graph("ogbl-ddi")
    return loaders


def run_report(
    datasets: Optional[list] = None,
    n_hyperbolicity_samples: int = 20_000,
    n_hyperbolicity_nodes: int = 5000,
    n_ricci_edges: int = 500,
    seed: int = 0,
) -> dict:
    """
    Run :func:`~experiments.geometry_diagnostics.diagnose` on every
    requested dataset (default: all of them).

    Returns ``{name: diagnose()-report}`` for datasets that loaded
    successfully, plus a separate ``_skipped: {name: reason}`` entry for
    ones that didn't (missing dependency, network/data unavailable, etc.).
    """
    loaders = _dataset_loaders()
    names = datasets if datasets is not None else list(loaders.keys())

    reports: dict = {}
    skipped: dict = {}
    for name in names:
        print(f"--- {name} ---", flush=True)
        try:
            G = loaders[name]()
            G = _largest_component(G)
            reports[name] = diagnose(
                G, n_hyperbolicity_samples=n_hyperbolicity_samples,
                n_hyperbolicity_nodes=n_hyperbolicity_nodes,
                n_ricci_edges=n_ricci_edges, seed=seed,
            )
            print(format_report(name, reports[name]))
        except Exception as exc:  # noqa: BLE001 -- keep the report going
            reason = f"{type(exc).__name__}: {exc}"
            skipped[name] = reason
            print(f"  SKIPPED ({reason})\n")

    reports["_skipped"] = skipped
    return reports


def format_summary_table(reports: dict) -> str:
    """Render one Markdown row per successfully-diagnosed dataset."""
    lines = [
        "| Dataset | N | E | mean deg | clustering | gamma | delta/diam | "
        "Ricci mean | Verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in reports.items():
        if name == "_skipped":
            continue
        dc, pl, hyp, ricci, rec = (
            r["degree_clustering"], r["powerlaw"], r["hyperbolicity"], r["ricci"],
            r["recommendation"],
        )
        lines.append(
            f"| {name} | {dc['n']} | {dc['m']} | {dc['mean_degree']:.1f} | "
            f"{dc['clustering']:.3f} | {pl['gamma']:.2f} | "
            f"{hyp['delta_normalized']:.3f} | {ricci['mean']:.3f} | "
            f"**{rec['verdict']}** ({rec['score']}/{rec['max_score']}) |"
        )
    skipped = reports.get("_skipped", {})
    if skipped:
        lines.append("")
        lines.append("Skipped: " + ", ".join(f"{k} ({v})" for k, v in skipped.items()))
    return "\n".join(lines)


if __name__ == "__main__":
    import json

    warnings.filterwarnings("ignore")
    reports = run_report()
    RESULTS.mkdir(parents=True, exist_ok=True)

    table = format_summary_table(reports)
    (RESULTS / "geometry_diagnostics.md").write_text(table + "\n")
    # `nan`/`inf` are not valid JSON; allow_nan=True (json's default) emits
    # the non-standard NaN/Infinity tokens Python's own json.load reads back
    # fine, which is all this file needs to round-trip for.
    (RESULTS / "geometry_diagnostics.json").write_text(json.dumps(reports, indent=2))

    print("\n=== SUMMARY ===")
    print(table)
