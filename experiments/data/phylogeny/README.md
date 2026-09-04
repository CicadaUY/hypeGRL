# Phylogenetic trees

Clade phylogenies from the [Open Tree of Life](https://tree.opentreeoflife.org),
used as real hierarchical graphs for the chart-schedule experiments. They are
vendored rather than fetched on demand for two reasons: the synthetic trees that
first showed the effect (`caterpillar`, `balanced_tree`) are generated from a
formula and so are trivially reproducible, and these are the real graphs a result
is compared against — a number is only checkable if the graph behind it is fixed.

Each file is a whitespace-separated edge list of a single connected tree, with
nodes and edges written in sorted order. That order is load-bearing: it fixes the
row order of every matrix downstream, so an arbitrary order would make two runs of
the same experiment incomparable while looking identical.

| File | Nodes | Diameter | Radial IQR |
|---|---|---|---|
| `fabaceae_sub.edgelist` | 1067 | 44 | 12.91 |
| `carnivora.edgelist` | 1252 | 53 | 5.90 |
| `cichlidae.edgelist` | 3134 | 64 | 10.00 |
| `poaceae_sub.edgelist` | 6017 | 72 | 13.64 |
| 23 further clades | 200–5000 | | |

"Radial IQR" is the interquartile range of the node radii in a HYDRA warm start,
which measures how widely a graph spreads its nodes across radii rather than
piling them at one depth. It is the property the chart schedule responds to, and
on a tree it can be read off the eccentricity distribution in linear time without
embedding anything.

## Provenance

Fetched from the Open Tree of Life v3 API (`api.opentreeoflife.org/v3/`): a clade
name is resolved to an OTT id, and `tree_of_life/subtree` returns its induced
subtree as Newick, which is then converted to an edge list.

Names ending in `_sub` are not whole clades. The clades whose shape is most
interesting are far too large to embed, so these are rooted subtrees extracted
locally from the parent clade (`fabaceae_sub` from Fabaceae, `poaceae_sub` from
Poaceae), keeping the largest-radial-spread subtree in the 600–8000 node band.
The parent is rooted at one end of its diameter first, so the subtrees considered
are the deep, extended parts of the tree rather than shallow ones.

Species names are dropped in favour of integer labels: only the topology is used.
