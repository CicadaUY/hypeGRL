"""OpenFlights: the D-Mercator embedding in its native (r, θ) coordinates.

One polar axes: radius is the hyperbolic radius r, angle is the S¹ coordinate,
marker size grows with degree, colour and marker are the continent.

The figure makes the S¹/H² "popularity × similarity" factorisation visible in a
single frame. The radius is *inverse* popularity, so the megahubs (AMS, FRA,
CDG, LHR) sit at the smallest r near the centre, while the periodic angle
carries geography — each continent occupies its own angular sector and
geographically adjacent continents are adjacent on the circle.

The continent label is a lat/lon box rule (see ``region()``), not a country
database, so a handful of border cases land in "Other". The named airports are
a fixed curated list of world hubs rather than a global top-k by degree: the
point of the figure is the angular sectors, so every sector needs a nameable
anchor, and the top-k by degree names only Europe and North America.

Coordinates come from ``results/embeddings/openflights_dmercator_native.npz``,
which is fitted and written on first use (see ``fit_native()``) and simply
loaded afterwards. A refit is not bit-for-bit reproducible against an existing
cache: ``openflights_graph()`` downloads the live OpenFlights data files, so N
varies with the snapshot, and the β inference and angular initialisation depend
on ``SEED``.

Output: ``results/openflights_dmercator_regions.png``.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
EMB = RESULTS / "embeddings"
DMERC = EMB / "openflights_dmercator_native.npz"

SEED = 0
D_EMBED = 2       # d=2, i.e. sphere dimension D=1 — the S¹/H² setting the
                  # D-Mercator paper uses for the airport network

# fixed continent -> (colour, marker); assigned in this order, never cycled
REGIONS = {
    "Europe":     ("tab:blue",   "o"),
    "N. America": ("tab:red",    "^"),
    "Asia":       ("tab:green",  "s"),
    "S. America": ("tab:orange", "D"),
    "Africa":     ("tab:purple", "v"),
    "Oceania":    ("tab:brown",  "+"),
    "Other":      ("0.75",       "."),
}

# curated anchors: the busiest airport of each major metropolitan region, so
# that every angular sector carries at least one readable name
HUBS = [
    "AMS", "FRA", "CDG", "LHR",                              # Europe
    "ATL", "ORD", "DFW", "JFK", "LAX",                       # N. America
    "PEK", "HKG", "ICN", "NRT", "HND", "SIN", "BKK", "DEL",  # Asia
    "DXB", "DOH",                                            # Gulf
    "GRU", "EZE", "SCL", "LIM", "BOG",                       # S. America
    "CAI", "NBO", "ADD", "JNB",                              # Africa
    "SYD", "MEL",                                            # Oceania
]


def region(lat, lon):
    """Continent from a lat/lon box rule (coarse by design; ties go to 'Other')."""
    if not np.isfinite(lat) or not np.isfinite(lon):
        return "Other"
    if (lon >= 110 or lon <= -140) and lat < -5:
        return "Oceania"
    if -25 <= lon <= 45 and 34 <= lat <= 72:
        return "Europe"
    if -20 <= lon <= 55 and -40 <= lat < 34:
        return "Africa"
    if 25 <= lon <= 180 and -12 <= lat <= 82:
        return "Asia"
    if -170 <= lon <= -30 and lat >= 13:
        return "N. America"
    if -95 <= lon <= -30 and lat < 13:
        return "S. America"
    return "Other"


def fit_native():
    """Fit D-Mercator on OpenFlights and cache its native coordinates.

    Writes ``r`` (hyperbolic radius), ``kappa`` (hidden degree), ``v`` (unit
    vectors on S^D) and ``nodes`` from the embedder, alongside the ``iata`` /
    ``lat`` / ``lon`` / ``deg`` metadata needed to colour and label the figure.
    Every array is in ``emb.nodes()`` row order, which is D-Mercator's own
    ordering and not ``G.nodes()``.
    """
    from experiments.datasets import openflights_graph
    from experiments.link_prediction_experiment import _unweighted
    from hypegrl.embedders.dmercator import DMercatorEmbedder

    G = openflights_graph()
    # Topology only; the airport metadata stays on G, since _unweighted() copies
    # node ids without their attributes.
    H = _unweighted(G)
    print(f"fitting D-Mercator on OpenFlights: N={H.number_of_nodes()}, "
          f"E={H.number_of_edges()}")

    emb = DMercatorEmbedder(d=D_EMBED, d1_init="mercator", random_state=SEED)
    emb.fit(H)

    native = emb.native_coordinates()
    nodes = list(emb.nodes())
    EMB.mkdir(parents=True, exist_ok=True)
    np.savez(
        DMERC,
        r=native["r"],
        kappa=native["kappa"],
        v=native["v"],
        nodes=np.asarray(nodes),
        iata=np.array([G.nodes[n].get("iata", "") for n in nodes]),
        lat=np.array([G.nodes[n].get("lat", np.nan) for n in nodes], dtype=float),
        lon=np.array([G.nodes[n].get("lon", np.nan) for n in nodes], dtype=float),
        deg=np.array([H.degree(n) for n in nodes], dtype=float),
    )
    print("wrote", DMERC.name)


if not DMERC.exists():
    fit_native()

d = np.load(DMERC, allow_pickle=True)
deg = d["deg"].astype(float)
iata = np.array([s if s else "?" for s in d["iata"]])
r = d["r"]                                    # hyperbolic radius
th = np.arctan2(d["v"][:, 1], d["v"][:, 0])   # S¹ angle
reg = np.array([region(a, o) for a, o in zip(d["lat"], d["lon"])])
size = 4 + 55 * (deg / deg.max())

print(f"N={len(deg)}  degree {deg.min():.0f}–{deg.max():.0f}  "
      f"r {r.min():.2f}–{r.max():.2f}  "
      f"corr(r, log k) = {np.corrcoef(r, np.log(deg))[0, 1]:+.2f}")
for name in REGIONS:
    print(f"  {name:12s} {(reg == name).sum():4d}")

fig = plt.figure(figsize=(12, 11))
ax = fig.add_subplot(111, projection="polar")
for name, (color, marker) in REGIONS.items():
    m = reg == name
    if not m.any():
        continue
    ax.scatter(th[m], r[m], s=size[m], c=color, marker=marker, linewidths=0.4,
               alpha=0.75, label=f"{name} ({m.sum()})")

for code in HUBS:
    hit = np.where(iata == code)[0]
    if not hit.size:
        continue
    i = hit[np.argmax(deg[hit])]
    ax.annotate(code, (th[i], r[i]), fontsize=9, weight="bold", color="black",
                zorder=5)

ax.set_rlabel_position(22)
ax.grid(alpha=0.35)
ax.set_title("OpenFlights — D-Mercator embedding (native coordinates), "
             "coloured by continent\n"
             "(radius = hyperbolic r; hubs central, continents in angular sectors)",
             fontsize=14, pad=24)
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=10)
RESULTS.mkdir(exist_ok=True)
p = RESULTS / "openflights_dmercator_regions.png"
fig.savefig(p, dpi=120, bbox_inches="tight")
print("wrote", p)
