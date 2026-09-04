# -*- coding: utf-8 -*-
"""
Every gradient embedder against every chart, and the ``representation_kwargs``
plumbing that carries chart options through them.

The chart is meant to be an axis orthogonal to the method, but until this
existed only two embedders were ever fitted in a non-default chart, so a break
in the shared plumbing could reach the other four unnoticed — which is exactly
what happened once: an embedder handing a chart-specific numerical clamp to
whatever chart was selected turned every non-hyperboloid chart on
:class:`LorentzEmbeddingsEmbedder` into a ``TypeError``. That clamp now lives
where it belongs, in ``representation_kwargs``, so every chart option has a
single source and a single rule. The matrix below is the guard.

Budgets are deliberately tiny: this asserts that a combination *runs and
produces finite coordinates*, not that it embeds well.
"""

import networkx as nx
import numpy as np
import pytest

from hypegrl.embedders.dmercator import DMercatorEmbedder
from hypegrl.embedders.hydra_plus import HydraPlusEmbedder
from hypegrl.embedders.hypermap import HyperMapEmbedder
from hypegrl.embedders.lorentz_embeddings import LorentzEmbeddingsEmbedder
from hypegrl.embedders.poincare_embeddings import PoincareEmbeddingsEmbedder
from hypegrl.embedders.poincare_maps import PoincareMapsEmbedder
from hypegrl.manifolds.lorentz import StableLorentz
from hypegrl.representations import (
    BallRepresentation,
    CurvedPolarRepresentation,
    ExactPolarRepresentation,
    HyperboloidRepresentation,
    PolarRepresentation,
    TangentRepresentation,
    representation_options,
)

CHARTS = ["polar", "exact_polar", "curved_polar", "tangent", "ball", "hyperboloid"]

EMBEDDERS = {
    "PoincareMaps": lambda c, **kw: PoincareMapsEmbedder(
        d=2, n_steps=3, log_every=0, random_state=0, representation=c, **kw),
    "HydraPlus": lambda c, **kw: HydraPlusEmbedder(
        dim=2, n_steps=3, log_every=0, random_state=0, representation=c, **kw),
    "HyperMap": lambda c, **kw: HyperMapEmbedder(
        d=2, n_steps=3, log_every=0, representation=c, **kw),
    "DMercator": lambda c, **kw: DMercatorEmbedder(
        d=2, n_steps=3, log_every=0, random_state=0, representation=c, **kw),
    "PoincareEmbeddings": lambda c, **kw: PoincareEmbeddingsEmbedder(
        d=2, n_steps=3, log_every=0, random_state=0, representation=c, **kw),
    "LorentzEmbeddings": lambda c, **kw: LorentzEmbeddingsEmbedder(
        d=2, n_steps=3, log_every=0, random_state=0, representation=c, **kw),
}

# LorentzEmbeddings in the ball chart trips a CheckpointError inside the
# gradient-checkpointed chunked ranking loss, and only from a cold process — it
# passes once another embedder has run first. Recorded rather than hidden;
# non-strict so the order-dependence does not itself fail the suite.
ORDER_DEPENDENT = {("LorentzEmbeddings", "ball")}


@pytest.fixture(scope="module")
def graph():
    return nx.karate_club_graph()


@pytest.mark.parametrize("chart", CHARTS)
@pytest.mark.parametrize("name", sorted(EMBEDDERS))
def test_every_embedder_fits_in_every_chart(request, graph, name, chart):
    if (name, chart) in ORDER_DEPENDENT:
        request.node.add_marker(pytest.mark.xfail(
            reason="checkpoint recompute mismatch in the Lorentz ranking loss "
                   "(pre-existing, order-dependent)", strict=False))
    X = EMBEDDERS[name](chart).fit(graph).embeddings()
    assert X.shape == (graph.number_of_nodes(), 2)
    assert np.isfinite(X).all()


@pytest.mark.parametrize("name", sorted(EMBEDDERS))
def test_chart_options_reach_the_chart(graph, name):
    """``representation_kwargs`` is the route from an embedder to its chart."""
    emb = EMBEDDERS[name]("curved_polar",
                          representation_kwargs={"chart_curvature": 0.09})
    emb.fit(graph)
    assert emb.embeddings_representation()._manifold.chart_curvature == pytest.approx(
        0.09)


@pytest.mark.parametrize("name", sorted(EMBEDDERS))
def test_an_option_the_selected_chart_ignores_is_rejected(graph, name):
    """
    Silently dropping it would run the fit with a default the caller thought
    they had replaced, so it is an error naming the chart instead.
    """
    emb = EMBEDDERS[name]("polar", representation_kwargs={"chart_curvature": 0.09})
    with pytest.raises(TypeError, match="chart_curvature"):
        emb.fit(graph)


# ---------------------------------------------------- the library's own options


def test_max_norm_is_a_chart_option_not_an_embedder_one(graph):
    """
    ``max_norm`` acts through the hyperboloid's ``projx``/retraction, so it
    belongs to that chart and travels in ``representation_kwargs``. The
    embedder does not take it, which is what keeps every chart's options in one
    place and the check on them uniform.
    """
    emb = LorentzEmbeddingsEmbedder(
        d=2, n_steps=3, log_every=0, random_state=0, representation="hyperboloid",
        representation_kwargs={"max_norm": 42.0})
    emb.fit(graph)
    assert emb.embeddings_representation()._manifold.max_norm == 42.0

    with pytest.raises(TypeError, match="max_norm"):
        LorentzEmbeddingsEmbedder(
            d=2, n_steps=3, log_every=0, random_state=0, max_norm=42.0)


def test_max_norm_asked_of_a_chart_without_one_is_rejected(graph):
    """No silent drop: a chart with no such coordinate says so."""
    emb = LorentzEmbeddingsEmbedder(
        d=2, n_steps=3, log_every=0, random_state=0, representation="tangent",
        representation_kwargs={"max_norm": 42.0})
    with pytest.raises(TypeError, match="max_norm"):
        emb.fit(graph)


def test_representation_options_reports_what_each_chart_understands():
    """The signature is the single source of truth for a chart's options."""
    assert representation_options(PolarRepresentation) == {"device"}
    assert representation_options(BallRepresentation) == {"device"}
    assert representation_options(TangentRepresentation) == {"device"}
    assert representation_options(HyperboloidRepresentation) == {"device", "max_norm"}
    assert representation_options(ExactPolarRepresentation) == {"device", "max_step"}
    assert representation_options(CurvedPolarRepresentation) == {
        "device", "max_step", "chart_curvature"}


def test_hyperboloid_chart_builds_its_own_manifold_instance():
    """``max_norm`` is per-fit state, so it must not mutate the shared manifold."""
    shared_default = StableLorentz().max_norm
    rep = HyperboloidRepresentation.from_polar(
        np.array([1.0]), np.array([[1.0, 0.0]]), max_norm=13.0)
    assert rep._manifold.max_norm == 13.0
    assert StableLorentz().max_norm == shared_default
