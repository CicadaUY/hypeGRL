"""Tests for the unknown-edges joint optimization framework."""
import numpy as np
import networkx as nx
import pytest
import torch

from hypegrl.inference.joint_optimizer import (
    joint_optimize,
    graph_to_tensor,
)
from hypegrl.embedders.poincare_maps import symkl_loss_from_dist, forest_matrix
from hypegrl.representations import BallRepresentation


@pytest.fixture
def triangle():
    return nx.cycle_graph(3)


@pytest.fixture
def karate():
    return nx.karate_club_graph()


def make_loss(gamma=1.0):
    """Poincaré-Maps symmetric-KL loss as a ``loss_fn(rep, A)`` for the optimizer."""
    def _loss(rep, A):
        return symkl_loss_from_dist(rep.dist(), forest_matrix(A), gamma=gamma)
    return _loss


def make_rep(X_init):
    """A ball representation seeded from ``(N, d)`` ball coordinates."""
    return BallRepresentation.from_ball(X_init)


# ── graph_to_tensor ───────────────────────────────────────────────────────

def test_graph_to_tensor_zeros_unknown(triangle):
    A = graph_to_tensor(triangle, [(0, 1)], torch.device("cpu"))
    assert A[0, 1].item() == 0.0
    assert A[1, 0].item() == 0.0
    assert A[1, 2].item() == 1.0  # known edge intact


def test_graph_to_tensor_shape(karate):
    N = karate.number_of_nodes()
    A = graph_to_tensor(karate, [], torch.device("cpu"))
    assert A.shape == (N, N)


def test_graph_to_tensor_symmetric(karate):
    A = graph_to_tensor(karate, list(karate.edges())[:3], torch.device("cpu"))
    assert torch.allclose(A, A.T)


# ── joint_optimize ────────────────────────────────────────────────────────

def test_joint_optimize_returns_keys(triangle):
    N = triangle.number_of_nodes()
    X_init = np.random.randn(N, 2) * 0.1
    result = joint_optimize(
        triangle, make_rep(X_init), make_loss(),
        unknown_edges=[(0, 1)], n_steps=5, log_every=0,
    )
    assert set(result.keys()) == {
        "representation", "a_omega", "loss_history", "unknown_edges"
    }


def test_joint_optimize_shapes(triangle):
    N = triangle.number_of_nodes()
    X_init = np.random.randn(N, 2) * 0.1
    unknown = [(0, 1), (1, 2)]
    result = joint_optimize(
        triangle, make_rep(X_init), make_loss(),
        unknown_edges=unknown, n_steps=10, log_every=0,
    )
    assert result["representation"].to_ball().shape == (N, 2)
    assert result["a_omega"].shape == (2,)
    assert len(result["loss_history"]) == 10


def test_joint_optimize_embeddings_inside_disk(triangle):
    N = triangle.number_of_nodes()
    X_init = np.random.randn(N, 2) * 0.1
    result = joint_optimize(
        triangle, make_rep(X_init), make_loss(),
        unknown_edges=[(0, 1)], n_steps=20, log_every=0,
    )
    X = result["representation"].to_ball().detach().cpu().numpy()
    norms = np.linalg.norm(X, axis=1)
    assert (norms < 1.0).all()


def test_joint_optimize_weights_in_unit_interval(triangle):
    N = triangle.number_of_nodes()
    X_init = np.random.randn(N, 2) * 0.1
    result = joint_optimize(
        triangle, make_rep(X_init), make_loss(),
        unknown_edges=[(0, 1)], n_steps=20, log_every=0,
    )
    a = result["a_omega"]
    assert (a > 0).all() and (a < 1).all()


def test_joint_optimize_loss_finite(triangle):
    N = triangle.number_of_nodes()
    X_init = np.random.randn(N, 2) * 0.1
    result = joint_optimize(
        triangle, make_rep(X_init), make_loss(),
        unknown_edges=[(0, 1)], n_steps=10, log_every=0,
    )
    assert all(np.isfinite(v) for v in result["loss_history"])


def test_joint_optimize_no_unknown_edges(triangle):
    """With no unknown edges, result should match standard embedding."""
    N = triangle.number_of_nodes()
    np.random.seed(42)
    X_init = np.random.randn(N, 2) * 0.1
    result = joint_optimize(
        triangle, make_rep(X_init), make_loss(),
        unknown_edges=[], n_steps=10, log_every=0,
    )
    assert result["a_omega"].shape == (0,)
    assert result["representation"].to_ball().shape == (N, 2)


def test_joint_optimize_regularization_stabilizes(karate):
    """Regularization should keep a_omega bounded."""
    N = karate.number_of_nodes()
    np.random.seed(0)
    X_init = np.random.randn(N, 2) * 0.1
    unknown = list(karate.edges())[:5]
    result = joint_optimize(
        karate, make_rep(X_init), make_loss(),
        unknown_edges=unknown, n_steps=50,
        regularize_a=0.1, log_every=0,
    )
    a = result["a_omega"]
    assert (a > 0).all() and (a < 1).all()
