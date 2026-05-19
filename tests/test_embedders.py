"""Tests for embedding methods."""
import numpy as np
import networkx as nx
import pytest
import torch

from hypegrl.embedders.poincare_maps import (
    PoincareMapsEmbedder,
    forest_matrix,
    soft_decoder,
    symkl_loss_fn,
)
from hypegrl.unknown_edges.joint_optimizer import (
    build_adjacency,
    logit_init,
    graph_to_tensor,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def small_graph():
    """Path graph P_5: simple, connected, no weights."""
    return nx.path_graph(5)


@pytest.fixture
def karate():
    return nx.karate_club_graph()


# ── forest_matrix ─────────────────────────────────────────────────────────

def test_forest_matrix_shape(small_graph):
    A = torch.tensor(nx.to_numpy_array(small_graph), dtype=torch.float64)
    Q = forest_matrix(A)
    assert Q.shape == (5, 5)


def test_forest_matrix_symmetric(small_graph):
    A = torch.tensor(nx.to_numpy_array(small_graph), dtype=torch.float64)
    Q = forest_matrix(A)
    assert torch.allclose(Q, Q.T, atol=1e-10)


def test_forest_matrix_positive_definite(small_graph):
    A = torch.tensor(nx.to_numpy_array(small_graph), dtype=torch.float64)
    Q = forest_matrix(A)
    eigvals = torch.linalg.eigvalsh(Q)
    assert (eigvals > 0).all(), "Forest matrix must be positive definite"


def test_forest_matrix_inverse_identity(small_graph):
    """Q = (I+L)^{-1} implies Q^{-1} = I+L."""
    A = torch.tensor(nx.to_numpy_array(small_graph), dtype=torch.float64)
    D = torch.diag(A.sum(dim=1))
    L = D - A
    I = torch.eye(5, dtype=torch.float64)
    Q = forest_matrix(A)
    assert torch.allclose(Q @ (I + L), I, atol=1e-9)


# ── soft_decoder ──────────────────────────────────────────────────────────

def test_soft_decoder_row_stochastic():
    X = torch.randn(6, 2, dtype=torch.float64) * 0.2
    A_hat = soft_decoder(X, gamma=1.0)
    row_sums = A_hat.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(6, dtype=torch.float64), atol=1e-9)


def test_soft_decoder_shape():
    X = torch.randn(8, 2, dtype=torch.float64) * 0.2
    A_hat = soft_decoder(X)
    assert A_hat.shape == (8, 8)


def test_soft_decoder_nonnegative():
    X = torch.randn(5, 2, dtype=torch.float64) * 0.2
    A_hat = soft_decoder(X)
    assert (A_hat >= 0).all()


# ── logit_init ────────────────────────────────────────────────────────────

def test_logit_init_roundtrip():
    vals = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    recovered = torch.sigmoid(torch.tensor(logit_init(vals))).numpy()
    np.testing.assert_allclose(recovered, vals, atol=1e-6)


def test_logit_init_clips_boundary():
    vals = np.array([0.0, 1.0])  # boundary values
    result = logit_init(vals)    # should not raise
    assert np.all(np.isfinite(result))


# ── build_adjacency ───────────────────────────────────────────────────────

def test_build_adjacency_symmetric():
    A_known = torch.zeros(4, 4, dtype=torch.float64)
    unknown_edges = [(0, 1), (2, 3)]
    a_omega = torch.tensor([0.6, 0.4], dtype=torch.float64)
    A = build_adjacency(A_known, unknown_edges, a_omega)
    assert A[0, 1] == pytest.approx(0.6)
    assert A[1, 0] == pytest.approx(0.6)
    assert A[2, 3] == pytest.approx(0.4)
    assert A[3, 2] == pytest.approx(0.4)
    assert torch.allclose(A, A.T)


# ── PoincareMapsEmbedder ──────────────────────────────────────────────────

def test_embedder_fit_shape(small_graph):
    emb = PoincareMapsEmbedder(d=2, n_steps=5, log_every=0)
    emb.fit(small_graph)
    X = emb.embeddings()
    assert X.shape == (5, 2)


def test_embedder_embeddings_inside_disk(small_graph):
    emb = PoincareMapsEmbedder(d=2, n_steps=10, log_every=0)
    emb.fit(small_graph)
    X = emb.embeddings()
    norms = np.linalg.norm(X, axis=1)
    assert (norms < 1.0).all(), "All embeddings must lie inside the Poincare disk"


def test_embedder_loss_decreases(small_graph):
    emb = PoincareMapsEmbedder(d=2, n_steps=50, log_every=0, random_state=0)
    emb.fit(small_graph)
    hist = emb.loss_history
    # Loss at end should be lower than at start (not guaranteed every step,
    # but should hold over 50 steps from a random init)
    assert hist[-1] < hist[0], "Loss should decrease over training"


def test_embedder_with_unknown_edges(small_graph):
    unknown = [(0, 1), (1, 2)]
    emb = PoincareMapsEmbedder(d=2, n_steps=10, log_every=0, random_state=0)
    emb.fit(small_graph, unknown_edges=unknown)
    assert emb.imputed_weights is not None
    assert emb.imputed_weights.shape == (2,)
    # Imputed weights must be in (0,1) due to sigmoid reparametrisation
    assert (emb.imputed_weights > 0).all()
    assert (emb.imputed_weights < 1).all()


def test_embedder_no_unknown_edges(small_graph):
    emb = PoincareMapsEmbedder(d=2, n_steps=5, log_every=0)
    emb.fit(small_graph, unknown_edges=[])
    assert emb.imputed_weights.shape == (0,)


def test_embedder_raises_before_fit():
    emb = PoincareMapsEmbedder()
    with pytest.raises(RuntimeError, match="fit"):
        emb.embeddings()


def test_embedder_decode_shape(small_graph):
    emb = PoincareMapsEmbedder(d=2, n_steps=5, log_every=0)
    emb.fit(small_graph)
    A_hat = emb.decode(emb.embeddings())
    assert A_hat.shape == (5, 5)
    # Row-stochastic
    np.testing.assert_allclose(A_hat.sum(axis=1), np.ones(5), atol=1e-6)


def test_embedder_structural_similarity_shape(small_graph):
    emb = PoincareMapsEmbedder(d=2, n_steps=5, log_every=0)
    Q = emb.structural_similarity(small_graph)
    assert Q.shape == (5, 5)


def test_embedder_repr():
    emb = PoincareMapsEmbedder(d=3, gamma=0.5)
    assert "d=3" in repr(emb)
    assert "gamma=0.5" in repr(emb)


def test_embedder_capability_flags():
    emb = PoincareMapsEmbedder()
    assert emb.is_gradient_based()
    assert emb.is_generative()
    assert emb.supports_update()
    assert emb.supports_node_update()


def test_disconnection_raises_on_update(small_graph):
    emb = PoincareMapsEmbedder(d=2, n_steps=5, log_every=0)
    emb.fit(small_graph)
    # Removing an edge from a path graph disconnects it
    with pytest.raises(ValueError, match="disconnect"):
        emb.update(removed_edges=[(0, 1)])
