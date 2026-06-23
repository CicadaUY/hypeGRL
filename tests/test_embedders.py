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
from hypegrl.embedders.hydra import HydraEmbedder
from hypegrl.manifolds.poincare import polar_to_poincare, poincare_distances_polar


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


# ── HydraEmbedder ──────────────────────────────────────────────────────────

@pytest.fixture
def hyperbolic_points():
    """15 random points in the Poincaré disk and their exact pairwise distances."""
    rng = np.random.default_rng(42)
    n = 15
    r_H = rng.uniform(0.2, 2.5, size=n)
    theta = rng.uniform(0, 2 * np.pi, size=n)
    X = polar_to_poincare(theta, r_H)
    r = np.linalg.norm(X, axis=1)
    directional = X / r[:, None]
    D = poincare_distances_polar(r, directional, curvature=1.0)
    return X, D


def test_hydra_fit_distance_shape(hyperbolic_points):
    _, D = hyperbolic_points
    emb = HydraEmbedder(dim=2, curvature=1.0, alpha=1.0, equi_adj=0.0)
    emb.fit_distance(D)
    assert emb.embeddings().shape == (15, 2)


def test_hydra_embeddings_inside_disk(hyperbolic_points):
    _, D = hyperbolic_points
    emb = HydraEmbedder(dim=2, curvature=1.0, alpha=1.0, equi_adj=0.0)
    emb.fit_distance(D)
    norms = np.linalg.norm(emb.embeddings(), axis=1)
    assert (norms < 1.0).all()


def test_hydra_recovers_distances_from_exact_hyperbolic_data(hyperbolic_points):
    """HYDRA recovers pairwise distances from an exact 2-D hyperbolic distance matrix.

    Points in H^2 lie on a rank-3 Gram matrix (1 timelike + 2 spatial
    eigenvalues); the spectral decomposition has zero residual, so the
    decoded distances must match the originals modulo numerical noise,
    regardless of which isometry maps the recovered coordinates to the
    original ones.
    """
    _, D = hyperbolic_points
    emb = HydraEmbedder(dim=2, curvature=1.0, alpha=1.0, equi_adj=0.0)
    emb.fit_distance(D)

    D_recovered = emb.decode(emb.embeddings())
    np.testing.assert_allclose(D_recovered, D, atol=1e-5)


def test_hydra_stress_near_zero_for_exact_data(hyperbolic_points):
    """Stress must be negligible when input distances come from a true H^2 embedding."""
    _, D = hyperbolic_points
    emb = HydraEmbedder(dim=2, curvature=1.0, alpha=1.0, equi_adj=0.0)
    emb.fit_distance(D)
    assert emb.stress < 1e-5, f"Expected near-zero stress, got {emb.stress}"


def test_hydra_fit_graph_shape(karate):
    emb = HydraEmbedder(dim=2, curvature=1.0)
    emb.fit(karate)
    assert emb.embeddings().shape == (34, 2)


def test_hydra_decode_shape(hyperbolic_points):
    _, D = hyperbolic_points
    emb = HydraEmbedder(dim=2, curvature=1.0, alpha=1.0, equi_adj=0.0)
    emb.fit_distance(D)
    D_hat = emb.decode(emb.embeddings())
    assert D_hat.shape == (15, 15)
    np.testing.assert_allclose(np.diag(D_hat), 0.0, atol=1e-10)
    np.testing.assert_allclose(D_hat, D_hat.T, atol=1e-10)


def test_hydra_unknown_edges_warns(small_graph):
    emb = HydraEmbedder(dim=2)
    with pytest.warns(UserWarning, match="non-gradient"):
        emb.fit(small_graph, unknown_edges=[(0, 1)])


def test_hydra_capability_flags():
    emb = HydraEmbedder()
    assert not emb.is_gradient_based()
    assert not emb.is_generative()
    assert not emb.supports_update()
    assert not emb.supports_node_update()


def test_hydra_raises_before_fit():
    emb = HydraEmbedder()
    with pytest.raises(RuntimeError, match="fit"):
        emb.embeddings()


def test_hydra_repr():
    emb = HydraEmbedder(dim=3, curvature=2.0)
    assert "dim=3" in repr(emb)
    assert "2.0" in repr(emb)
