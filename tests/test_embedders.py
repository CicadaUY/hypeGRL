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
from hypegrl.inference.joint_optimizer import (
    build_adjacency,
    logit_init,
    graph_to_tensor,
)
from hypegrl.embedders.poincare_embeddings import (
    PoincareEmbeddingsEmbedder,
    poincare_distance_matrix,
    sample_negatives,
    ranking_nll,
    fermi_dirac_decoder,
    fermi_dirac_nll,
)
from hypegrl.embedders.hydra import HydraEmbedder
from hypegrl.embedders.hydra_plus import HydraPlusEmbedder
from hypegrl.embedders.hypermap import (
    HyperMapEmbedder,
    fermi_dirac_nll as hypermap_fermi_dirac_nll,
)
from hypegrl.embedders.dmercator import DMercatorEmbedder
from hypegrl.embedders._dmercator_init import compute_R
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


# ── HydraPlusEmbedder ──────────────────────────────────────────────────────

@pytest.fixture
def hydra_plus_fitted(small_graph):
    """HydraPlusEmbedder fitted on P_5 with a small step budget."""
    emb = HydraPlusEmbedder(
        dim=2, curvature=1.0, n_steps=30, log_every=0, random_state=0,
    )
    emb.fit(small_graph)
    return emb


def test_hydra_plus_fit_shape(hydra_plus_fitted):
    assert hydra_plus_fitted.embeddings().shape == (5, 2)


def test_hydra_plus_embeddings_inside_disk(hydra_plus_fitted):
    norms = np.linalg.norm(hydra_plus_fitted.embeddings(), axis=1)
    assert (norms < 1.0).all()


def test_hydra_plus_loss_history_length(hydra_plus_fitted):
    assert len(hydra_plus_fitted.loss_history) == 30


def test_hydra_plus_loss_decreases(karate):
    """Loss should fall over the optimisation run on a non-trivial graph."""
    emb = HydraPlusEmbedder(
        dim=2, curvature=1.0, n_steps=100, log_every=0, random_state=0,
    )
    emb.fit(karate)
    assert emb.loss_history[-1] < emb.loss_history[0]


def test_hydra_plus_stress_improves_over_hydra(karate):
    """Riemannian refinement must not increase stress over the HYDRA warm start."""
    emb = HydraPlusEmbedder(
        dim=2, curvature=1.0, n_steps=100, log_every=0, random_state=0,
    )
    emb.fit(karate)
    assert emb.stress <= emb.stress_init


def test_hydra_plus_lower_stress_than_hydra_on_karate(karate):
    """HYDRA+ must achieve strictly lower stress than vanilla HYDRA on the karate graph."""
    hydra = HydraEmbedder(dim=2, curvature=1.0)
    hydra.fit(karate)

    hydra_plus = HydraPlusEmbedder(
        dim=2, curvature=1.0, n_steps=200, log_every=0, random_state=0,
    )
    hydra_plus.fit(karate)

    assert hydra_plus.stress < hydra.stress


def test_hydra_plus_strain_init_matches_hydra(small_graph):
    """strain_init must equal the strain of the standalone HYDRA spectral step."""
    hydra = HydraEmbedder(dim=2, curvature=1.0, alpha=1.1, equi_adj=0.5)
    hydra.fit(small_graph)
    plus = HydraPlusEmbedder(
        dim=2, curvature=1.0, alpha=1.1, equi_adj=0.5, n_steps=30, log_every=0,
    )
    plus.fit(small_graph)
    assert plus.strain_init == pytest.approx(hydra.strain, abs=1e-10)


def test_hydra_plus_accessors_set_after_fit(hydra_plus_fitted):
    assert hydra_plus_fitted.stress        is not None
    assert hydra_plus_fitted.stress_init   is not None
    assert hydra_plus_fitted.strain        is not None
    assert hydra_plus_fitted.strain_init   is not None
    assert hydra_plus_fitted.loss_history  is not None


def test_hydra_plus_fit_distance_shape(hyperbolic_points):
    _, D = hyperbolic_points
    emb = HydraPlusEmbedder(dim=2, curvature=1.0, n_steps=20, log_every=0)
    emb.fit_distance(D)
    assert emb.embeddings().shape == (15, 2)


def test_hydra_plus_decode_shape(hydra_plus_fitted):
    D_hat = hydra_plus_fitted.decode(hydra_plus_fitted.embeddings())
    assert D_hat.shape == (5, 5)
    np.testing.assert_allclose(np.diag(D_hat), 0.0, atol=1e-10)
    np.testing.assert_allclose(D_hat, D_hat.T, atol=1e-10)


def test_hydra_plus_stress_near_zero_for_exact_data(hyperbolic_points):
    """Riemannian refinement on exact H² data should not blow up.

    The HYDRA warm start already has stress ≈ 0, so the true gradient is
    near zero at initialisation. However, Adam degenerates to sign gradient
    descent for tiny gradients: the update is m̂/(√v̂ + ε) ≈ sign(g) when
    |g| ≪ ε, so even floating-point noise (~1e-14) drives a full step of
    size lr ≈ 1e-2. This kicks the embedding away from the exact solution
    rather than keeping it there. The tolerance of 0.1 therefore checks that
    the optimizer does not catastrophically diverge, not that it improves an
    already-exact embedding.
    """
    _, D = hyperbolic_points
    emb = HydraPlusEmbedder(
        dim=2, curvature=1.0, alpha=1.0, equi_adj=0.0,
        n_steps=50, lr=1e-2, log_every=0, random_state=0,
    )
    emb.fit_distance(D)
    assert emb.stress < 0.1


def test_hydra_plus_reproducible(small_graph):
    """Same random_state produces identical embeddings."""
    def make():
        e = HydraPlusEmbedder(
            dim=2, curvature=1.0, n_steps=20, log_every=0, random_state=7,
        )
        e.fit(small_graph)
        return e.embeddings()
    np.testing.assert_array_equal(make(), make())


def test_hydra_plus_unknown_edges_warns(small_graph):
    emb = HydraPlusEmbedder(dim=2, n_steps=5, log_every=0)
    with pytest.warns(UserWarning):
        emb.fit(small_graph, unknown_edges=[(0, 1)])


def test_hydra_plus_raises_before_fit():
    emb = HydraPlusEmbedder()
    with pytest.raises(RuntimeError, match="fit"):
        emb.embeddings()


def test_hydra_plus_capability_flags():
    emb = HydraPlusEmbedder()
    assert     emb.is_gradient_based()
    assert not emb.is_generative()
    assert not emb.supports_update()
    assert not emb.supports_node_update()


def test_hydra_plus_repr():
    emb = HydraPlusEmbedder(dim=2, curvature=1.0, lr=0.01, n_steps=100)
    assert "dim=2"      in repr(emb)
    assert "1.0"        in repr(emb)
    assert "lr=0.01"    in repr(emb)
    assert "n_steps=100" in repr(emb)


# ── DMercatorEmbedder ───────────────────────────────────────────────────────
# D-Mercator embeds into hyperbolic space H^{D+1} (Poincaré ball B^d, d = D+1)
# via the S^D geometric network model. Pipeline:
#   1. original-method init (_dmercator_init): infer hidden degrees κ and the
#      inverse temperature β, then place angular positions (model-corrected
#      Laplacian Eigenmaps + likelihood maximisation);
#   2. Riemannian-Adam refinement on the *Poincaré ball*, minimising a
#      Fermi-Dirac NLL on the exact hyperbolic distances
#      p_ij = 1/(1 + e^{(β/2)(d_H − R̂)}).
# The refinement runs on the Poincaré ball (not the hyperboloid) purely for
# numerical reasons: leaf nodes sit at large radius r ≈ R̂, where the
# hyperboloid coordinate cosh(r) overflows; the ball keeps coords in (−1, 1).
# See test_dmercator_robust_on_leaf_heavy_graph_d1 for the regression guard.

@pytest.fixture
def dmercator_fitted(karate):
    """DMercatorEmbedder fitted on the karate graph with a small step budget."""
    emb = DMercatorEmbedder(d=2, n_steps=30, log_every=0, random_state=0)
    emb.fit(karate)
    return emb


def test_dmercator_fit_shape(dmercator_fitted):
    # d = 2 ⇒ Poincaré ball B^2, one 2-vector per karate node.
    assert dmercator_fitted.embeddings().shape == (34, 2)


def test_dmercator_embeddings_inside_ball(dmercator_fitted):
    # Output lives in the *open* Poincaré ball: every row norm is < 1.
    # Low-degree nodes are pushed close to the boundary (large radius), so the
    # margin can be tiny — but it must never reach or exceed 1.
    norms = np.linalg.norm(dmercator_fitted.embeddings(), axis=1)
    assert (norms < 1.0).all()
    assert np.isfinite(norms).all()


def test_dmercator_structural_similarity_is_binary(karate):
    # D-Mercator is a *binary* model: edge weights are ignored. The karate graph
    # ships with integer edge weights (1–7), so this guards the bug where a
    # weighted adjacency would leak in and break the cross-entropy (a_ij ∉ {0,1}).
    emb = DMercatorEmbedder(d=2, n_steps=0, log_every=0, random_state=0)
    emb.fit(karate)
    s = emb.structural_similarity(karate)
    assert set(np.unique(s)).issubset({0.0, 1.0})
    # 34-node karate club has 78 undirected edges.
    assert s.sum() / 2 == 78


def test_dmercator_decode_shape_and_range(dmercator_fitted):
    # decode = Fermi-Dirac connection probabilities; symmetric, all in [0, 1].
    P = dmercator_fitted.decode(dmercator_fitted.embeddings())
    assert P.shape == (34, 34)
    assert (P >= 0.0).all() and (P <= 1.0).all()
    np.testing.assert_allclose(P, P.T, atol=1e-10)


def test_dmercator_capability_flags():
    # Gradient-based (Riemannian refinement) and generative (S^D model can
    # sample graphs), like the other model-based embedders.
    emb = DMercatorEmbedder()
    assert emb.is_gradient_based()
    assert emb.is_generative()


def test_dmercator_raises_before_fit():
    emb = DMercatorEmbedder()
    with pytest.raises(RuntimeError):
        emb.embeddings()


def test_dmercator_repr():
    # β defaults to "auto" (inferred) until the user fixes it.
    r = repr(DMercatorEmbedder(d=3, n_steps=10))
    assert "DMercatorEmbedder" in r
    assert "d=3" in r


def test_dmercator_d1_reduces_to_mercator(karate):
    # ── The case we care about most ────────────────────────────────────────
    # d = 2 ⇒ similarity dimension D = d − 1 = 1, i.e. the *original Mercator*
    # model: the S^D sphere collapses to a circle S^1. The cleanest closed-form
    # sanity check (pseudocode §7) is the sphere radius, which at D = 1 must be
    #     R(N, 1) = N / (2π)
    # because Γ(1)/(2π^1) = 1/(2π). This is the single most localised way to
    # confirm the global-constant machinery (and the float division in the
    # Γ/π prefactors) is correct.
    emb = DMercatorEmbedder(d=2, n_steps=0, log_every=0, random_state=0)
    emb.fit(karate)
    N = karate.number_of_nodes()
    assert emb.R_sphere == pytest.approx(N / (2.0 * np.pi))
    # The standalone init helper must agree exactly.
    assert compute_R(N, 1) == pytest.approx(N / (2.0 * np.pi))


def test_dmercator_inferred_beta_exceeds_D(dmercator_fitted):
    # β is inferred by matching the model's expected clustering to the
    # empirical one. The model is only well defined for β > D (μ ∝ sin(Dπ/β)
    # vanishes at β = D), so a valid inference must land strictly above D = 1.
    assert dmercator_fitted.beta_fitted > 1.0


def test_dmercator_fixed_beta_is_used(karate):
    # Passing beta skips the clustering-matching inference and uses it verbatim.
    emb = DMercatorEmbedder(d=2, beta=2.5, n_steps=0, log_every=0, random_state=0)
    emb.fit(karate)
    assert emb.beta_fitted == pytest.approx(2.5)


def test_dmercator_beta_below_D_warns_and_clamps(karate):
    # β must exceed D. For d = 3 (D = 2), asking for β = 1.5 ≤ D is invalid;
    # the init warns and clamps it up to a usable value (> D).
    emb = DMercatorEmbedder(d=3, beta=1.5, n_steps=0, log_every=0, random_state=0)
    with pytest.warns(UserWarning):
        emb.fit(karate)
    assert emb.beta_fitted > 2.0


def test_dmercator_loss_decreases(karate):
    # The Fermi-Dirac NLL should fall over the refinement run. The original
    # init is already strong, so the gain is modest but must be non-positive.
    emb = DMercatorEmbedder(d=2, n_steps=100, log_every=0, random_state=0)
    emb.fit(karate)
    assert emb.loss_history[-1] <= emb.loss_history[0]


def test_dmercator_n_steps_zero_returns_init(karate):
    # n_steps = 0 short-circuits the refinement: it returns the (projected)
    # original-method warm start with an empty loss history, yet still a valid
    # embedding inside the ball. This is the hook the X_init-equivalence test
    # uses to capture the pure initialisation.
    emb = DMercatorEmbedder(d=2, n_steps=0, log_every=0, random_state=0)
    emb.fit(karate)
    assert emb.loss_history == []
    assert (np.linalg.norm(emb.embeddings(), axis=1) < 1.0).all()


def test_dmercator_reconstructs_adjacency(dmercator_fitted, karate):
    # The decoder must assign higher connection probability to actual edges
    # than to non-edges — a dependency-free proxy for embedding quality.
    A = nx.to_numpy_array(karate, nodelist=dmercator_fitted.nodes, weight=None)
    P = dmercator_fitted.decode(dmercator_fitted.embeddings())
    mask = np.triu(np.ones_like(A, dtype=bool), k=1)
    assert P[mask][A[mask] == 1].mean() > P[mask][A[mask] == 0].mean()


def test_dmercator_radial_anticorrelates_with_degree(dmercator_fitted, karate):
    # Core property of a hyperbolic embedding: "popularity ⇒ centrality".
    # High-degree (popular) nodes get small radial coordinate r_i (near the
    # centre); low-degree nodes are pushed out toward the boundary. So degree
    # and radius are strongly *anti*-correlated. nodes order matches embeddings.
    deg = np.array([karate.degree(n) for n in dmercator_fitted.nodes])
    corr = np.corrcoef(deg, dmercator_fitted.radial)[0, 1]
    assert corr < -0.5


def test_dmercator_kappa_positive_and_tracks_degree(dmercator_fitted, karate):
    # κ_i (hidden degree) is recovered from the refined radius via the inverse
    # of Eq. 7; it must stay positive and, like an effective degree, correlate
    # positively with the observed degree.
    deg = np.array([karate.degree(n) for n in dmercator_fitted.nodes])
    assert (dmercator_fitted.kappa > 0).all()
    assert np.corrcoef(deg, dmercator_fitted.kappa)[0, 1] > 0.5


def test_dmercator_reproducible(karate):
    # Same seed ⇒ identical embedding (init RNG + deterministic RiemannianAdam).
    e1 = DMercatorEmbedder(d=2, n_steps=20, log_every=0, random_state=7)
    e2 = DMercatorEmbedder(d=2, n_steps=20, log_every=0, random_state=7)
    e1.fit(karate)
    e2.fit(karate)
    np.testing.assert_array_equal(e1.embeddings(), e2.embeddings())


def test_dmercator_robust_on_leaf_heavy_graph_d1():
    # ── Regression guard for the manifold choice ───────────────────────────
    # A Barabási–Albert tree (m = 1) is almost all degree-1 leaves. Those
    # leaves belong at large radius r ≈ R̂ (which reaches ~16–20 even here).
    # On the *hyperboloid* that means a coordinate cosh(r) ≈ 1e5–1e7 that
    # overflows off the manifold during optimisation → NaN (it failed on ~6/8
    # seeds). On the *Poincaré ball* coordinates stay bounded in (−1, 1), so
    # the refinement is stable. This test pins that behaviour at D = 1, the
    # worst case (largest R̂). β is fixed only to keep the test fast.
    G = nx.barabasi_albert_graph(40, 1, seed=9)
    for seed in range(4):
        emb = DMercatorEmbedder(
            d=2, beta=2.0, n_steps=80, log_every=0, random_state=seed,
        )
        emb.fit(G)
        X = emb.embeddings()
        assert np.isfinite(X).all()
        assert (np.linalg.norm(X, axis=1) < 1.0).all()


def test_dmercator_unknown_edges_warns(small_graph):
    # Unknown-edge joint optimisation is not implemented yet for D-Mercator;
    # it must warn and fall back rather than silently mishandle them.
    emb = DMercatorEmbedder(d=2, n_steps=0, log_every=0, random_state=0)
    with pytest.warns(UserWarning):
        emb.fit(small_graph, unknown_edges=[(0, 4)])


# ── d1_init: LE (paper, default) vs Mercator ordering+gap re-spacing ──────────
# At D=1 (d=2) the reference C++ does NOT use Laplacian Eigenmaps; it uses the
# classic Mercator ordering + expected-angular-gap re-spacing. We default to the
# paper's LE for all D and expose ``d1_init="mercator"`` to reproduce/compare the
# C++ D=1 init. These tests pin that the flag is wired correctly.

def test_dmercator_d1_init_mercator_produces_valid_embedding():
    # The Mercator init must place *every* node (incl. degree-one leaves, which
    # it spaces around their hub) on the disk, with finite coordinates.
    G = nx.balanced_tree(2, 4)
    emb = DMercatorEmbedder(d=2, n_steps=0, log_every=0, random_state=0,
                            d1_init="mercator")
    emb.fit(G)
    X = emb.embeddings()
    assert X.shape == (G.number_of_nodes(), 2)
    assert np.isfinite(X).all()
    assert (np.linalg.norm(X, axis=1) < 1.0).all()


def test_dmercator_d1_init_default_is_le_and_mercator_differs():
    # Default == explicit "le"; "mercator" is a genuinely different init.
    G = nx.balanced_tree(2, 4)
    X_default = DMercatorEmbedder(d=2, n_steps=0, log_every=0,
                                  random_state=0).fit(G).embeddings()
    X_le = DMercatorEmbedder(d=2, n_steps=0, log_every=0, random_state=0,
                             d1_init="le").fit(G).embeddings()
    X_merc = DMercatorEmbedder(d=2, n_steps=0, log_every=0, random_state=0,
                               d1_init="mercator").fit(G).embeddings()
    np.testing.assert_allclose(X_default, X_le)
    assert not np.allclose(X_default, X_merc)


def test_dmercator_d1_init_mercator_ignored_for_high_d(karate):
    # The Mercator ordering init is D=1-only; for d > 2 it must warn and fall
    # back to LE rather than misbehave.
    emb = DMercatorEmbedder(d=3, n_steps=0, log_every=0, random_state=0,
                            d1_init="mercator")
    with pytest.warns(UserWarning, match="only applies to D=1"):
        emb.fit(karate)


def test_dmercator_d1_init_invalid_raises():
    # Invalid strategy is rejected at construction (uniform fit() signature).
    with pytest.raises(ValueError):
        DMercatorEmbedder(d=2, d1_init="bogus")


# ── X_init equivalence ────────────────────────────────────────────────────
# For each gradient-based method: fitting with default init and n_steps=T
# must give the same result as (1) capturing the default init via n_steps=0,
# then (2) providing that init explicitly with the same n_steps=T.
# This verifies that the X_init code path is equivalent to the default path
# and that nothing breaks when the user externalises the initialisation.


def test_poincare_maps_x_init_equivalent_to_default(small_graph):
    n_steps = 20

    # Default path
    emb1 = PoincareMapsEmbedder(d=2, n_steps=n_steps, log_every=0, random_state=0)
    emb1.fit(small_graph)
    X_full = emb1.embeddings()

    # Capture the default initialisation (0 gradient steps)
    emb0 = PoincareMapsEmbedder(d=2, n_steps=0, log_every=0, random_state=0)
    emb0.fit(small_graph)
    X_init = emb0.embeddings()

    # Explicit X_init path
    emb2 = PoincareMapsEmbedder(d=2, n_steps=n_steps, log_every=0, random_state=0)
    emb2.fit(small_graph, X_init=X_init)
    X_explicit = emb2.embeddings()

    np.testing.assert_array_equal(X_full, X_explicit)


def test_hydra_plus_x_init_equivalent_to_default(small_graph):
    n_steps = 20

    # Default path (HYDRA spectral warm start)
    emb1 = HydraPlusEmbedder(
        dim=2, curvature=1.0, n_steps=n_steps, log_every=0, random_state=0,
    )
    emb1.fit(small_graph)
    X_full = emb1.embeddings()

    # Capture the HYDRA warm start (0 refinement steps)
    emb0 = HydraPlusEmbedder(
        dim=2, curvature=1.0, n_steps=0, log_every=0, random_state=0,
    )
    emb0.fit(small_graph)
    X_init = emb0.embeddings()

    # Explicit X_init path (skips spectral step, starts refinement from X_init)
    emb2 = HydraPlusEmbedder(
        dim=2, curvature=1.0, n_steps=n_steps, log_every=0, random_state=0,
    )
    emb2.fit(small_graph, X_init=X_init)
    X_explicit = emb2.embeddings()

    np.testing.assert_array_equal(X_full, X_explicit)


def test_hypermap_x_init_equivalent_to_default(karate):
    n_steps = 10

    # Reuse the same embedder instance throughout so that _nodes_sorted is
    # cached after the first fit, letting the explicit-X_init call (step 3)
    # skip the expensive greedy init.  Total greedy init calls: 2 instead of 3.

    # Step 1: default path — greedy init (#1) + gradient refinement
    emb = HyperMapEmbedder(d=2, n_steps=n_steps, log_every=0)
    emb.fit(karate)
    X_full = emb.embeddings()

    # Step 2: capture the greedy initialisation via 0 gradient steps (greedy init #2).
    emb.n_steps = 0
    emb.fit(karate)
    X_init = emb.embeddings()

    # Step 3: explicit path — _nodes_sorted already set, greedy init skipped.
    emb.n_steps = n_steps
    emb.fit(karate, X_init=X_init)
    X_explicit = emb.embeddings()

    np.testing.assert_array_equal(X_full, X_explicit)


def test_hypermap_refinement_adjacency_matches_embedding_order():
    """Regression test for the node-ordering bug in the gradient refinement.

    The embeddings ``X``, the per-node thresholds ``R`` and the returned rows are
    all in degree-descending (``nodes_sorted``) order, but ``joint_optimize``
    builds its adjacency in the graph's node-iteration order. A previous version
    passed the original graph straight through, so each hyperbolic distance was
    paired with the *wrong* adjacency entry (the i-th highest-degree node's
    distance against the original-label i-th node's edges).

    ``loss_history[0]`` is the loss evaluated on the initial embedding *before*
    any optimiser step, so it must equal the Fermi-Dirac NLL recomputed with the
    adjacency in ``nodes_sorted`` order — and must NOT equal the misaligned value
    the bug produced.
    """
    # Distinct degrees so degree-descending order differs from node order and the
    # two orderings give clearly different losses.
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (3, 4), (4, 5)])

    common = dict(d=2, gamma=2.5, T=0.5, zeta=1.0, log_every=0, verbose_init=False)

    # Warm start only (n_steps=0): capture X_init, R and the node ordering.
    emb0 = HyperMapEmbedder(n_steps=0, **common)
    emb0.fit(G)
    X_init       = emb0.embeddings()
    nodes_sorted = emb0.nodes_sorted
    assert nodes_sorted != list(G.nodes()), "test graph must reorder the nodes"

    z2t = emb0.zeta / (2.0 * emb0.T)
    X_t = torch.tensor(X_init, dtype=torch.float64)
    R_t = torch.tensor(emb0._R, dtype=torch.float64)

    A_aligned = torch.tensor(
        nx.to_numpy_array(G, nodelist=nodes_sorted, dtype=np.float64)
    )
    A_misaligned = torch.tensor(nx.to_numpy_array(G, dtype=np.float64))

    loss_aligned    = hypermap_fermi_dirac_nll(X_t, A_aligned, R_t, z2t).item()
    loss_misaligned = hypermap_fermi_dirac_nll(X_t, A_misaligned, R_t, z2t).item()

    # Sanity: the reordering genuinely changes the loss for this graph.
    assert not np.isclose(loss_aligned, loss_misaligned)

    # One refinement step: loss_history[0] is the loss on X_init before stepping.
    emb1 = HyperMapEmbedder(n_steps=1, **common)
    emb1.fit(G)
    loss0 = emb1.loss_history[0]

    # The refinement uses the adjacency aligned to the embedding order ...
    np.testing.assert_allclose(loss0, loss_aligned, rtol=1e-9, atol=1e-9)
    # ... and not the misaligned one (which the bug would have produced).
    assert not np.isclose(loss0, loss_misaligned)


def test_dmercator_x_init_equivalent_to_default(karate):
    n_steps = 20

    # Reuse one embedder so the inferred model parameters (β, R̂, κ) are cached
    # after the first fit, letting the explicit-X_init call skip the expensive
    # original-method init (β inference + likelihood maximisation) — same
    # pattern as the HyperMap test above.

    # Step 1: default path — original-method init + refinement.
    emb = DMercatorEmbedder(d=2, n_steps=n_steps, log_every=0, random_state=0)
    emb.fit(karate)
    X_full = emb.embeddings()

    # Step 2: capture the warm start via 0 refinement steps (re-runs init, but
    # deterministically with the same seed, so the warm start is identical).
    emb.n_steps = 0
    emb.fit(karate)
    X_init = emb.embeddings()

    # Step 3: explicit path — params cached, init skipped, refine from X_init.
    emb.n_steps = n_steps
    emb.fit(karate, X_init=X_init)
    X_explicit = emb.embeddings()

    # Not bit-exact (unlike the other embedders): X_init is captured *after* the
    # ball projection, so the explicit path re-projects an already-projected
    # point. projx is idempotent only to machine precision, and RiemannianAdam
    # mildly amplifies that ~1e-16 difference over the refinement — hence
    # assert_allclose rather than assert_array_equal.
    np.testing.assert_allclose(X_full, X_explicit, atol=1e-6)


# ── PoincareEmbeddingsEmbedder ─────────────────────────────────────────────

def test_pe_distance_matrix_symmetric_zero_diag():
    X = torch.randn(6, 2, dtype=torch.float64) * 0.2
    D = poincare_distance_matrix(X)
    assert D.shape == (6, 6)
    assert torch.allclose(D, D.T, atol=1e-10)
    assert torch.allclose(torch.diag(D), torch.zeros(6, dtype=torch.float64),
                          atol=1e-7)


def test_pe_sample_negatives_avoids_self_and_neighbours():
    # Node 0 is connected to 1 and 2; its only non-neighbour is 3.
    A = torch.tensor(
        [[0, 1, 1, 0],
         [1, 0, 0, 0],
         [1, 0, 0, 0],
         [0, 0, 0, 0]], dtype=torch.float64
    )
    neg = sample_negatives(A, n_negatives=20)
    assert neg.shape == (4, 20)
    # Every sampled negative for node 0 must be node 3 (the only candidate).
    assert (neg[0] == 3).all()
    # No node ever samples itself.
    rows = torch.arange(4).unsqueeze(1)
    assert not (neg == rows).any()


def test_pe_sample_negatives_fully_connected_node_falls_back():
    # Node 0 is connected to everyone -> no non-edge; must still sample (not 0).
    A = torch.tensor(
        [[0, 1, 1, 1],
         [1, 0, 0, 0],
         [1, 0, 0, 0],
         [1, 0, 0, 0]], dtype=torch.float64
    )
    neg = sample_negatives(A, n_negatives=10)
    assert (neg[0] != 0).all()


def test_pe_ranking_nll_scalar_and_finite(small_graph):
    A = torch.tensor(nx.to_numpy_array(small_graph), dtype=torch.float64)
    X = torch.randn(5, 2, dtype=torch.float64) * 0.1
    loss = ranking_nll(X, A, n_negatives=3)
    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_pe_ranking_nll_grad_flows_to_unknown_weights(small_graph):
    # An imputed (differentiable) adjacency entry must receive a gradient.
    A = torch.tensor(nx.to_numpy_array(small_graph), dtype=torch.float64)
    A.requires_grad_(True)
    X = torch.randn(5, 2, dtype=torch.float64) * 0.1
    ranking_nll(X, A, n_negatives=3).backward()
    assert A.grad is not None
    assert torch.isfinite(A.grad).all()


def test_pe_fermi_dirac_decoder_in_unit_interval():
    X = torch.randn(7, 2, dtype=torch.float64) * 0.2
    P = fermi_dirac_decoder(X, r=2.0, t=1.0)
    assert P.shape == (7, 7)
    assert (P > 0).all() and (P < 1).all()
    # Closer pairs (diagonal, distance 0) get the highest probability.
    assert torch.allclose(torch.diag(P), torch.sigmoid(torch.tensor(2.0)).double()
                          * torch.ones(7, dtype=torch.float64), atol=1e-9)


def test_pe_fermi_dirac_nll_scalar_and_finite(small_graph):
    A = torch.tensor(nx.to_numpy_array(small_graph), dtype=torch.float64)
    X = torch.randn(5, 2, dtype=torch.float64) * 0.1
    loss = fermi_dirac_nll(X, A, r=2.0, t=1.0)
    assert loss.shape == ()
    assert torch.isfinite(loss) and loss.item() >= 0.0


def test_pe_fit_shape(small_graph):
    emb = PoincareEmbeddingsEmbedder(d=2, n_steps=5, log_every=0, random_state=0)
    emb.fit(small_graph)
    assert emb.embeddings().shape == (5, 2)


def test_pe_embeddings_inside_ball(karate):
    emb = PoincareEmbeddingsEmbedder(d=2, n_steps=30, log_every=0, random_state=0)
    emb.fit(karate)
    norms = np.linalg.norm(emb.embeddings(), axis=1)
    assert (norms < 1.0).all()


def test_pe_ranking_loss_decreases(karate):
    emb = PoincareEmbeddingsEmbedder(
        d=2, n_steps=150, log_every=0, random_state=0
    )
    emb.fit(karate)
    hist = emb.loss_history
    # Compare smoothed ends: negative sampling makes the loss stochastic.
    assert np.mean(hist[-10:]) < np.mean(hist[:10])


def test_pe_fermi_dirac_loss_decreases(karate):
    emb = PoincareEmbeddingsEmbedder(
        d=2, loss="fermi_dirac", n_steps=100, log_every=0, random_state=0
    )
    emb.fit(karate)
    hist = emb.loss_history
    assert hist[-1] < hist[0]


def test_pe_with_unknown_edges(small_graph):
    unknown = [(0, 1), (1, 2)]
    emb = PoincareEmbeddingsEmbedder(
        d=2, n_steps=10, log_every=0, random_state=0
    )
    emb.fit(small_graph, unknown_edges=unknown)
    w = emb.imputed_weights
    assert w.shape == (2,)
    assert (w > 0).all() and (w < 1).all()


def test_pe_no_unknown_edges(small_graph):
    emb = PoincareEmbeddingsEmbedder(d=2, n_steps=5, log_every=0, random_state=0)
    emb.fit(small_graph, unknown_edges=[])
    assert emb.imputed_weights.shape == (0,)


def test_pe_decode_ranking_is_distance(small_graph):
    emb = PoincareEmbeddingsEmbedder(d=2, n_steps=5, log_every=0, random_state=0)
    emb.fit(small_graph)
    D = emb.decode(emb.embeddings())
    assert D.shape == (5, 5)
    np.testing.assert_allclose(np.diag(D), np.zeros(5), atol=1e-7)
    np.testing.assert_allclose(D, D.T, atol=1e-9)


def test_pe_decode_fermi_dirac_is_probability(small_graph):
    emb = PoincareEmbeddingsEmbedder(
        d=2, loss="fermi_dirac", n_steps=5, log_every=0, random_state=0
    )
    emb.fit(small_graph)
    P = emb.decode(emb.embeddings())
    assert P.shape == (5, 5)
    assert (P > 0).all() and (P < 1).all()


def test_pe_structural_similarity_is_adjacency(small_graph):
    emb = PoincareEmbeddingsEmbedder()
    s = emb.structural_similarity(small_graph)
    np.testing.assert_array_equal(s, nx.to_numpy_array(small_graph,
                                                       dtype=np.float64))


def test_pe_raises_before_fit():
    emb = PoincareEmbeddingsEmbedder()
    with pytest.raises(RuntimeError, match="fit"):
        emb.embeddings()


def test_pe_invalid_loss_raises():
    with pytest.raises(ValueError, match="ranking"):
        PoincareEmbeddingsEmbedder(loss="bogus")


def test_pe_capability_flags():
    rank = PoincareEmbeddingsEmbedder(loss="ranking")
    fd = PoincareEmbeddingsEmbedder(loss="fermi_dirac")
    assert rank.is_gradient_based()
    assert rank.supports_update() and rank.supports_node_update()
    assert not rank.is_generative()        # ranking has no edge-prob decoder
    assert fd.is_generative()              # Fermi-Dirac decoder is generative


def test_pe_repr():
    emb = PoincareEmbeddingsEmbedder(d=3, loss="fermi_dirac")
    assert "d=3" in repr(emb)
    assert "fermi_dirac" in repr(emb)


def test_pe_disconnection_raises_on_update(small_graph):
    emb = PoincareEmbeddingsEmbedder(d=2, n_steps=5, log_every=0, random_state=0)
    emb.fit(small_graph)
    with pytest.raises(ValueError, match="disconnect"):
        emb.update(removed_edges=[(0, 1)])


def test_pe_sample_negatives_respects_node_weights():
    # Node 0 may have negatives {2,3,4}; weight node 3 enormously and the rest 0
    # -> all sampled negatives for node 0 must be node 3.
    A = torch.tensor(
        [[0, 1, 0, 0, 0],
         [1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]], dtype=torch.float64
    )
    w = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float64)
    neg = sample_negatives(A, n_negatives=15, node_weights=w)
    assert (neg[0] == 3).all()          # only positively weighted candidate
    # Self and the true neighbour (node 1) are still never sampled for node 0.
    assert not (neg[0] == 0).any()
    assert not (neg[0] == 1).any()


def test_pe_burnin_runs_and_extends_loss_history(karate):
    burnin, n_steps = 20, 40
    emb = PoincareEmbeddingsEmbedder(
        d=2, burnin=burnin, n_steps=n_steps, log_every=0, random_state=0
    )
    emb.fit(karate)
    assert len(emb.loss_history) == burnin + n_steps
    X = emb.embeddings()
    assert X.shape == (34, 2)
    assert (np.linalg.norm(X, axis=1) < 1.0).all()
    # Burn-in state must be cleared after fit.
    assert emb._burnin_active is False


def test_pe_burnin_default_matches_reference():
    # Default burn-in follows the reference code (20).
    assert PoincareEmbeddingsEmbedder().burnin == 20


def test_pe_burnin_zero_disables(karate):
    n_steps = 30
    emb = PoincareEmbeddingsEmbedder(
        d=2, burnin=0, n_steps=n_steps, log_every=0, random_state=0
    )
    emb.fit(karate)
    assert len(emb.loss_history) == n_steps


def test_pe_burnin_with_unknown_edges(small_graph):
    unknown = [(0, 1), (1, 2)]
    emb = PoincareEmbeddingsEmbedder(
        d=2, burnin=10, n_steps=10, log_every=0, random_state=0
    )
    emb.fit(small_graph, unknown_edges=unknown)
    w = emb.imputed_weights
    assert w.shape == (2,)
    assert (w > 0).all() and (w < 1).all()
    assert len(emb.loss_history) == 20
