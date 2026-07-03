"""
Lorentz / hyperboloid manifold helpers.

Exposes ``LORENTZ``, a shared :class:`StableLorentz` instance — a thin
``geoopt.Lorentz`` subclass that clamps the spatial norm of every point to
``max_norm`` so leaf-heavy graphs cannot blow the optimisation up.

Why the clamp is needed
-----------------------
A point on the hyperboloid is ``x = (x_0, x')`` with ``x_0 = sqrt(1/k + ||x'||^2)``.
The exponential map moves a point by ``cosh(||v||_L) x + sinh(||v||_L) v/||v||_L``,
so a single step with a large tangent norm scales the coordinates by
``cosh(||v||_L)`` — which overflows float64 (``cosh(710) ≈ inf``) in one update
when the learning rate is high or a leaf is being pushed hard toward the
boundary. Once ``x_0`` overflows, the Lorentzian scalar product and hence every
distance becomes ``NaN``. Clamping ``||x'|| <= max_norm`` after each retraction
and projection keeps points in the numerically safe region, mirroring the
reference implementation's ``set_dim0`` renorm (Nickel & Kiela 2018; the
theSage21 PyTorch port renorms the spatial part to ``maxnorm=1e2`` with the
comment *"otherwise leaves will explode"*). At ``max_norm=1e2`` the reachable
Poincaré radius is ``100/101 ≈ 0.99``, so leaves can still sit essentially on
the boundary — the clamp only removes the overflow tail.

The clamp is a no-op for points already inside ``max_norm`` (small graphs never
reach it), and is defined here on the manifold — not inside an embedder — so
any future hyperboloid-based method can reuse the same numerically stable
geometry.
"""
import geoopt
import torch

__all__ = ["StableLorentz", "LORENTZ"]


class StableLorentz(geoopt.Lorentz):
    """
    Lorentz manifold that clamps each point's spatial norm to ``max_norm``.

    Identical to ``geoopt.Lorentz`` except that :meth:`projx` and the
    retraction (:meth:`expmap` / :meth:`retr`) additionally cap ``||x'||`` at
    ``max_norm`` and recompute the timelike coordinate. Every ``RiemannianAdam``
    step routes through ``retr_transp -> retr``, so clamping the retraction
    bounds every update, independent of the optimiser's ``stabilize`` setting.

    Parameters
    ----------
    k:
        Curvature magnitude (passed through to ``geoopt.Lorentz``).
    learnable:
        Whether ``k`` is a learnable parameter (passed through).
    max_norm:
        Maximum allowed spatial norm ``||x'||``. Default ``1e2`` follows the
        reference implementation and permits a Poincaré radius up to
        ``max_norm / (sqrt(1 + max_norm^2) + 1) ≈ 0.99``.
    """

    def __init__(self, k: float = 1.0, learnable: bool = False, max_norm: float = 1e2):
        super().__init__(k=k, learnable=learnable)
        self.max_norm = max_norm

    def _clamp_to_hyperboloid(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Project the spatial part onto ``||x'|| <= max_norm`` and set the timelike
        coordinate to ``x_0 = sqrt(1/k + ||x'||^2)`` (on-manifold, ``x_0 > 0``).
        A no-op where the spatial norm is already within ``max_norm``.

        Overflow-safe: the largest coordinate is factored out before the norm is
        formed, so a spatial part whose *squared* sum would overflow to ``inf``
        is still projected onto the boundary in its own direction rather than
        collapsing to the origin (which a naive ``max_norm / ||x'||`` does once
        ``||x'||`` overflows).
        """
        d = x.size(dim)
        spatial = x.narrow(dim, 1, d - 1)
        maxabs = spatial.abs().amax(dim=dim, keepdim=True)
        unit = spatial / maxabs.clamp_min(1e-15)                # coords in [-1, 1]
        unit_norm = unit.norm(dim=dim, keepdim=True).clamp_min(1e-15)
        norm = unit_norm * maxabs                               # true norm (may be inf)
        projected = unit / unit_norm * self.max_norm            # direction * max_norm
        spatial = torch.where(norm > self.max_norm, projected, spatial)
        time = torch.sqrt(1.0 / self.k + spatial.pow(2).sum(dim=dim, keepdim=True))
        return torch.cat([time, spatial], dim=dim)

    def projx(self, x: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
        return self._clamp_to_hyperboloid(x, dim=dim)

    def expmap(
        self, x: torch.Tensor, u: torch.Tensor, *, norm_tan=True, project=True, dim=-1
    ) -> torch.Tensor:
        res = super().expmap(x, u, norm_tan=norm_tan, project=project, dim=dim)
        # A step that overflowed to non-finite coordinates keeps the previous
        # point (as the reference implementation's RSGD does) before clamping.
        finite = torch.isfinite(res).all(dim=dim, keepdim=True)
        res = torch.where(finite, res, x)
        return self._clamp_to_hyperboloid(res, dim=dim)

    # RiemannianAdam moves points via retr_transp -> retr; keep retr == expmap
    # (as in geoopt.Lorentz) so the clamp bounds every optimiser step. This
    # relies on geoopt's internal retraction routing, not a promised public API,
    # so the geoopt dependency is upper-bounded in pyproject.toml — a new minor
    # could route around this and silently disable the guard (caught by
    # tests/test_embedders.py::test_lorentz_embeddings_stable_on_deep_tree).
    retr = expmap


# Shared, numerically stable Lorentz manifold (curvature k = 1).
LORENTZ = StableLorentz()
