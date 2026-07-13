"""
Utility functions for hyperbolic embeddings.

This package provides geometric conversion utilities and other helper functions
for working with hyperbolic embeddings.
"""

from .geometric_conversions import (
    HyperbolicConversions,
    compute_distances,
    convert_coordinates,
    half_plane_to_hyperboloid,
    hemisphere_to_hyperboloid,
    hyperboloid_to_half_plane,
    hyperboloid_to_hemisphere,
    hyperboloid_to_klein,
    hyperboloid_to_poincare,
    hyperboloid_to_spherical,
    klein_to_hyperboloid,
    poincare_to_hyperboloid,
    spherical_to_hyperboloid,
    validate_embeddings,
)

__all__ = [
    "HyperbolicConversions",
    "hyperboloid_to_poincare",
    "poincare_to_hyperboloid",
    "klein_to_hyperboloid",
    "hyperboloid_to_klein",
    "hyperboloid_to_hemisphere",
    "hemisphere_to_hyperboloid",
    "hyperboloid_to_half_plane",
    "half_plane_to_hyperboloid",
    "hyperboloid_to_spherical",
    "spherical_to_hyperboloid",
    "convert_coordinates",
    "validate_embeddings",
    "compute_distances",
]
