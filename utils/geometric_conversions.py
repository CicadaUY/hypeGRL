"""
Geometric conversion utilities for hyperbolic spaces.

This module provides functions to convert between different models of hyperbolic geometry:
- Poincaré ball model
- Hyperboloid model
- Klein model
- Hemisphere model
- Half-plane model
- Spherical coordinates

All functions support both single points and batches of points through numpy broadcasting.
"""

import numpy as np


class HyperbolicConversions:
    """Utility class for converting between different hyperbolic geometry models."""

    SUPPORTED_SPACES = ["poincare", "hyperboloid", "klein", "hemisphere", "half_plane", "spherical"]

    @staticmethod
    def hyperboloid_to_poincare(x: np.ndarray) -> np.ndarray:
        """
        Convert coordinates from the N-dimensional hyperboloid model to the N-dimensional Poincaré ball model.

        Parameters:
        - x: Coordinates in the N+1-dimensional hyperboloid model (R^(N+1))

        Returns:
        - y: Coordinates in the N-dimensional Poincaré ball model (R^N)
        """

        # Convert to Poincaré ball coordinates
        # y[i] = x[i] / (1 + t) where t = x[n] (last coordinate)
        t = x[..., -1]  # Get the last coordinate (time-like coordinate)
        y = x[..., :-1] / (1 + t[..., np.newaxis])  # Apply conversion to spatial coordinates

        return y

    @staticmethod
    def poincare_to_hyperboloid(y: np.ndarray) -> np.ndarray:
        """
        Convert coordinates from the N-dimensional Poincaré ball model to the N-dimensional hyperboloid model.

        Parameters:
        - y: Coordinates in the N-dimensional Poincaré ball model (R^N)

        Returns:
        - x: Coordinates in the N-dimensional hyperboloid model (R^(N+1))
        """

        # Convert to hyperboloid coordinates
        # t = (1 + ||y||^2) / (1 - ||y||^2)
        # x_i = (2y_i) / (1 - ||y||^2)

        # Calculate the squared L2 norm of y
        y_norm_sq = np.sum(y**2, axis=-1, keepdims=True)

        # Calculate t (time-like coordinate)
        t = (1 + y_norm_sq) / (1 - y_norm_sq)

        # Calculate spatial coordinates x_i
        x_spatial = (2 * y) / (1 - y_norm_sq)

        # Combine spatial coordinates with time coordinate
        x = np.concatenate([x_spatial, t], axis=-1)

        return x

    @staticmethod
    def klein_to_hyperboloid(y: np.ndarray) -> np.ndarray:
        """
        Convert coordinates from the N-dimensional Klein model to the N-dimensional hyperboloid model.

        Parameters:
        - y: Coordinates in the N-dimensional Klein model (R^N)

        Returns:
        - x: Coordinates in the N-dimensional hyperboloid model (R^(N+1))
        """

        # Convert to hyperboloid coordinates
        # t = 1 / sqrt(1 - ||y||^2)
        # x_i = y_i / sqrt(1 - ||y||^2)

        # Calculate the squared L2 norm of y
        y_norm_sq = np.sum(y**2, axis=-1, keepdims=True)

        # Calculate t (time-like coordinate)
        t = 1 / np.sqrt(1 - y_norm_sq)

        # Calculate spatial coordinates x_i
        x_spatial = y / np.sqrt(1 - y_norm_sq)

        # Combine spatial coordinates with time coordinate
        x = np.concatenate([x_spatial, t], axis=-1)

        return x

    @staticmethod
    def hyperboloid_to_klein(x: np.ndarray) -> np.ndarray:
        """
        Convert coordinates from the N-dimensional hyperboloid model to the N-dimensional Klein model.

        Parameters:
        - x: Coordinates in the N-dimensional hyperboloid model (R^(N+1))

        Returns:
        - y: Coordinates in the N-dimensional Klein model (R^N)
        """

        # Convert to Klein coordinates
        # y_i = x_i / t where t is the last coordinate (time-like)

        t = x[..., -1]  # Get the last coordinate (time-like coordinate)
        y = x[..., :-1] / t[..., np.newaxis]  # Apply conversion to spatial coordinates

        return y

    @staticmethod
    def hyperboloid_to_hemisphere(x: np.ndarray) -> np.ndarray:
        """
        Convert coordinates from the N-dimensional hyperboloid model to the N-dimensional hemisphere model.

        Parameters:
        - x: Coordinates in the N-dimensional hyperboloid model (R^(N+1))

        Returns:
        - y: Coordinates in the N-dimensional hemisphere model (R^(N+1))
        """

        # Convert to hemisphere coordinates
        # z = 1/t
        # y_i = x_i / t

        t = x[..., -1]  # Get the last coordinate (time-like coordinate)

        # Calculate z (last coordinate of hemisphere)
        z = 1 / t

        # Calculate spatial coordinates y_i
        y_spatial = x[..., :-1] / t[..., np.newaxis]

        # Combine spatial coordinates with z coordinate
        y = np.concatenate([y_spatial, z[..., np.newaxis]], axis=-1)

        return y

    @staticmethod
    def hemisphere_to_hyperboloid(y: np.ndarray) -> np.ndarray:
        """
        Convert coordinates from the N-dimensional hemisphere model to the N-dimensional hyperboloid model.

        Parameters:
        - y: Coordinates in the N-dimensional hemisphere model (R^(N+1))

        Returns:
        - x: Coordinates in the N-dimensional hyperboloid model (R^(N+1))
        """

        # Convert to hyperboloid coordinates
        # t = 1/z
        # x_i = y_i / z

        z = y[..., -1]  # Get the last coordinate (z coordinate)

        # Calculate t (time-like coordinate)
        t = 1 / z

        # Calculate spatial coordinates x_i
        x_spatial = y[..., :-1] / z[..., np.newaxis]

        # Combine spatial coordinates with time coordinate
        x = np.concatenate([x_spatial, t[..., np.newaxis]], axis=-1)

        return x

    @staticmethod
    def hyperboloid_to_half_plane(x: np.ndarray) -> np.ndarray:
        """
        Convert coordinates from the N-dimensional hyperboloid model to the N-dimensional half-plane model.

        Parameters:
        - x: Coordinates in the N-dimensional hyperboloid model (R^(N+1))

        Returns:
        - y: Coordinates in the N-dimensional half-plane model (R^N)
        """
        # Convert to half-plane coordinates
        # y_i = (2x_i) / (x_1 + t) for i >= 2
        # z = 2 / (x_1 + t)

        x1 = x[..., 0]  # First spatial coordinate
        t = x[..., -1]  # Time-like coordinate
        denominator = x1 + t

        # Calculate spatial coordinates y_i (starting from index 1, so y_2, y_3, ...)
        y_spatial = (2 * x[..., 1:]) / denominator[..., np.newaxis]

        # Calculate z (last coordinate)
        z = 2 / denominator

        # Combine spatial coordinates with z coordinate
        y = np.concatenate([y_spatial, z[..., np.newaxis]], axis=-1)

        return y

    @staticmethod
    def half_plane_to_hyperboloid(y: np.ndarray) -> np.ndarray:
        """
        Convert coordinates from the N-dimensional half-plane model to the N-dimensional hyperboloid model.

        Parameters:
        - y: Coordinates in the N-dimensional half-plane model (R^N)

        Returns:
        - x: Coordinates in the N-dimensional hyperboloid model (R^(N+1))
        """
        # Convert to hyperboloid coordinates
        # x_1 = (4 - z^2 - Σ_{i>=2} y_i^2) / (4z)
        # x_i = y_i / z for i >= 2
        # t = (4 + z^2 + Σ_{i>=2} y_i^2) / (4z)

        z = y[..., -1]  # Last coordinate (z)
        y_spatial = y[..., :-1]  # Spatial coordinates (y_2, y_3, ...)

        # Calculate sum of squares of spatial coordinates
        y_sum_sq = np.sum(y_spatial**2, axis=-1)

        # Calculate x_1 (first spatial coordinate)
        x1 = (4 - z**2 - y_sum_sq) / (4 * z)

        # Calculate remaining spatial coordinates x_i
        x_spatial = y_spatial / z[..., np.newaxis]

        # Calculate t (time-like coordinate)
        t = (4 + z**2 + y_sum_sq) / (4 * z)

        # Combine all coordinates: (x_1, x_2, ..., x_n, t)
        x = np.concatenate([x1[..., np.newaxis], x_spatial, t[..., np.newaxis]], axis=-1)

        return x

    @staticmethod
    def hyperboloid_to_spherical(x: np.ndarray) -> np.ndarray:
        """
        Convert coordinates from the N-dimensional hyperboloid model to spherical coordinates.

        Parameters:
        - x: Coordinates in the N-dimensional hyperboloid model (R^(N+1))

        Returns:
        - spherical: Spherical coordinates (r, θ_1, ..., θ_{n-1})
        """
        # Convert to spherical coordinates
        # r = acosh(t)
        # For 2D: θ = atan2(x_2, x_1) to get full circle [0, 2π]
        # For higher dimensions:
        # θ_1 = acos(x_1 / sinh r)
        # θ_i = acos(x_i / (sinh r * Π_{j=1}^{i-1} sin θ_j)) for i = 2, ..., n-1
        # θ_n = atan2(x_n, x_{n-1})

        t = x[..., -1]  # Time-like coordinate
        x_spatial = x[..., :-1]  # Spatial coordinates

        # Calculate r (hyperbolic radius)
        r = np.arccosh(t)

        # Special case for 2D: use atan2 directly for full circle
        if x_spatial.shape[-1] == 2:
            # For 2D, use atan2 to get angle in [-π, π], then convert to [0, 2π]
            theta = np.arctan2(x_spatial[..., 1], x_spatial[..., 0])
            # Convert to [0, 2π] range
            theta = np.where(theta < 0, theta + 2 * np.pi, theta)
            spherical = np.stack([r, theta], axis=-1)
        else:
            # General n-dimensional case
            angles = []
            current_radius = np.sinh(r)

            for i in range(len(x_spatial.shape) - 1):
                if i == 0:
                    # First angle
                    theta = np.arccos(np.clip(x_spatial[..., 0] / current_radius, -1, 1))
                else:
                    # Subsequent angles
                    theta = np.arccos(np.clip(x_spatial[..., i] / current_radius, -1, 1))
                    current_radius = current_radius * np.sin(angles[-1])
                angles.append(theta)

            # Last angle using atan2
            if x_spatial.shape[-1] > 1:
                last_theta = np.arctan2(x_spatial[..., -1], x_spatial[..., -2])
                angles.append(last_theta)

            # Combine r with angles
            spherical = np.concatenate([r[..., np.newaxis], np.stack(angles, axis=-1)], axis=-1)

        return spherical

    @staticmethod
    def spherical_to_hyperboloid(spherical: np.ndarray) -> np.ndarray:
        """
        Convert coordinates from spherical coordinates to the N-dimensional hyperboloid model.

        Parameters:
        - spherical: Spherical coordinates (r, θ_1, ..., θ_{n-1})
                     For 2D: (r, θ) where θ is in [0, 2π]

        Returns:
        - x: Coordinates in the N-dimensional hyperboloid model (R^(N+1))
        """
        # Convert to hyperboloid coordinates
        # t = cosh r
        # For 2D: x_1 = sinh r cos θ, x_2 = sinh r sin θ
        # For higher dimensions:
        # x_1 = sinh r cos θ_1
        # x_2 = sinh r sin θ_1 cos θ_2
        # x_i = sinh r * (Π_{j=1}^{i-1} sin θ_j) * cos θ_i for i = 1, ..., n-1
        # x_n = sinh r * (Π_{j=1}^{n-1} sin θ_j)

        r = spherical[..., 0]  # Distance
        angles = spherical[..., 1:]  # Angles

        # Calculate t (time coordinate)
        t = np.cosh(r)
        sinh_r = np.sinh(r)

        # Special case for 2D: single angle θ in [0, 2π]
        if angles.shape[-1] == 1:
            theta = angles[..., 0]
            x1 = sinh_r * np.cos(theta)
            x2 = sinh_r * np.sin(theta)
            x_spatial = [x1, x2]
        else:
            # General n-dimensional case
            x_spatial = []

            for i in range(angles.shape[-1]):
                if i == 0:
                    # First coordinate: x_1 = sinh r cos θ_1
                    x_i = sinh_r * np.cos(angles[..., 0])
                else:
                    # Subsequent coordinates: x_i = sinh r * (Π_{j=1}^{i-1} sin θ_j) * cos θ_i
                    sin_product = np.prod(np.sin(angles[..., :i]), axis=-1)
                    x_i = sinh_r * sin_product * np.cos(angles[..., i])
                x_spatial.append(x_i)

            # Last coordinate: x_n = sinh r * (Π_{j=1}^{n-1} sin θ_j)
            if angles.shape[-1] > 0:
                last_sin_product = np.prod(np.sin(angles), axis=-1)
            x_n = sinh_r * last_sin_product
            x_spatial.append(x_n)

        # Combine spatial coordinates with time coordinate
        x = np.concatenate([np.stack(x_spatial, axis=-1), t[..., np.newaxis]], axis=-1)

        return x

    @staticmethod
    def spherical_to_poincare(spherical: np.ndarray) -> np.ndarray:
        """
        Convert coordinates from spherical coordinates to the N-dimensional Poincaré ball model.
        Uses hyperboloid as an intermediate space.

        Parameters:
        - spherical: Spherical coordinates (r, θ_1, ..., θ_{n-1})

        Returns:
        - y: Coordinates in the N-dimensional Poincaré ball model (R^N)
        """
        # Convert spherical to hyperboloid first
        hyperboloid_coords = HyperbolicConversions.spherical_to_hyperboloid(spherical)

        # Then convert hyperboloid to Poincaré
        poincare_coords = HyperbolicConversions.hyperboloid_to_poincare(hyperboloid_coords)

        return poincare_coords

    @classmethod
    def convert_coordinates(cls, embeddings: np.ndarray, from_space: str, to_space: str) -> np.ndarray:
        """
        Convert embeddings between different hyperbolic spaces.

        Parameters:
        - embeddings: Input embeddings array (N x D matrix)
        - from_space: Source space ("poincare", "hyperboloid", "klein", "hemisphere", "half_plane", "spherical")
        - to_space: Target space ("poincare", "hyperboloid", "klein", "hemisphere", "half_plane", "spherical")

        Returns:
        - Converted embeddings array (N x D' matrix)
        """

        if from_space.lower() not in cls.SUPPORTED_SPACES:
            raise ValueError(f"from_space must be one of {cls.SUPPORTED_SPACES}")
        if to_space.lower() not in cls.SUPPORTED_SPACES:
            raise ValueError(f"to_space must be one of {cls.SUPPORTED_SPACES}")

        if from_space.lower() == to_space.lower():

            return embeddings

        # Handle matrix of embeddings by converting each row
        if embeddings.ndim > 2:

            converted_embeddings = []
            for i in range(embeddings.shape[0]):
                # Convert single embedding vector
                converted_vector = cls._convert_single_coordinate(embeddings[i], from_space, to_space)
                converted_embeddings.append(converted_vector)
            result = np.array(converted_embeddings)

            return result
        else:
            # Handle single embedding vector

            result = cls._convert_single_coordinate(embeddings, from_space, to_space)

            return result

    @classmethod
    def _convert_single_coordinate(cls, embedding: np.ndarray, from_space: str, to_space: str) -> np.ndarray:
        """
        Convert a single embedding vector between different hyperbolic spaces.

        Parameters:
        - embedding: Single embedding vector
        - from_space: Source space
        - to_space: Target space

        Returns:
        - Converted embedding vector
        """

        # Use hyperboloid as intermediate space for conversions
        if from_space.lower() != "hyperboloid":

            if from_space.lower() == "poincare":
                embedding = cls.poincare_to_hyperboloid(embedding)
            elif from_space.lower() == "klein":
                embedding = cls.klein_to_hyperboloid(embedding)
            elif from_space.lower() == "hemisphere":
                embedding = cls.hemisphere_to_hyperboloid(embedding)
            elif from_space.lower() == "half_plane":
                embedding = cls.half_plane_to_hyperboloid(embedding)
            elif from_space.lower() == "spherical":
                embedding = cls.spherical_to_hyperboloid(embedding)

        if to_space.lower() != "hyperboloid":

            if to_space.lower() == "poincare":
                embedding = cls.hyperboloid_to_poincare(embedding)
            elif to_space.lower() == "klein":
                embedding = cls.hyperboloid_to_klein(embedding)
            elif to_space.lower() == "hemisphere":
                embedding = cls.hyperboloid_to_hemisphere(embedding)
            elif to_space.lower() == "half_plane":
                embedding = cls.hyperboloid_to_half_plane(embedding)
            elif to_space.lower() == "spherical":
                embedding = cls.hyperboloid_to_spherical(embedding)

        return embedding

    @staticmethod
    def validate_embeddings(embeddings: np.ndarray, space: str, tol: float = 1e-6) -> bool:
        """
        Validate that embeddings satisfy the constraints of the given space.

        Parameters:
        - embeddings: Embeddings to validate
        - space: Space to validate against

        Returns:
        - True if valid, raises ValueError if invalid
        """
        if space.lower() == "poincare":
            # Check if all points are within the unit ball
            norms = np.linalg.norm(embeddings, axis=-1)
            if np.any(norms >= 1.0 + tol):
                raise ValueError("Poincaré embeddings must have norm < 1")
        elif space.lower() == "hyperboloid":
            # Check hyperboloid constraint: ||x||^2 - t^2 = -1
            spatial_norms = np.sum(embeddings[..., :-1] ** 2, axis=-1)
            t_squared = embeddings[..., -1] ** 2
            constraint = spatial_norms - t_squared
            if not np.allclose(constraint, -1, atol=1e-6):
                raise ValueError("Hyperboloid embeddings must satisfy ||x||^2 - t^2 = -1")
        elif space.lower() == "klein":
            # Check if all points are within the unit ball
            norms = np.linalg.norm(embeddings, axis=-1)
            if np.any(norms >= 1.0):
                raise ValueError("Klein embeddings must have norm < 1")
        elif space.lower() == "hemisphere":
            # Check sphere constraint: ||y||^2 + z^2 = 1 and z > 0
            spatial_norms = np.sum(embeddings[..., :-1] ** 2, axis=-1)
            z_squared = embeddings[..., -1] ** 2
            constraint = spatial_norms + z_squared
            if not np.allclose(constraint, 1, atol=1e-6):
                raise ValueError("Hemisphere embeddings must satisfy ||y||^2 + z^2 = 1")
            if np.any(embeddings[..., -1] <= 0):
                raise ValueError("Hemisphere embeddings must have z > 0")
        elif space.lower() == "half_plane":
            # Check half-plane constraint: z > 0
            if np.any(embeddings[..., -1] <= 0):
                raise ValueError("Half-plane embeddings must have z > 0")
        elif space.lower() == "spherical":
            # Check spherical constraints: r > 0, 0 <= θ_i <= π for i < n-1, 0 <= θ_{n-1} < 2π
            r = embeddings[..., 0]
            angles = embeddings[..., 1:]
            if np.any(r <= 0):
                raise ValueError("Spherical embeddings must have r > 0")
            if angles.shape[-1] > 0:
                # Check angle constraints
                for i in range(angles.shape[-1] - 1):
                    if np.any(angles[..., i] < 0) or np.any(angles[..., i] > np.pi):
                        raise ValueError(f"Spherical angle θ_{i+1} must be in [0, π]")
                # Last angle can be in [0, 2π)
                if np.any(angles[..., -1] < 0) or np.any(angles[..., -1] >= 2 * np.pi):
                    raise ValueError(f"Spherical angle θ_{angles.shape[-1]} must be in [0, 2π)")
        else:
            raise ValueError(f"Unsupported space: {space}")
        return True

    @staticmethod
    def compute_distances(embeddings: np.ndarray, space: str = "hyperboloid") -> np.ndarray:
        """
        Compute pairwise hyperbolic distances between embeddings.

        Parameters:
        - embeddings: Embeddings array
        - space: Space of the embeddings

        Returns:
        - Distance matrix
        """
        if space.lower() == "poincare":
            # Poincaré distance: d(x,y) = 2 * arctanh(||x-y|| / ||1 - xy*||)
            n = len(embeddings)
            distances = np.zeros((n, n))

            for i in range(n):
                for j in range(n):
                    if i != j:
                        x, y = embeddings[i], embeddings[j]
                        diff_norm = np.linalg.norm(x - y)
                        denom_norm = np.linalg.norm(1 - np.dot(x, y))
                        distances[i, j] = 2 * np.arctanh(diff_norm / denom_norm)

        elif space.lower() == "hyperboloid":
            # Hyperboloid distance: d(x,y) = arccosh(-<x,y>)
            # where <x,y> = x1*y1 + ... + xn*yn - xn+1*yn+1
            n = len(embeddings)
            distances = np.zeros((n, n))

            for i in range(n):
                for j in range(n):
                    if i != j:
                        x, y = embeddings[i], embeddings[j]
                        # Lorentzian inner product
                        spatial_dot = np.dot(x[:-1], y[:-1])
                        time_product = x[-1] * y[-1]
                        lorentzian_product = spatial_dot - time_product
                        distances[i, j] = np.arccosh(-lorentzian_product)
        else:
            raise ValueError(f"Distance computation not implemented for {space} space")

        return distances


# Convenience functions for direct use
def hyperboloid_to_poincare(x: np.ndarray) -> np.ndarray:
    """Convert from hyperboloid to Poincaré coordinates."""
    return HyperbolicConversions.hyperboloid_to_poincare(x)


def poincare_to_hyperboloid(y: np.ndarray) -> np.ndarray:
    """Convert from Poincaré to hyperboloid coordinates."""
    return HyperbolicConversions.poincare_to_hyperboloid(y)


def klein_to_hyperboloid(y: np.ndarray) -> np.ndarray:
    """Convert from Klein to hyperboloid coordinates."""
    return HyperbolicConversions.klein_to_hyperboloid(y)


def hyperboloid_to_klein(x: np.ndarray) -> np.ndarray:
    """Convert from hyperboloid to Klein coordinates."""
    return HyperbolicConversions.hyperboloid_to_klein(x)


def hyperboloid_to_hemisphere(x: np.ndarray) -> np.ndarray:
    """Convert from hyperboloid to hemisphere coordinates."""
    return HyperbolicConversions.hyperboloid_to_hemisphere(x)


def hemisphere_to_hyperboloid(y: np.ndarray) -> np.ndarray:
    """Convert from hemisphere to hyperboloid coordinates."""
    return HyperbolicConversions.hemisphere_to_hyperboloid(y)


def convert_coordinates(embeddings: np.ndarray, from_space: str, to_space: str) -> np.ndarray:
    """Convert embeddings between different hyperbolic spaces."""
    return HyperbolicConversions.convert_coordinates(embeddings, from_space, to_space)


def validate_embeddings(embeddings: np.ndarray, space: str) -> bool:
    """Validate that embeddings satisfy the constraints of the given space."""
    return HyperbolicConversions.validate_embeddings(embeddings, space)


def compute_distances(embeddings: np.ndarray, space: str = "hyperboloid") -> np.ndarray:
    """Compute pairwise hyperbolic distances between embeddings."""
    return HyperbolicConversions.compute_distances(embeddings, space)


def hyperboloid_to_half_plane(x: np.ndarray) -> np.ndarray:
    """Convert from hyperboloid to half-plane coordinates."""
    return HyperbolicConversions.hyperboloid_to_half_plane(x)


def half_plane_to_hyperboloid(y: np.ndarray) -> np.ndarray:
    """Convert from half-plane to hyperboloid coordinates."""
    return HyperbolicConversions.half_plane_to_hyperboloid(y)


def hyperboloid_to_spherical(x: np.ndarray) -> np.ndarray:
    """Convert from hyperboloid to spherical coordinates."""
    return HyperbolicConversions.hyperboloid_to_spherical(x)


def spherical_to_hyperboloid(spherical: np.ndarray) -> np.ndarray:
    """Convert from spherical to hyperboloid coordinates."""
    return HyperbolicConversions.spherical_to_hyperboloid(spherical)


def spherical_to_poincare(spherical: np.ndarray) -> np.ndarray:
    """Convert from spherical to Poincaré coordinates."""
    return HyperbolicConversions.spherical_to_poincare(spherical)
