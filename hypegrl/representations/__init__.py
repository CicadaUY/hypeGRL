# -*- coding: utf-8 -*-
"""Swappable embedding representations (charts) — see :mod:`.base`."""

from hypegrl.representations.ball import BallRepresentation
from hypegrl.representations.base import (
    Representation,
    as_tensor,
    build_representation,
    zero_diagonal,
)
from hypegrl.representations.hyperboloid import HyperboloidRepresentation
from hypegrl.representations.polar import (
    CurvedPolarRepresentation,
    ExactPolarRepresentation,
    PolarRepresentation,
)
from hypegrl.representations.tangent import TangentRepresentation

__all__ = [
    "Representation",
    "as_tensor",
    "build_representation",
    "zero_diagonal",
    "PolarRepresentation",
    "ExactPolarRepresentation",
    "CurvedPolarRepresentation",
    "BallRepresentation",
    "HyperboloidRepresentation",
    "TangentRepresentation",
]
