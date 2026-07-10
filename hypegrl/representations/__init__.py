# -*- coding: utf-8 -*-
"""Swappable embedding representations (charts) — see :mod:`.base`."""

from hypegrl.representations.ball import BallRepresentation
from hypegrl.representations.base import (
    Representation,
    as_tensor,
    build_representation,
)
from hypegrl.representations.hyperboloid import HyperboloidRepresentation
from hypegrl.representations.polar import PolarRepresentation

__all__ = [
    "Representation",
    "as_tensor",
    "build_representation",
    "PolarRepresentation",
    "BallRepresentation",
    "HyperboloidRepresentation",
]
