"""Evaluation utilities: dataset-agnostic protocols and metrics.

Reusable, dependency-light building blocks for benchmarking embedders —
edge-removal link-prediction splits, ranking metrics, distance-based
classification, and graph geometry statistics. Paper-specific dataset loaders
and baselines live outside the library (in ``experiments/``).
"""
from hypegrl.evaluation.link_prediction import (
    LinkPredictionSplit,
    link_prediction_split,
    training_graph,
)

__all__ = [
    "LinkPredictionSplit",
    "link_prediction_split",
    "training_graph",
]
