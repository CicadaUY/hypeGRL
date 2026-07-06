"""Evaluation utilities: dataset-agnostic protocols and metrics.

Reusable, dependency-light building blocks for benchmarking embedders —
edge-removal link-prediction splits, ranking metrics, distance-based
classification, and graph geometry statistics. Paper-specific dataset loaders
and baselines live outside the library (in ``experiments/``).
"""
from hypegrl.evaluation.link_prediction import (
    LinkPredictionSplit,
    candidate_scores,
    link_prediction_split,
    training_graph,
)
from hypegrl.evaluation.ranking import (
    LiftCurve,
    f1_at_k,
    lift_curve,
    precision_recall_f1_at_k,
)

__all__ = [
    "LinkPredictionSplit",
    "candidate_scores",
    "link_prediction_split",
    "training_graph",
    "LiftCurve",
    "f1_at_k",
    "lift_curve",
    "precision_recall_f1_at_k",
]
