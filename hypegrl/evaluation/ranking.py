"""Ranking metrics for link prediction: F1 at a cutoff and the lift curve.

These operate on candidate-level arrays — a ``scores`` vector and a boolean
``is_positive`` mask over the same candidates — and know nothing about graphs.
The ``higher_is_link`` flag selects the ranking direction: ``True`` when a
larger score means a more likely link (edge probabilities), ``False`` when a
smaller score does (hyperbolic distances).

Precision/recall/F1 are delegated to :mod:`sklearn.metrics`; only the ranking
and top-``k`` thresholding are done here. The lift curve is computed directly,
as scikit-learn has no decile-lift equivalent.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def _rank_order(scores: np.ndarray, higher_is_link: bool) -> np.ndarray:
    """Indices of ``scores`` ordered most-likely-link first (stable ties).

    Rejects ``NaN`` scores: ``np.argsort`` sorts ``NaN`` to the end regardless of
    direction, so a ``NaN``-scored candidate is silently ranked least-likely-link
    rather than raising — a ``NaN`` in the decoder output would corrupt the metric
    invisibly. Fail loudly instead. (``±inf`` is left alone: it orders
    deterministically and a ``+inf`` distance is a legitimate "definitely not a
    link".)
    """
    scores = np.asarray(scores, dtype=float)
    if np.isnan(scores).any():
        raise ValueError(
            "scores contains NaN; ranking is undefined (np.argsort places NaN "
            "last regardless of `higher_is_link`, silently mis-ranking those "
            "candidates). This usually means the decoder produced NaN — check the "
            "embedding/decoder output before scoring."
        )
    order = np.argsort(scores, kind="stable")
    if higher_is_link:
        order = order[::-1]
    return order


def precision_recall_f1_at_k(
    scores,
    is_positive,
    k: int = None,
    higher_is_link: bool = True,
) -> dict:
    """Precision, recall and F1 when predicting the top-``k`` candidates.

    Ranks candidates by ``scores``, turns the top ``k`` into a binary
    prediction, and defers precision/recall/F1 to :mod:`sklearn.metrics`
    (``zero_division=0``). Only the ranking and thresholding are done here;
    the metrics themselves are sklearn's.

    Parameters
    ----------
    scores:
        Score per candidate.
    is_positive:
        Boolean mask, ``True`` for the held-out positive candidates.
    k:
        Number of links to predict. ``None`` uses the number of positives —
        the paper's protocol, under which precision, recall and F1 coincide.
    higher_is_link:
        Ranking direction (see module docstring).

    Returns
    -------
    dict
        ``precision``, ``recall``, ``f1``, ``tp`` (true positives), ``k``,
        ``n_positives``.
    """
    y_true = np.asarray(is_positive, dtype=bool)
    n_pos = int(y_true.sum())
    if k is None:
        k = n_pos
    order = _rank_order(scores, higher_is_link)
    y_pred = np.zeros(y_true.size, dtype=bool)
    y_pred[order[:k]] = True
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tp": int(np.logical_and(y_true, y_pred).sum()),
        "k": k,
        "n_positives": n_pos,
    }


def f1_at_k(scores, is_positive, k: int = None, higher_is_link: bool = True) -> float:
    """F1 for the top-``k`` predictions (see :func:`precision_recall_f1_at_k`)."""
    return precision_recall_f1_at_k(scores, is_positive, k, higher_is_link)["f1"]


@dataclass(frozen=True)
class LiftCurve:
    """Per-bin true-positive breakdown of a ranked candidate list.

    Candidates are ranked most-likely-link first and split into ``n_bins``
    equally sized bins (the last bin absorbs the remainder). ``bin_true_positives[b]``
    counts held-out positives landing in bin ``b``.
    """

    bin_true_positives: list[int]
    bin_counts: list[int]
    n_positives: int
    n_candidates: int

    @property
    def n_bins(self) -> int:
        return len(self.bin_counts)

    @property
    def baseline_rate(self) -> float:
        """Overall positive rate ``n_positives / n_candidates`` (random baseline)."""
        return self.n_positives / self.n_candidates if self.n_candidates else 0.0

    @property
    def captured_in_first_bin(self) -> tuple[int, int]:
        """``(positives in the top bin, total positives)``.

        With ``n_bins=10`` this is the paper's "Lift (1st decile)" figure —
        the share of held-out edges recovered in the top-10% of candidates.
        """
        return self.bin_true_positives[0], self.n_positives

    @property
    def lift(self) -> list[float]:
        """Per-bin lift: bin positive rate divided by the baseline rate."""
        base = self.baseline_rate
        return [
            (tp / c) / base if c and base else 0.0
            for tp, c in zip(self.bin_true_positives, self.bin_counts)
        ]


def lift_curve(
    scores, is_positive, n_bins: int = 10, higher_is_link: bool = True
) -> LiftCurve:
    """Bin a ranked candidate list and count positives per bin.

    Parameters
    ----------
    scores:
        Score per candidate.
    is_positive:
        Boolean mask, ``True`` for held-out positive candidates.
    n_bins:
        Number of equally sized bins (``10`` for deciles).
    higher_is_link:
        Ranking direction (see module docstring).

    Returns
    -------
    LiftCurve
    """
    is_positive = np.asarray(is_positive, dtype=bool)
    order = _rank_order(scores, higher_is_link)
    ranked = is_positive[order]
    n = ranked.size
    bin_size = n // n_bins

    bin_tp: list[int] = []
    bin_counts: list[int] = []
    for b in range(n_bins):
        start = b * bin_size
        end = n if b == n_bins - 1 else (b + 1) * bin_size
        chunk = ranked[start:end]
        bin_tp.append(int(chunk.sum()))
        bin_counts.append(int(chunk.size))

    return LiftCurve(
        bin_true_positives=bin_tp,
        bin_counts=bin_counts,
        n_positives=int(is_positive.sum()),
        n_candidates=n,
    )
