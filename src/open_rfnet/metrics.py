"""Open-set metrics used by the paper."""

from __future__ import annotations

from typing import Any

import numpy as np


def open_set_metrics(targets: np.ndarray, predictions: np.ndarray, num_known: int) -> dict[str, Any]:
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    true_known = targets < num_known
    true_unknown = ~true_known
    predicted_known = predictions < num_known
    predicted_unknown = ~predicted_known
    correct_known_class = true_known & predicted_known & (targets == predictions)
    correct_unknown = true_unknown & predicted_unknown

    def ratio(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else 0.0

    kar = ratio(int(correct_known_class.sum()), int(true_known.sum()))
    uar = ratio(int(correct_unknown.sum()), int(true_unknown.sum()))
    kp = ratio(int(correct_known_class.sum()), int(predicted_known.sum()))
    up = ratio(int(correct_unknown.sum()), int(predicted_unknown.sum()))
    return {
        "KAR": kar,
        "UAR": uar,
        "KP": kp,
        "UP": up,
        "GAP": abs(uar - kar),
        "known_samples": int(true_known.sum()),
        "unknown_samples": int(true_unknown.sum()),
    }

