"""Weibull tail fitting and IG-OpenMax calibration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import weibull_min
from torch import Tensor


def fit_openmax(
    logits: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    num_known: int,
    tail_size: int,
) -> dict[str, Any]:
    models: list[dict[str, Any]] = []
    for class_index in range(num_known):
        correct = (targets == class_index) & (predictions == class_index)
        class_logits = logits[correct]
        if len(class_logits) == 0:
            class_logits = logits[targets == class_index]
        if len(class_logits) == 0:
            raise ValueError(f"No activation vectors available for class {class_index}")
        mav = class_logits.mean(axis=0)
        distances = np.linalg.norm(class_logits - mav[None, :], axis=1)
        tail = np.sort(distances)[-min(tail_size, len(distances)) :]
        if np.allclose(tail, tail[0]):
            shape, location, scale = 1.0, 0.0, max(float(tail[0]), 1e-6)
        else:
            shape, location, scale = weibull_min.fit(tail, floc=0.0)
        models.append(
            {
                "class_index": class_index,
                "mav": mav.tolist(),
                "shape": float(shape),
                "location": float(location),
                "scale": max(float(scale), 1e-8),
                "tail_count": int(len(tail)),
            }
        )
    return {"num_known": num_known, "tail_size": tail_size, "models": models}


def save_openmax(model: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, indent=2), encoding="utf-8")


def load_openmax(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def calibrate_openmax(logits: Tensor, fitted: dict[str, Any], alpha: int = 3) -> Tensor:
    """Redistribute top-known activation mass to the explicit unknown logit."""
    num_known = int(fitted["num_known"])
    if logits.shape[1] != num_known + 1:
        raise ValueError("IG-OpenMax expects K known logits plus one synthetic-unknown logit")
    adjusted = logits.clone()
    known = logits[:, :num_known]
    top_count = min(alpha, num_known)
    top_indices = known.argsort(dim=1, descending=True)[:, :top_count]
    unknown_addition = torch.zeros(logits.shape[0], device=logits.device, dtype=logits.dtype)
    for batch_index in range(logits.shape[0]):
        activation = logits[batch_index]
        for rank, class_index_tensor in enumerate(top_indices[batch_index]):
            class_index = int(class_index_tensor.item())
            model = fitted["models"][class_index]
            mav = torch.tensor(model["mav"], device=logits.device, dtype=logits.dtype)
            distance = torch.linalg.vector_norm(activation - mav).item()
            cdf = float(
                weibull_min.cdf(
                    distance,
                    model["shape"],
                    loc=model["location"],
                    scale=model["scale"],
                )
            )
            rank_weight = (top_count - rank) / top_count
            omega = rank_weight * cdf
            removable = torch.relu(activation[class_index]) * omega
            adjusted[batch_index, class_index] -= removable
            unknown_addition[batch_index] += removable
    adjusted[:, num_known] += unknown_addition
    return torch.softmax(adjusted, dim=1)

