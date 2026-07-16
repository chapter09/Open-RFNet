"""Supervised contrastive objective from Eq. (13)."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: Tensor, labels: Tensor) -> Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, views, dimension]")
        batch, views, _ = features.shape
        if batch < 2:
            raise ValueError("supervised contrastive loss requires batch_size >= 2")
        features = torch.nn.functional.normalize(features, dim=-1)
        contrast = torch.cat(torch.unbind(features, dim=1), dim=0)
        logits = contrast @ contrast.T / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        label_mask = labels.view(-1, 1).eq(labels.view(1, -1)).float()
        positive_mask = label_mask.repeat(views, views).to(logits.device)
        self_mask = torch.ones_like(positive_mask)
        self_mask.fill_diagonal_(0)
        positive_mask = positive_mask * self_mask

        exp_logits = torch.exp(logits) * self_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
        positive_count = positive_mask.sum(dim=1).clamp_min(1.0)
        mean_log_prob = (positive_mask * log_prob).sum(dim=1) / positive_count
        return -mean_log_prob.mean()

