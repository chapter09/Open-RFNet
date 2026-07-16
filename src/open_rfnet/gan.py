"""Conditional WGAN-GP used to synthesize boundary/unknown samples."""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class ConditionalGenerator(nn.Module):
    def __init__(
        self,
        image_size: int,
        num_classes: int,
        latent_dim: int,
        label_dim: int,
        base_channels: int,
    ):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.label_embedding = nn.Embedding(num_classes, label_dim)
        initial = base_channels * 8
        self.fc = nn.Linear(latent_dim + label_dim, initial * 4 * 4)
        blocks: list[nn.Module] = []
        current = initial
        upsample_count = max(1, math.ceil(math.log2(image_size / 4)))
        for _ in range(upsample_count):
            following = max(base_channels // 2, current // 2, 8)
            blocks.extend(
                (
                    nn.ConvTranspose2d(current, following, 4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(following),
                    nn.ReLU(inplace=True),
                )
            )
            current = following
        self.blocks = nn.Sequential(*blocks)
        self.output = nn.Sequential(nn.Conv2d(current, 1, 3, padding=1), nn.Sigmoid())
        self.initial_channels = initial

    def forward(self, noise: Tensor, labels: Tensor) -> Tensor:
        conditioned = torch.cat((noise, self.label_embedding(labels)), dim=1)
        image = self.fc(conditioned).view(-1, self.initial_channels, 4, 4)
        image = self.output(self.blocks(image))
        if image.shape[-1] != self.image_size:
            image = F.interpolate(image, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return image


class ProjectionCritic(nn.Module):
    def __init__(self, image_size: int, num_classes: int, base_channels: int):
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 1
        channels = base_channels
        size = image_size
        while size > 4:
            layers.append(nn.Conv2d(in_channels, channels, 4, stride=2, padding=1))
            if in_channels != 1:
                layers.append(nn.InstanceNorm2d(channels, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = channels
            channels = min(channels * 2, base_channels * 8)
            size = max(1, size // 2)
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.score = nn.Linear(in_channels, 1)
        self.label_embedding = nn.Embedding(num_classes, in_channels)

    def forward(self, image: Tensor, labels: Tensor) -> Tensor:
        feature = self.pool(self.features(image)).flatten(1)
        projection = (feature * self.label_embedding(labels)).sum(dim=1, keepdim=True)
        return self.score(feature) + projection


def gradient_penalty(
    critic: ProjectionCritic,
    real: Tensor,
    fake: Tensor,
    labels: Tensor,
) -> Tensor:
    epsilon = torch.rand((real.shape[0], 1, 1, 1), device=real.device)
    interpolated = (epsilon * real + (1.0 - epsilon) * fake).requires_grad_(True)
    score = critic(interpolated, labels)
    gradient = torch.autograd.grad(
        outputs=score,
        inputs=interpolated,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return (gradient.flatten(1).norm(2, dim=1) - 1.0).square().mean()


def train_wgan(
    loader: DataLoader,
    image_size: int,
    num_classes: int,
    config: dict[str, Any],
    device: torch.device,
    output_path: str | Path,
) -> ConditionalGenerator:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    latent_dim = int(config["latent_dim"])
    generator = ConditionalGenerator(
        image_size,
        num_classes,
        latent_dim,
        int(config["label_dim"]),
        int(config["base_channels"]),
    ).to(device)
    critic = ProjectionCritic(image_size, num_classes, int(config["base_channels"])).to(device)
    learning_rate = float(config["learning_rate"])
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.0, 0.9))
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=learning_rate, betas=(0.0, 0.9))
    critic_steps = int(config.get("critic_steps", 5))
    penalty_weight = float(config.get("gradient_penalty", 10.0))
    start_epoch = 0
    if bool(config.get("resume", True)) and output_path.exists():
        checkpoint = torch.load(output_path, map_location=device, weights_only=False)
        generator.load_state_dict(checkpoint["generator"])
        if "critic" in checkpoint:
            critic.load_state_dict(checkpoint["critic"])
        if "generator_optimizer" in checkpoint:
            generator_optimizer.load_state_dict(checkpoint["generator_optimizer"])
        if "critic_optimizer" in checkpoint:
            critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        start_epoch = int(checkpoint.get("completed_epochs", 0))

    for epoch in range(start_epoch, int(config["epochs"])):
        progress = tqdm(loader, desc=f"wgan {epoch + 1}", leave=False)
        for step, (real, labels, _) in enumerate(progress):
            real = real.to(device)
            labels = labels.to(device)
            noise = torch.randn(real.shape[0], latent_dim, device=device)
            fake = generator(noise, labels).detach()
            critic_loss = critic(fake, labels).mean() - critic(real, labels).mean()
            penalty = gradient_penalty(critic, real, fake, labels)
            total_critic = critic_loss + penalty_weight * penalty
            critic_optimizer.zero_grad(set_to_none=True)
            total_critic.backward()
            critic_optimizer.step()

            if step % critic_steps == 0:
                noise = torch.randn(real.shape[0], latent_dim, device=device)
                generated = generator(noise, labels)
                generator_loss = -critic(generated, labels).mean()
                generator_optimizer.zero_grad(set_to_none=True)
                generator_loss.backward()
                generator_optimizer.step()
                progress.set_postfix(g=f"{generator_loss.item():.3f}", c=f"{total_critic.item():.3f}")

        # GAN training is the longest stage. Save a fully resumable state after
        # every epoch so empirical epoch-count tuning never discards progress.
        torch.save(
            {
                "generator": generator.state_dict(),
                "critic": critic.state_dict(),
                "generator_optimizer": generator_optimizer.state_dict(),
                "critic_optimizer": critic_optimizer.state_dict(),
                "completed_epochs": epoch + 1,
                "config": config,
            },
            output_path,
        )

    return generator


@torch.inference_mode()
def select_synthetic_unknown(
    generator: ConditionalGenerator,
    closed_model: nn.Module,
    num_classes: int,
    config: dict[str, Any],
    device: torch.device,
    output_path: str | Path,
) -> tuple[Tensor, Tensor]:
    generator.eval()
    closed_model.eval()
    per_class = int(config.get("candidates_per_class", 1000))
    maximum = int(config.get("max_synthetic_unknown", 5000))
    # G-OpenMax-style boundary criterion: a generated sample that the closed
    # model misclassifies *with high confidence* lies inside another class's
    # decision region, not on a boundary; training the open head on it
    # relabels that region as unknown and rejects real known samples there.
    max_confidence = float(config.get("synthetic_max_confidence", 1.0))
    batch_size = int(config.get("selection_batch_size", max(int(config.get("batch_size", 32)), 64)))
    latent_dim = int(config["latent_dim"])
    cache_tag = int(config.get("selection_cache_tag", -1))
    output_path = Path(output_path)
    partial_dir = output_path.with_name(output_path.stem + "_partial")
    partial_dir.mkdir(parents=True, exist_ok=True)
    # The hosting session can kill this stage at any moment, so each class's
    # result is checkpointed to its own file (atomically) and re-entry skips
    # finished classes instead of regenerating everything.
    per_class_results: dict[int, tuple[Tensor, Tensor]] = {}
    for class_file in sorted(partial_dir.glob("c*.pt")):
        try:
            saved = torch.load(class_file, map_location="cpu", weights_only=False)
            if int(saved.get("cache_tag", -2)) == cache_tag:
                per_class_results[int(saved["class_index"])] = (saved["images"], saved["labels"])
        except (RuntimeError, EOFError, KeyError):
            continue
    fallback: list[tuple[Tensor, Tensor, Tensor]] = []
    fallback_count = 0
    selected_any = any(pair[0].shape[0] for pair in per_class_results.values())
    for class_index in range(num_classes):
        if class_index in per_class_results:
            continue
        remaining = per_class
        class_images: list[Tensor] = []
        class_labels: list[Tensor] = []
        # Every conditioning class must contribute candidates before any global
        # cap is applied; otherwise the boundary set only covers the feature
        # regions of the first few classes and unknowns elsewhere are missed.
        while remaining > 0:
            count = min(batch_size, remaining)
            labels = torch.full((count,), class_index, device=device, dtype=torch.long)
            noise = torch.randn(count, latent_dim, device=device)
            images = generator(noise, labels)
            logits = closed_model(images)
            predictions = logits.argmax(dim=1)
            mask = predictions.ne(labels)
            if max_confidence < 1.0:
                mask &= logits.softmax(dim=1).amax(dim=1) <= max_confidence
            if mask.any():
                class_images.append(images[mask].to(dtype=torch.float16, device="cpu"))
                class_labels.append(labels[mask].cpu())
                selected_any = True
                fallback.clear()
                fallback_count = 0
            elif not selected_any and fallback_count < maximum:
                keep = min(count, maximum - fallback_count)
                fallback.append(
                    (
                        images[:keep].to(dtype=torch.float16, device="cpu"),
                        labels[:keep].cpu(),
                        logits.softmax(dim=1).amax(dim=1)[:keep].cpu(),
                    )
                )
                fallback_count += keep
            remaining -= count
        per_class_results[class_index] = (
            torch.cat(class_images) if class_images else torch.empty(0, 1, generator.image_size, generator.image_size, dtype=torch.float16),
            torch.cat(class_labels) if class_labels else torch.empty(0, dtype=torch.long),
        )
        class_file = partial_dir / f"c{class_index:02d}.pt"
        temp = class_file.with_suffix(".tmp")
        torch.save(
            {
                "cache_tag": cache_tag,
                "class_index": class_index,
                "images": per_class_results[class_index][0],
                "labels": per_class_results[class_index][1],
            },
            temp,
        )
        temp.replace(class_file)

    selected = [pair[0] for pair in per_class_results.values() if pair[0].shape[0]]
    source_labels = [pair[1] for pair in per_class_results.values() if pair[1].shape[0]]
    if selected:
        images = torch.cat(selected)
        labels = torch.cat(source_labels)
        if images.shape[0] > maximum:
            # Stratified subsample so the retained boundary samples keep the
            # per-class coverage instead of truncating to the earliest classes.
            keep: list[Tensor] = []
            unique_labels = labels.unique()
            quota = maximum // len(unique_labels)
            for value in unique_labels:
                indices = (labels == value).nonzero(as_tuple=True)[0]
                order = indices[torch.randperm(len(indices))[: max(quota, 1)]]
                keep.append(order)
            kept = torch.cat(keep)
            if len(kept) < maximum:
                remaining_mask = torch.ones(images.shape[0], dtype=torch.bool)
                remaining_mask[kept] = False
                leftovers = remaining_mask.nonzero(as_tuple=True)[0]
                extra = leftovers[torch.randperm(len(leftovers))[: maximum - len(kept)]]
                kept = torch.cat((kept, extra))
            kept = kept[:maximum]
            images = images[kept]
            labels = labels[kept]
    else:
        # A perfectly classified generator produces no Eq. (22) samples. Keep
        # the least-confident generated boundary candidates so the pipeline can
        # report and continue, while marking the fallback in the saved file.
        candidates = torch.cat([item[0] for item in fallback])
        labels = torch.cat([item[1] for item in fallback])
        confidence = torch.cat([item[2] for item in fallback])
        order = confidence.argsort()[: min(maximum, len(confidence))]
        images = candidates[order].to(torch.float16)
        labels = labels[order]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_suffix(".tmp")
    torch.save({"images": images, "source_labels": labels, "selected_count": len(images)}, temp)
    temp.replace(output_path)
    shutil.rmtree(partial_dir, ignore_errors=True)
    return images, labels
