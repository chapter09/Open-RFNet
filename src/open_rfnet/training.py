"""Closed-set, GAN, IG-OpenMax, and evaluation stages."""

from __future__ import annotations

import json
import random
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from .constants import ALL_LABELS, KNOWN_LABELS
from .data import SpectrogramAugment, SpectrogramDataset
from .gan import ConditionalGenerator, select_synthetic_unknown, train_wgan
from .losses import SupervisedContrastiveLoss
from .metrics import open_set_metrics
from .model import OpenRFNet, build_model
from .openmax import calibrate_openmax, fit_openmax, load_openmax, save_openmax


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(
    dataset: Dataset,
    training: dict[str, Any],
    shuffle: bool,
    batch_size: int | None = None,
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader:
    workers = int(training.get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=int(batch_size or training["batch_size"]),
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=shuffle and len(dataset) >= int(batch_size or training["batch_size"]),
    )


def _save_checkpoint(model: OpenRFNet, model_config: dict[str, Any], path: Path, stage: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": stage,
            "model_config": model_config,
            "num_classes": model.classifier.out_features,
            "model": model.state_dict(),
        },
        path,
    )


def load_checkpoint(path: str | Path, device: torch.device) -> OpenRFNet:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(checkpoint["model_config"], int(checkpoint["num_classes"]))
    model.load_state_dict(checkpoint["model"])
    return model.to(device)


def train_closed(config: dict[str, Any], manifest_path: str | Path) -> Path:
    seed_everything(int(config.get("seed", 42)))
    device = resolve_device()
    training = config["training"]
    model_config = config["model"]
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = SpectrogramDataset(manifest_path, "train")
    val_dataset = SpectrogramDataset(manifest_path, "val")
    train_loader = make_loader(train_dataset, training, shuffle=True)
    val_loader = make_loader(val_dataset, training, shuffle=False) if len(val_dataset) else None
    model = build_model(model_config).to(device)
    amp = bool(training.get("amp", True)) and device.type == "cuda"
    augmenter = SpectrogramAugment(**training.get("augmentation", {}))
    supcon = SupervisedContrastiveLoss(float(training.get("temperature", 0.07)))
    parameters = chain(model.encoder.parameters(), model.projection.parameters())
    optimizer = torch.optim.Adam(
        parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(training["supcon_epochs"]))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history: dict[str, list[float]] = {"supcon_loss": [], "classifier_loss": [], "val_accuracy": []}

    for epoch in range(int(training["supcon_epochs"])):
        model.train()
        losses: list[float] = []
        progress = tqdm(train_loader, desc=f"supcon {epoch + 1}", leave=False)
        for images, labels, _ in progress:
            images, labels = images.to(device), labels.to(device)
            first, second = augmenter(images), augmenter(images)
            joined = torch.cat((first, second), dim=0)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                projected = model.project(model.encode(joined))
                batch = images.shape[0]
                views = torch.stack((projected[:batch], projected[batch:]), dim=1)
                loss = supcon(views, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))
            progress.set_postfix(loss=f"{loss.item():.3f}")
        scheduler.step()
        history["supcon_loss"].append(float(np.mean(losses)))

    _save_checkpoint(model, model_config, run_dir / "contrastive.pt", "contrastive")

    freeze = bool(training.get("freeze_encoder_for_classifier", True))
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(not freeze)
    for parameter in model.projection.parameters():
        parameter.requires_grad_(False)
    classifier_learning_rate = float(training["classifier_learning_rate"])
    classifier_parameters: list[dict[str, Any]] = [
        {"params": model.classifier.parameters(), "lr": classifier_learning_rate}
    ]
    if not freeze:
        classifier_parameters.append(
            {
                "params": model.encoder.parameters(),
                "lr": float(
                    training.get("classifier_encoder_learning_rate", classifier_learning_rate)
                ),
            }
        )
    optimizer = torch.optim.Adam(
        classifier_parameters,
        lr=classifier_learning_rate,
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(training["classifier_epochs"]))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(training.get("classifier_label_smoothing", 0.0))
    )
    for epoch in range(int(training["classifier_epochs"])):
        model.train()
        if freeze:
            model.encoder.eval()
        losses = []
        for images, labels, _ in tqdm(train_loader, desc=f"classifier {epoch + 1}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))
        scheduler.step()
        history["classifier_loss"].append(float(np.mean(losses)))
        if val_loader is not None:
            predictions, targets, _ = collect_predictions(model, val_loader, device)
            history["val_accuracy"].append(float((predictions == targets).mean()))

    closed_path = run_dir / "closed.pt"
    _save_checkpoint(model, model_config, closed_path, "closed")
    (run_dir / "closed_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return closed_path


def finetune_closed(
    config: dict[str, Any],
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Fine-tune the encoder and classifier from an existing Stage-I checkpoint."""
    seed_everything(int(config.get("seed", 42)))
    device = resolve_device()
    training = config["training"]
    model = load_checkpoint(checkpoint_path, device)
    freeze = bool(training.get("freeze_encoder_for_classifier", False))
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(not freeze)
    for parameter in model.projection.parameters():
        parameter.requires_grad_(False)
    for parameter in model.classifier.parameters():
        parameter.requires_grad_(True)

    train_dataset = SpectrogramDataset(manifest_path, "train")
    val_dataset = SpectrogramDataset(manifest_path, "val")
    train_loader = make_loader(train_dataset, training, shuffle=True)
    val_loader = make_loader(val_dataset, training, shuffle=False)
    classifier_lr = float(training["classifier_learning_rate"])
    parameter_groups: list[dict[str, Any]] = [
        {"params": model.classifier.parameters(), "lr": classifier_lr}
    ]
    if not freeze:
        parameter_groups.append(
            {
                "params": model.encoder.parameters(),
                "lr": float(training.get("classifier_encoder_learning_rate", classifier_lr)),
            }
        )
    optimizer = torch.optim.Adam(
        parameter_groups,
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    epochs = int(training["classifier_epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    amp = bool(training.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(training.get("classifier_label_smoothing", 0.0))
    )
    history: dict[str, list[float]] = {"classifier_loss": [], "val_accuracy": []}
    for epoch in range(epochs):
        model.train()
        if freeze:
            model.encoder.eval()
        losses: list[float] = []
        for images, labels, _ in tqdm(train_loader, desc=f"finetune {epoch + 1}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))
        scheduler.step()
        predictions, targets, _ = collect_predictions(model, val_loader, device)
        history["classifier_loss"].append(float(np.mean(losses)))
        history["val_accuracy"].append(float((predictions == targets).mean()))

    output = Path(output_path)
    _save_checkpoint(model, config["model"], output, "closed-finetuned")
    output.with_suffix(".history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    return output


def evaluate_closed(
    config: dict[str, Any], manifest_path: str | Path, checkpoint_path: str | Path
) -> dict[str, Any]:
    """Evaluate exact known-class accuracy before open-set calibration."""
    device = resolve_device()
    dataset = SpectrogramDataset(manifest_path, "test")
    loader = make_loader(dataset, config["training"], shuffle=False)
    model = load_checkpoint(checkpoint_path, device)
    predictions, targets, class_indices = collect_predictions(model, loader, device)
    known = targets < len(KNOWN_LABELS)
    per_class: dict[str, float] = {}
    for class_index, label in enumerate(ALL_LABELS):
        if label not in KNOWN_LABELS:
            continue
        mask = known & (class_indices == class_index)
        if mask.any():
            expected = KNOWN_LABELS.index(label)
            per_class[label] = float((predictions[mask] == expected).mean())
    return {
        "accuracy": float((predictions[known] == targets[known]).mean()),
        "known_samples": int(known.sum()),
        "per_class_accuracy": per_class,
    }


@torch.inference_mode()
def collect_predictions(
    model: OpenRFNet, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    all_indices: list[np.ndarray] = []
    for images, labels, class_indices in loader:
        logits = model(images.to(device))
        predictions.append(logits.argmax(dim=1).cpu().numpy())
        targets.append(labels.numpy())
        all_indices.append(class_indices.numpy())
    return np.concatenate(predictions), np.concatenate(targets), np.concatenate(all_indices)


@torch.inference_mode()
def collect_logits(
    model: OpenRFNet, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    all_indices: list[np.ndarray] = []
    for images, labels, class_indices in loader:
        logits.append(model(images.to(device)).float().cpu().numpy())
        targets.append(labels.numpy())
        all_indices.append(class_indices.numpy())
    return np.concatenate(logits), np.concatenate(targets), np.concatenate(all_indices)


def train_gan_stage(config: dict[str, Any], manifest_path: str | Path, closed_path: str | Path) -> Path:
    seed_everything(int(config.get("seed", 42)))
    device = resolve_device()
    run_dir = Path(config["run_dir"])
    train_dataset = SpectrogramDataset(manifest_path, "train")
    gan_cfg = config["gan"]
    gan_training = dict(config["training"])
    gan_training["num_workers"] = config["training"].get("num_workers", 0)
    loader = make_loader(train_dataset, gan_training, shuffle=True, batch_size=int(gan_cfg["batch_size"]))
    generator_path = run_dir / "generator.pt"
    train_wgan(
        loader,
        int(config["model"]["input_size"]),
        len(KNOWN_LABELS),
        gan_cfg,
        device,
        generator_path,
    )
    return generator_path


class SyntheticUnknownDataset(Dataset[tuple[Tensor, int, int]]):
    def __init__(self, images: Tensor, unknown_index: int):
        self.images = images
        self.unknown_index = unknown_index

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, int, int]:
        return self.images[index].float(), self.unknown_index, -1


def _load_generator(config: dict[str, Any], path: Path, device: torch.device) -> ConditionalGenerator:
    gan_cfg = config["gan"]
    generator = ConditionalGenerator(
        int(config["model"]["input_size"]),
        len(KNOWN_LABELS),
        int(gan_cfg["latent_dim"]),
        int(gan_cfg["label_dim"]),
        int(gan_cfg["base_channels"]),
    ).to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    generator.load_state_dict(checkpoint["generator"])
    return generator


def train_open_stage(
    config: dict[str, Any],
    manifest_path: str | Path,
    closed_path: str | Path,
    generator_path: str | Path,
) -> tuple[Path, Path]:
    seed_everything(int(config.get("seed", 42)))
    device = resolve_device()
    run_dir = Path(config["run_dir"])
    training = config["training"]
    gan_cfg = config["gan"]
    closed_model = load_checkpoint(closed_path, device)
    generator_epochs = int(
        torch.load(Path(generator_path), map_location="cpu", weights_only=False).get(
            "completed_epochs", 0
        )
    )
    synthetic_path = run_dir / "synthetic_unknown.pt"
    # The whole stage may be killed and re-entered at any point (the hosting
    # session can terminate background work), so every expensive step below is
    # cached on disk and skipped when its inputs are unchanged.
    images = None
    if synthetic_path.exists():
        try:
            cached = torch.load(synthetic_path, map_location="cpu", weights_only=False)
            if int(cached.get("generator_epochs", -1)) == generator_epochs:
                images = cached["images"]
        except (RuntimeError, EOFError, KeyError):
            images = None
    if images is None:
        generator = _load_generator(config, Path(generator_path), device)
        images, _ = select_synthetic_unknown(
            generator,
            closed_model,
            len(KNOWN_LABELS),
            {**gan_cfg, "selection_cache_tag": generator_epochs},
            device,
            synthetic_path,
        )
        cached = torch.load(synthetic_path, map_location="cpu", weights_only=False)
        cached["generator_epochs"] = generator_epochs
        torch.save(cached, synthetic_path)
        del generator
        if device.type == "cuda":
            torch.cuda.empty_cache()

    model = closed_model
    model.expand_for_unknown()
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in model.projection.parameters():
        parameter.requires_grad_(False)
    train_dataset = SpectrogramDataset(manifest_path, "train")
    synthetic_dataset = SyntheticUnknownDataset(images, len(KNOWN_LABELS))
    combined = ConcatDataset((train_dataset, synthetic_dataset))
    real_count, synthetic_count = len(train_dataset), len(synthetic_dataset)
    if str(gan_cfg.get("open_sampling", "union")) == "balanced":
        weights = torch.cat(
            (
                torch.ones(real_count),
                torch.full((synthetic_count,), real_count / max(1, synthetic_count)),
            )
        )
        sampler = WeightedRandomSampler(weights, num_samples=2 * real_count, replacement=True)
        loader = make_loader(combined, training, shuffle=False, sampler=sampler)
    else:
        # Equation (23) specifies the natural union D_train ∪ D'_gan. The old
        # 50/50 sampler greatly over-weighted the single synthetic unknown class
        # relative to each of the twenty known classes and reduced KAR.
        loader = make_loader(combined, training, shuffle=True)
    optimizer = torch.optim.Adam(
        model.classifier.parameters(),
        lr=float(training["classifier_learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(gan_cfg["open_classifier_epochs"]))
    )
    criterion = nn.CrossEntropyLoss()
    amp = bool(training.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    progress_path = run_dir / "open_head_progress.pt"
    start_epoch = 0
    if progress_path.exists():
        try:
            saved = torch.load(progress_path, map_location=device, weights_only=False)
            if int(saved.get("generator_epochs", -1)) == generator_epochs:
                model.load_state_dict(saved["model"])
                optimizer.load_state_dict(saved["optimizer"])
                scheduler.load_state_dict(saved["scheduler"])
                scaler.load_state_dict(saved["scaler"])
                start_epoch = int(saved["completed_epochs"])
        except (RuntimeError, EOFError, KeyError):
            start_epoch = 0
    for epoch in range(start_epoch, int(gan_cfg["open_classifier_epochs"])):
        model.train()
        model.encoder.eval()
        for batch_images, labels, _ in tqdm(loader, desc=f"open-head {epoch + 1}", leave=False):
            batch_images, labels = batch_images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = criterion(model(batch_images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        torch.save(
            {
                "generator_epochs": generator_epochs,
                "completed_epochs": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
            },
            progress_path,
        )

    open_path = run_dir / "open.pt"
    _save_checkpoint(model, config["model"], open_path, "open")
    progress_path.unlink(missing_ok=True)
    real_loader = make_loader(train_dataset, training, shuffle=False)
    logits, targets, _ = collect_logits(model, real_loader, device)
    predictions = logits.argmax(axis=1)
    openmax_cfg = config["openmax"]
    tail_sizes = [
        int(value)
        for value in openmax_cfg.get("tail_sizes", [openmax_cfg.get("tail_size", 20)])
    ]
    alphas = [int(value) for value in openmax_cfg.get("alphas", [openmax_cfg.get("alpha", 3)])]
    validation_known = SpectrogramDataset(manifest_path, "val")
    validation_unknown = SpectrogramDataset(manifest_path, "open_val")
    validation_dataset = ConcatDataset((validation_known, validation_unknown))
    validation_loader = make_loader(validation_dataset, training, shuffle=False)
    validation_logits, validation_targets, _ = collect_logits(model, validation_loader, device)
    validation_tensor = torch.from_numpy(validation_logits)
    selection: list[dict[str, Any]] = []
    fitted_candidates: dict[int, dict[str, Any]] = {}
    for tail_size in tail_sizes:
        candidate = fit_openmax(
            logits,
            targets,
            predictions,
            len(KNOWN_LABELS),
            tail_size,
        )
        fitted_candidates[tail_size] = candidate
        for alpha in alphas:
            calibrated = calibrate_openmax(validation_tensor, candidate, alpha)
            candidate_predictions = calibrated.argmax(dim=1).numpy()
            candidate_metrics = open_set_metrics(
                validation_targets, candidate_predictions, len(KNOWN_LABELS)
            )
            kar, uar = candidate_metrics["KAR"], candidate_metrics["UAR"]
            harmonic = 2.0 * kar * uar / max(kar + uar, 1e-12)
            selection.append(
                {
                    "tail_size": tail_size,
                    "alpha": alpha,
                    "harmonic_accuracy": harmonic,
                    **candidate_metrics,
                }
            )
    # The paper selects the tail length that "yields a balanced performance
    # between closed-set and open-set". Among candidates whose harmonic
    # accuracy is statistically indistinguishable from the best, prefer the
    # most balanced one (smallest KAR/UAR gap) rather than the raw maximum.
    top_harmonic = max(item["harmonic_accuracy"] for item in selection)
    near_ties = [
        item for item in selection if item["harmonic_accuracy"] >= top_harmonic - 0.005
    ]
    best = min(
        near_ties,
        key=lambda item: (item["GAP"], -item["harmonic_accuracy"]),
    )
    fitted = fitted_candidates[int(best["tail_size"])]
    fitted["alpha"] = int(best["alpha"])
    openmax_path = run_dir / "openmax.json"
    save_openmax(fitted, openmax_path)
    (run_dir / "openmax_selection.json").write_text(
        json.dumps({"selected": best, "candidates": selection}, indent=2), encoding="utf-8"
    )
    return open_path, openmax_path


def evaluate_open(
    config: dict[str, Any], manifest_path: str | Path, open_path: str | Path, openmax_path: str | Path
) -> dict[str, Any]:
    device = resolve_device()
    test_dataset = SpectrogramDataset(manifest_path, "test")
    loader = make_loader(test_dataset, config["training"], shuffle=False)
    model = load_checkpoint(open_path, device)
    fitted = load_openmax(openmax_path)
    all_predictions: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_class_indices: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for images, targets, class_indices in loader:
            logits = model(images.to(device))
            alpha = int(fitted.get("alpha", config["openmax"].get("alpha", 3)))
            probabilities = calibrate_openmax(logits, fitted, alpha)
            all_predictions.append(probabilities.argmax(dim=1).cpu().numpy())
            all_targets.append(targets.numpy())
            all_class_indices.append(class_indices.numpy())
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    class_indices = np.concatenate(all_class_indices)
    result = open_set_metrics(targets, predictions, len(KNOWN_LABELS))
    per_class: dict[str, float] = {}
    for class_index, label in enumerate(ALL_LABELS):
        mask = class_indices == class_index
        if not mask.any():
            continue
        expected = len(KNOWN_LABELS) if label not in KNOWN_LABELS else KNOWN_LABELS.index(label)
        per_class[label] = float((predictions[mask] == expected).mean())
    result["per_class_accuracy"] = per_class
    result_path = Path(config["run_dir"]) / "metrics.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
