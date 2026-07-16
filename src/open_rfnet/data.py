"""DroneRFa discovery, denoising, STFT caching, and PyTorch datasets."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from .constants import (
    ALL_LABELS,
    CLASS_INFO,
    INFO_BY_LABEL,
    KNOWN_INDEX,
    KNOWN_LABELS,
    UNKNOWN_LABELS,
    label_from_filename,
    paper_channels_for_file,
    paper_uses_file,
)


@dataclass(frozen=True)
class Source:
    path: Path
    label: str
    channel: str
    length: int


def _safe_source(path: Path, label: str, channel: str) -> Source | None:
    try:
        with h5py.File(path, "r") as handle:
            i_name, q_name = f"{channel}_I", f"{channel}_Q"
            if i_name not in handle or q_name not in handle:
                return None
            length = min(handle[i_name].shape[-1], handle[q_name].shape[-1])
        return Source(path=path, label=label, channel=channel, length=int(length))
    except OSError:
        return None


def discover_sources(dataset_root: str | Path, labels: Iterable[str] = ALL_LABELS) -> tuple[dict[str, list[Source]], list[str]]:
    """Discover paper-selected files and skip unreadable/truncated HDF5 files."""
    root = Path(dataset_root)
    wanted = set(labels)
    grouped: dict[str, list[Source]] = defaultdict(list)
    skipped: list[str] = []
    for path in sorted(root.glob("*.mat")):
        label = label_from_filename(path.name)
        if label not in wanted or label not in INFO_BY_LABEL:
            continue
        if not paper_uses_file(path.name, label):
            continue
        for channel in paper_channels_for_file(path.name, label):
            source = _safe_source(path, label, channel)
            if source is None:
                skipped.append(f"{path.name}:{channel}")
            else:
                grouped[label].append(source)
    return dict(grouped), skipped


def inspect_dataset(dataset_root: str | Path) -> dict[str, Any]:
    sources, skipped = discover_sources(dataset_root)
    return {
        "dataset_root": str(Path(dataset_root).resolve()),
        "usable_source_channels": sum(map(len, sources.values())),
        "usable_files": len({str(s.path) for values in sources.values() for s in values}),
        "per_class_source_channels": {label: len(sources.get(label, [])) for label in ALL_LABELS},
        "skipped": skipped,
    }


class STFTTransform:
    """Convert one 3 ms complex I/Q slice to a normalized spectrogram."""

    def __init__(self, config: dict[str, Any], device: torch.device):
        self.n_fft = int(config["n_fft"])
        self.win_length = int(config.get("win_length", self.n_fft))
        self.hop_length = int(config["hop_length"])
        self.output_size = int(config["output_size"])
        self.device = device
        self.window = torch.hann_window(self.win_length, device=device)

    @torch.inference_mode()
    def __call__(self, iq: np.ndarray) -> np.ndarray:
        signal = torch.as_tensor(iq, dtype=torch.complex64, device=self.device)
        spectrum = torch.stft(
            signal,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=True,
            onesided=False,
        )
        spectrum = torch.fft.fftshift(spectrum, dim=0)
        power_db = 20.0 * torch.log10(spectrum.abs().clamp_min(1e-8))
        low, high = power_db.amin(), power_db.amax()
        normalized = (power_db - low) / (high - low).clamp_min(1e-8)
        normalized = F.interpolate(
            normalized[None, None],
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        return normalized.to(dtype=torch.float16, device="cpu").numpy()


def _read_complex(handle: h5py.File, channel: str, start: int, length: int) -> np.ndarray:
    end = start + length
    i = np.asarray(handle[f"{channel}_I"][..., start:end], dtype=np.float32).reshape(-1)
    q = np.asarray(handle[f"{channel}_Q"][..., start:end], dtype=np.float32).reshape(-1)
    return i + 1j * q


def _complex_rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal.real) + np.square(signal.imag))))


def _even_offsets(length: int, chunk: int, count: int) -> np.ndarray:
    if length < chunk or count <= 0:
        return np.empty(0, dtype=np.int64)
    capacity = length // chunk
    count = min(count, capacity)
    raw = np.linspace(0, length - chunk, num=count, dtype=np.int64)
    aligned = (raw // chunk) * chunk
    return np.unique(aligned)


def _probe_rms(source: Source, sub_length: int, probes: int = 64) -> np.ndarray:
    """Sample evenly spaced sub-slices and return their RMS values."""
    offsets = _even_offsets(source.length, sub_length, probes)
    if not len(offsets):
        return np.empty(0, dtype=np.float64)
    values: list[float] = []
    try:
        with h5py.File(source.path, "r") as handle:
            for offset in offsets:
                chunk = _read_complex(handle, source.channel, int(offset), sub_length)
                values.append(_complex_rms(chunk))
    except OSError:
        return np.empty(0, dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


def _adaptive_threshold(
    probe_rms: np.ndarray,
    absolute_threshold: float,
    floor_percentile: float,
    floor_factor: float | None,
) -> float:
    """Per-capture denoise threshold anchored to that capture's own noise floor.

    Receiver gain varies by more than 20 dB across DroneRFa captures, so a single
    absolute threshold either passes pure noise (high-gain captures) or rejects
    everything (low-gain captures). The capture's low-percentile sub-slice RMS
    estimates its noise floor; sub-slices must exceed it by ``floor_factor``.
    """
    if floor_factor is None or not len(probe_rms):
        return absolute_threshold
    floor = float(np.percentile(probe_rms, floor_percentile))
    return max(absolute_threshold, floor * float(floor_factor))


def assign_splits(records: list[dict[str, Any]], data_cfg: dict[str, Any], seed: int) -> None:
    """Assign the paper's known/unknown protocol without training on unknowns.

    Known classes use train/validation/test partitions. Unknown classes use a
    held-out open-set validation partition and a test partition of the same
    configured size as the known test split; all remaining unknown examples
    stay unused. This avoids the previous four-times-larger unknown test set
    and permits the paper-described validation of the Weibull tail length.
    """
    rng = np.random.default_rng(seed)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        record.pop("split", None)
        record.pop("target", None)
        by_label[record["label"]].append(record)

    test_fraction = float(data_cfg.get("test_fraction", 0.25))
    validation_fraction = float(data_cfg.get("validation_fraction", 0.10))
    unknown_validation_fraction = float(
        data_cfg.get("unknown_validation_fraction", validation_fraction)
    )
    for label, label_records in by_label.items():
        order = rng.permutation(len(label_records))
        test_count = max(1, round(len(order) * test_fraction))
        remaining = len(order) - test_count
        if label in UNKNOWN_LABELS:
            open_val_count = (
                max(1, round(remaining * unknown_validation_fraction)) if remaining >= 3 else 0
            )
            for position, idx in enumerate(order):
                record = label_records[int(idx)]
                record["target"] = len(KNOWN_LABELS)
                if position < test_count:
                    record["split"] = "test"
                elif position < test_count + open_val_count:
                    record["split"] = "open_val"
                else:
                    record["split"] = "unused"
            continue

        val_count = max(1, round(remaining * validation_fraction)) if remaining >= 3 else 0
        for position, idx in enumerate(order):
            record = label_records[int(idx)]
            record["target"] = KNOWN_INDEX[label]
            if position < test_count:
                record["split"] = "test"
            elif position < test_count + val_count:
                record["split"] = "val"
            else:
                record["split"] = "train"


def resplit_manifest(manifest_path: str | Path, seed: int = 42) -> Path:
    """Update an existing cache manifest to the current split protocol."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assign_splits(manifest["records"], manifest["config"], seed)
    manifest["version"] = 2
    manifest["split_counts"] = dict(
        sorted(
            (split, sum(record["split"] == split for record in manifest["records"]))
            for split in {record["split"] for record in manifest["records"]}
        )
    )
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def prepare_dataset(config: dict[str, Any], labels: Iterable[str] | None = None) -> Path:
    """Build resumable, memory-mapped float16 spectrogram caches.

    The paper's unknown implementation choices are deliberately surfaced in the
    YAML configuration. The cache manifest records every source offset.
    """
    seed = int(config.get("seed", 42))
    dataset_root = Path(config["dataset_root"])
    cache_dir = Path(config["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_cfg = config["data"]
    labels = tuple(labels or ALL_LABELS)
    max_per_class = int(data_cfg["max_samples_per_class"])
    target_length = int(data_cfg["sample_length"])
    sub_length = int(data_cfg["sub_slice_length"])
    if target_length % sub_length:
        raise ValueError("sample_length must be divisible by sub_slice_length")
    sub_slices_per_sample = target_length // sub_length
    threshold = 10.0 ** (float(data_cfg["noise_threshold_dbfs"]) / 20.0)
    denoise = bool(data_cfg.get("denoise", True))
    candidate_multiplier = float(data_cfg.get("candidate_multiplier", 2.0))
    occupancy_probes = int(data_cfg.get("occupancy_probes", 64))
    floor_percentile = float(data_cfg.get("noise_floor_percentile", 10.0))
    floor_factor = data_cfg.get("noise_floor_factor")
    if floor_factor is not None:
        floor_factor = float(floor_factor)

    requested_device = str(data_cfg.get("preprocessing_device", "cpu"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    transform = STFTTransform(data_cfg["stft"], device)
    output_size = transform.output_size

    sources_by_label, skipped = discover_sources(dataset_root, labels)
    records: list[dict[str, Any]] = []
    counts: dict[str, int] = {}

    source_thresholds: dict[str, float] = {}
    for label in labels:
        sources = sources_by_label.get(label, [])
        if not sources:
            counts[label] = 0
            continue
        thresholds: dict[Source, float] = {source: 0.0 for source in sources}
        if denoise and label != "T0000":
            keep_fractions: dict[Source, float] = {}
            for source in sources:
                probe = _probe_rms(source, sub_length, occupancy_probes)
                thresholds[source] = _adaptive_threshold(
                    probe, threshold, floor_percentile, floor_factor
                )
                keep_fractions[source] = (
                    float(np.mean(probe >= thresholds[source])) if len(probe) else 0.0
                )
            # Process low-occupancy captures first so later signal-rich captures
            # absorb their unused per-source quota without discarding weak
            # captures entirely.
            sources = sorted(sources, key=keep_fractions.__getitem__)
        for source in sources:
            source_thresholds[f"{source.path.name}:{source.channel}"] = thresholds[source]
        cache_path = cache_dir / f"{label}.npy"
        cache = np.lib.format.open_memmap(
            cache_path,
            mode="w+",
            dtype=np.float16,
            shape=(max_per_class, 1, output_size, output_size),
        )
        written = 0
        quota = math.ceil(max_per_class / len(sources))
        quota_deficit = 0
        progress = tqdm(total=max_per_class, desc=f"prepare {label}", unit="sample")
        for source in sources:
            if written >= max_per_class:
                break
            source_target = min(quota + quota_deficit, max_per_class - written)
            # Probe broadly enough for the denoiser to reject receiver noise.
            # Strong captures processed later absorb any quota left by weak
            # captures, preserving class balance after filtering.
            candidate_count = math.ceil(
                source_target * sub_slices_per_sample * candidate_multiplier
            )
            offsets = _even_offsets(source.length, sub_length, candidate_count)
            accepted: list[tuple[np.ndarray, int, float]] = []
            made_from_source = 0
            source_threshold = thresholds[source]
            try:
                with h5py.File(source.path, "r") as handle:
                    for offset in offsets:
                        chunk = _read_complex(handle, source.channel, int(offset), sub_length)
                        rms = _complex_rms(chunk)
                        # Background is a real class; filtering it by a UAV
                        # signal threshold would erase its defining examples.
                        keep = (not denoise) or label == "T0000" or rms >= source_threshold
                        if not keep:
                            continue
                        accepted.append((chunk, int(offset), rms))
                        if len(accepted) < sub_slices_per_sample:
                            continue
                        parts = accepted[:sub_slices_per_sample]
                        del accepted[:sub_slices_per_sample]
                        iq = np.concatenate([part[0] for part in parts])
                        cache[written, 0] = transform(iq)
                        records.append(
                            {
                                "cache": cache_path.name,
                                "index": written,
                                "label": label,
                                "all_class_index": ALL_LABELS.index(label),
                                "source": source.path.name,
                                "channel": source.channel,
                                "offsets": [part[1] for part in parts],
                                "rms": [part[2] for part in parts],
                            }
                        )
                        written += 1
                        made_from_source += 1
                        progress.update(1)
                        if written >= max_per_class or made_from_source >= source_target:
                            break
            except OSError:
                skipped.append(f"{source.path.name}:{source.channel}")
            quota_deficit = max(0, source_target - made_from_source)
        progress.close()
        cache.flush()
        counts[label] = written

    assign_splits(records, data_cfg, seed)
    split_counts = {
        split: sum(record["split"] == split for record in records)
        for split in sorted({record["split"] for record in records})
    }

    manifest = {
        "version": 2,
        "dataset_root": str(dataset_root),
        "cache_dir": str(cache_dir),
        "known_labels": list(KNOWN_LABELS),
        "unknown_labels": list(UNKNOWN_LABELS),
        "counts": counts,
        "skipped": sorted(set(skipped)),
        "config": data_cfg,
        "source_thresholds": source_thresholds,
        "split_counts": split_counts,
        "records": records,
    }
    manifest_path = cache_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


class SpectrogramDataset(Dataset[tuple[Tensor, int, int]]):
    def __init__(self, manifest_path: str | Path, split: str):
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.records = [record for record in manifest["records"] if record["split"] == split]
        self._arrays: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _array(self, name: str) -> np.ndarray:
        if name not in self._arrays:
            self._arrays[name] = np.load(self.root / name, mmap_mode="r")
        return self._arrays[name]

    def __getitem__(self, index: int) -> tuple[Tensor, int, int]:
        record = self.records[index]
        image = np.array(self._array(record["cache"])[record["index"]], dtype=np.float32, copy=True)
        return torch.from_numpy(image), int(record["target"]), int(record["all_class_index"])


class SpectrogramAugment:
    """Two-view augmentations; the paper does not disclose its exact policy.

    Position shifts are optional because Open-RFNet explicitly relies on the
    absolute time-frequency position of signal blocks. Large random rolls can
    therefore erase the feature that the Transformer branches are meant to
    learn.
    """

    def __init__(
        self,
        mask_fraction: float = 0.0,
        shift_fraction: float = 0.0,
        noise_std: float = 0.005,
        gain_min: float = 0.95,
        gain_max: float = 1.05,
    ):
        self.mask_fraction = float(mask_fraction)
        self.shift_fraction = float(shift_fraction)
        self.noise_std = float(noise_std)
        self.gain_min = float(gain_min)
        self.gain_max = float(gain_max)

    def __call__(self, batch: Tensor) -> Tensor:
        output = batch.clone()
        b, _, height, width = output.shape
        gains = torch.empty((b, 1, 1, 1), device=output.device).uniform_(self.gain_min, self.gain_max)
        output = output * gains
        if self.noise_std > 0:
            output = output + torch.randn_like(output) * self.noise_std
        max_shift_h = round(height * self.shift_fraction)
        max_shift_w = round(width * self.shift_fraction)
        max_mask_h = round(height * self.mask_fraction)
        max_mask_w = round(width * self.mask_fraction)
        for i in range(b):
            if max_shift_h or max_shift_w:
                shift_h = int(torch.randint(-max_shift_h, max_shift_h + 1, ()).item())
                shift_w = int(torch.randint(-max_shift_w, max_shift_w + 1, ()).item())
                output[i] = torch.roll(output[i], shifts=(shift_h, shift_w), dims=(-2, -1))
            if max_mask_h:
                h0 = int(torch.randint(0, max(1, height - max_mask_h + 1), ()).item())
                output[i, :, h0 : h0 + max_mask_h, :] = 0
            if max_mask_w:
                w0 = int(torch.randint(0, max(1, width - max_mask_w + 1), ()).item())
                output[i, :, :, w0 : w0 + max_mask_w] = 0
        return output.clamp_(0.0, 1.0)
