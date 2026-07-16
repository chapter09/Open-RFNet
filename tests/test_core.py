from __future__ import annotations

import numpy as np
import torch

from open_rfnet.constants import (
    ALL_LABELS,
    KNOWN_LABELS,
    UNKNOWN_LABELS,
    label_from_filename,
    paper_channels_for_file,
    paper_uses_file,
)
from open_rfnet.data import STFTTransform, _adaptive_threshold, assign_splits
from open_rfnet.losses import SupervisedContrastiveLoss
from open_rfnet.metrics import open_set_metrics
from open_rfnet.model import build_model
from open_rfnet.openmax import calibrate_openmax, fit_openmax


def tiny_model_config() -> dict:
    return {
        "input_size": 32,
        "resnet_base_width": 2,
        "transformer_dim": 8,
        "transformer_heads": 2,
        "transformer_layers": 1,
        "transformer_ff_dim": 16,
        "position_dim": 8,
        "fused_dim": 16,
        "projection_hidden_dim": 12,
        "projection_dim": 6,
        "dropout": 0.0,
    }


def test_variable_width_label_and_paper_file_selection() -> None:
    assert label_from_filename("T11000_S0010.mat") == "T11000"
    assert paper_uses_file("T0010_D00_S0000.mat", "T0010")
    assert not paper_uses_file("T0010_D01_S0000.mat", "T0010")
    assert paper_uses_file("T11000_S0000.mat", "T11000")


def test_dual_band_files_use_only_the_active_receiver() -> None:
    assert paper_channels_for_file("T0010_D00_S0111.mat", "T0010") == ("RF0",)
    assert paper_channels_for_file("T0010_D00_S1110.mat", "T0010") == ("RF1",)
    assert paper_channels_for_file("T10010_S0000.mat", "T10010") == ("RF0",)
    assert paper_channels_for_file("T10010_S1000.mat", "T10010") == ("RF1",)
    assert paper_channels_for_file("T0100_D00_S1100.mat", "T0100") == ("RF0",)
    assert paper_channels_for_file("T0000_D00_S1000.mat", "T0000") == ("RF0", "RF1")


def test_stft_shape_range() -> None:
    transform = STFTTransform(
        {"n_fft": 64, "win_length": 64, "hop_length": 16, "output_size": 32},
        torch.device("cpu"),
    )
    time = np.arange(2048, dtype=np.float32)
    iq = np.exp(2j * np.pi * 0.12 * time).astype(np.complex64)
    output = transform(iq)
    assert output.shape == (32, 32)
    assert np.isfinite(output).all()
    assert output.min() >= 0 and output.max() <= 1


def test_model_and_supcon_forward() -> None:
    model = build_model(tiny_model_config(), num_classes=3)
    images = torch.rand(4, 1, 32, 32)
    logits, projection, feature = model(images, return_projection=True)
    assert logits.shape == (4, 3)
    assert projection.shape == (4, 6)
    assert feature.shape == (4, 16)
    views = torch.stack((projection, projection.roll(1, dims=0)), dim=1)
    loss = SupervisedContrastiveLoss(0.1)(views, torch.tensor([0, 0, 1, 1]))
    assert torch.isfinite(loss)


def test_openmax_and_metrics() -> None:
    logits = np.array(
        [
            [5.0, 0.1, -1.0],
            [4.7, 0.2, -1.0],
            [0.1, 5.1, -1.0],
            [0.2, 4.8, -1.0],
        ],
        dtype=np.float32,
    )
    targets = np.array([0, 0, 1, 1])
    fitted = fit_openmax(logits, targets, logits.argmax(1), num_known=2, tail_size=2)
    probabilities = calibrate_openmax(torch.from_numpy(logits), fitted, alpha=2)
    assert probabilities.shape == (4, 3)
    assert torch.allclose(probabilities.sum(1), torch.ones(4), atol=1e-5)
    metrics = open_set_metrics(np.array([0, 1, 2, 2]), np.array([0, 1, 2, 0]), 2)
    assert metrics["KAR"] == 1.0
    assert metrics["UAR"] == 0.5


def test_adaptive_threshold_tracks_capture_noise_floor() -> None:
    low_gain = np.full(64, 0.0025)
    high_gain = np.full(64, 0.034)
    absolute = 10.0 ** (-51.0 / 20.0)
    # Legacy behaviour when no factor is configured.
    assert _adaptive_threshold(low_gain, absolute, 10.0, None) == absolute
    # The floor of a high-gain capture dominates the absolute threshold.
    assert _adaptive_threshold(high_gain, absolute, 10.0, 1.4) == 0.034 * 1.4
    # A low-gain capture still respects the absolute lower bound.
    assert _adaptive_threshold(low_gain, absolute, 10.0, 1.0) == absolute
    # Missing probes fall back to the absolute threshold.
    assert _adaptive_threshold(np.empty(0), absolute, 10.0, 1.4) == absolute


def test_paper_split_has_twenty_known_classes() -> None:
    assert len(KNOWN_LABELS) == 20


def test_unknown_examples_are_held_out_with_paper_sized_test_split() -> None:
    records = [{"label": label, "index": index} for label in ALL_LABELS for index in range(1000)]
    assign_splits(
        records,
        {"test_fraction": 0.25, "validation_fraction": 0.10},
        seed=42,
    )
    for label in UNKNOWN_LABELS:
        class_records = [record for record in records if record["label"] == label]
        assert sum(record["split"] == "test" for record in class_records) == 250
        assert sum(record["split"] == "open_val" for record in class_records) == 75
        assert not any(record["split"] == "train" for record in class_records)
        assert {record["target"] for record in class_records} == {len(KNOWN_LABELS)}
