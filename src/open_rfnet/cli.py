"""Command-line entry point for each reproducible pipeline stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from .constants import ALL_LABELS
from .data import inspect_dataset, prepare_dataset, resplit_manifest
from .model import build_model, parameter_count
from .training import (
    evaluate_closed,
    evaluate_open,
    finetune_closed,
    train_closed,
    train_gan_stage,
    train_open_stage,
)


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def manifest_path(config: dict[str, Any]) -> Path:
    return Path(config["cache_dir"]) / "manifest.json"


def run_reproduction(config: dict[str, Any]) -> dict[str, Any]:
    manifest = manifest_path(config)
    if not manifest.exists():
        manifest = prepare_dataset(config)
    closed = train_closed(config, manifest)
    generator = train_gan_stage(config, manifest, closed)
    open_model, openmax = train_open_stage(config, manifest, closed, generator)
    return evaluate_open(config, manifest, open_model, openmax)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open-RFNet clean-room reproduction")
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect", help="validate DroneRFa source files")
    inspect_parser.add_argument("--dataset", default="/home/coder/DroneRFa")
    for command in (
        "model-info",
        "prepare",
        "resplit",
        "train-closed",
        "finetune-closed",
        "evaluate-closed",
        "train-gan",
        "train-open",
        "evaluate",
        "reproduce",
    ):
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True)
    prepare_parser = subparsers.choices["prepare"]
    prepare_parser.add_argument("--labels", nargs="*", choices=ALL_LABELS)
    finetune_parser = subparsers.choices["finetune-closed"]
    finetune_parser.add_argument("--checkpoint")
    finetune_parser.add_argument("--output")
    evaluate_closed_parser = subparsers.choices["evaluate-closed"]
    evaluate_closed_parser.add_argument("--checkpoint")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "inspect":
        print(json.dumps(inspect_dataset(args.dataset), indent=2))
        return
    config = load_config(args.config)
    manifest = manifest_path(config)
    run_dir = Path(config["run_dir"])
    if args.command == "model-info":
        model = build_model(config["model"])
        print(json.dumps({"parameters": parameter_count(model), "parameter_size_mb_fp32": parameter_count(model) * 4 / 2**20}, indent=2))
    elif args.command == "prepare":
        print(prepare_dataset(config, labels=args.labels or None))
    elif args.command == "resplit":
        print(resplit_manifest(manifest, int(config.get("seed", 42))))
    elif args.command == "train-closed":
        print(train_closed(config, manifest))
    elif args.command == "finetune-closed":
        checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir / "closed.pt"
        output = Path(args.output) if args.output else run_dir / "closed-finetuned.pt"
        print(finetune_closed(config, manifest, checkpoint, output))
    elif args.command == "evaluate-closed":
        checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir / "closed.pt"
        print(json.dumps(evaluate_closed(config, manifest, checkpoint), indent=2))
    elif args.command == "train-gan":
        print(train_gan_stage(config, manifest, run_dir / "closed.pt"))
    elif args.command == "train-open":
        print(train_open_stage(config, manifest, run_dir / "closed.pt", run_dir / "generator.pt"))
    elif args.command == "evaluate":
        print(json.dumps(evaluate_open(config, manifest, run_dir / "open.pt", run_dir / "openmax.json"), indent=2))
    elif args.command == "reproduce":
        print(json.dumps(run_reproduction(config), indent=2))


if __name__ == "__main__":
    main()
