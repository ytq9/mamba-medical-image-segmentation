#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scan_geometry.models import ModelConfig, build_phase_b_model, count_parameters
from scan_geometry.phase_b import PhaseBConfig, TrainingConfig, run_training
from scan_geometry.phase_b.model_protocol import (
    condition_model_config_hash,
    condition_scan_order_hash,
    normalized_model_protocol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase B trainer adapter for one dataset-condition-seed run.")
    parser.add_argument("--config", default="configs/experiments/phase_b_10pct.yaml", help="Phase B YAML config.")
    parser.add_argument("--dataset", required=True, help="Dataset name from the Phase B config.")
    parser.add_argument("--condition", required=True, help="Condition name from the Phase B config.")
    parser.add_argument("--scan", default="", help="Scan order from the run matrix.")
    parser.add_argument("--seed", type=int, required=True, help="Training seed.")
    parser.add_argument("--split-file", required=True, help="Locked Phase B split CSV.")
    parser.add_argument("--output-dir", required=True, help="Run output directory.")
    parser.add_argument("--dry-run", action="store_true", help="Write metadata and instantiate the model, but do not train.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Override AdamW weight decay.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader workers.")
    parser.add_argument("--cache-dir", default=None, help="Override preprocessed tensor cache directory. Use an empty string to disable.")
    parser.add_argument("--device", default=None, help="Override device: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--max-slices-per-volume", type=int, default=None, help="Optional cap for slices expanded from each volume.")
    parser.add_argument(
        "--allow-mamba-fallback",
        action="store_true",
        help="Use a tiny sequence block if mamba_ssm is unavailable. Intended for smoke tests only.",
    )
    parser.add_argument(
        "--install-mamba-deps",
        action="store_true",
        help="Ensure mamba_ssm is installed before model construction. Intended for one-off SSH/H100 runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.install_mamba_deps:
        subprocess.check_call([sys.executable, str(ROOT / "scripts" / "ensure_mamba_ssm.py")])

    config = PhaseBConfig.from_yaml(args.config)
    dataset = _find_dataset(config, args.dataset)
    condition = _find_condition(config, args.condition)
    if str(condition.scan) != str(args.scan):
        raise SystemExit(f"Run matrix scan={args.scan!r} does not match config condition scan={condition.scan!r}.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    protocol = normalized_model_protocol(config)
    model_payload = dict(protocol)
    if args.allow_mamba_fallback:
        model_payload["allow_mamba_fallback"] = True
    model_config = ModelConfig.from_payload(
        model_payload,
        input_channels=dataset.input_channels,
        num_classes=dataset.num_classes,
    )
    model = build_phase_b_model(condition_family=condition.family, scan_order=condition.scan, config=model_config)
    param_summary = count_parameters(model)

    metadata = {
        "dataset": dataset.name,
        "condition": condition.name,
        "condition_family": condition.family,
        "scan": condition.scan,
        "seed": int(args.seed),
        "split_file": args.split_file,
        "output_dir": str(output_dir),
        "model": model_config.public_payload(),
        "model_config_hash": condition_model_config_hash(config, condition, dataset),
        "scan_order_hash": condition_scan_order_hash(config, condition),
        **param_summary,
    }
    (output_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output_dir / "model_summary.json").write_text(json.dumps(param_summary, indent=2) + "\n", encoding="utf-8")

    if args.dry_run:
        print(f"Dry run complete: {output_dir}")
        return

    training_config = _training_config_from_protocol({**protocol, "training": config.training}, args)
    print(
        "[run] "
        f"dataset={dataset.name} condition={condition.name} scan={condition.scan or 'CNN'} "
        f"seed={int(args.seed)} output_dir={output_dir}",
        flush=True,
    )
    outputs = run_training(
        model=model,
        split_file=args.split_file,
        output_dir=output_dir,
        dataset_name=dataset.name,
        condition_name=condition.name,
        seed=int(args.seed),
        input_channels=model_config.input_channels,
        num_classes=model_config.num_classes,
        config=training_config,
    )
    print(f"Training complete: {output_dir}")
    print(f"Checkpoint: {outputs.checkpoint}")
    print(f"Metrics: {outputs.metrics_csv}")


def _find_dataset(config: PhaseBConfig, name: str):
    for dataset in config.datasets:
        if dataset.name == name:
            return dataset
    raise SystemExit(f"Unknown dataset {name!r}.")


def _find_condition(config: PhaseBConfig, name: str):
    for condition in config.conditions:
        if condition.name == name:
            return condition
    raise SystemExit(f"Unknown condition {name!r}.")


def _training_config_from_protocol(protocol: dict, args: argparse.Namespace) -> TrainingConfig:
    training = dict(protocol.get("training", {}) or {})
    input_size = protocol.get("input_size", [256, 256])
    if len(input_size) != 2 or int(input_size[0]) != int(input_size[1]):
        raise SystemExit(f"Only square input_size is supported by the current trainer, got {input_size!r}.")
    cache_dir = args.cache_dir if args.cache_dir is not None else training.get("cache_dir")
    return TrainingConfig(
        epochs=int(args.epochs if args.epochs is not None else training.get("epochs", 200)),
        batch_size=int(args.batch_size if args.batch_size is not None else training.get("batch_size", 8)),
        learning_rate=float(args.learning_rate if args.learning_rate is not None else training.get("learning_rate", 1e-3)),
        weight_decay=float(args.weight_decay if args.weight_decay is not None else training.get("weight_decay", 1e-4)),
        num_workers=int(args.num_workers if args.num_workers is not None else training.get("num_workers", 0)),
        pin_memory=bool(training.get("pin_memory", True)),
        cache_dir=Path(str(cache_dir)) if cache_dir not in {None, ""} else None,
        target_size=int(input_size[0]),
        max_slices_per_volume=args.max_slices_per_volume
        if args.max_slices_per_volume is not None
        else training.get("max_slices_per_volume"),
        include_empty_slices=bool(training.get("include_empty_slices", False)),
        normalize=str(training.get("normalize", "zscore")),
        device=str(args.device if args.device is not None else training.get("device", "auto")),
    )


if __name__ == "__main__":
    main()
