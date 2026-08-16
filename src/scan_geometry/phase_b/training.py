from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import ndimage as ndi
from torch import nn
from torch.utils.data import DataLoader

from .dataset import ManifestSegmentationDataset


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 200
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    pin_memory: bool = True
    cache_dir: Path | None = None
    target_size: int = 256
    max_slices_per_volume: int | None = None
    include_empty_slices: bool = False
    normalize: str = "zscore"
    device: str = "auto"


@dataclass(frozen=True, slots=True)
class TrainingOutputs:
    checkpoint: Path
    metrics_csv: Path
    history_json: Path


def run_training(
    *,
    model: nn.Module,
    split_file: str | Path,
    output_dir: str | Path,
    dataset_name: str,
    condition_name: str,
    seed: int,
    input_channels: int,
    num_classes: int,
    config: TrainingConfig,
) -> TrainingOutputs:
    set_reproducible_seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "checkpoints").mkdir(exist_ok=True)

    device = _resolve_device(config.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model.to(device)

    train_set = ManifestSegmentationDataset(
        split_file,
        split="train",
        labeled_only=True,
        input_channels=input_channels,
        num_classes=num_classes,
        target_size=config.target_size,
        include_empty_slices=config.include_empty_slices,
        max_slices_per_volume=config.max_slices_per_volume,
        normalize=config.normalize,
        cache_dir=config.cache_dir,
    )
    val_set = ManifestSegmentationDataset(
        split_file,
        split="val",
        input_channels=input_channels,
        num_classes=num_classes,
        target_size=config.target_size,
        include_empty_slices=config.include_empty_slices,
        max_slices_per_volume=config.max_slices_per_volume,
        normalize=config.normalize,
        cache_dir=config.cache_dir,
    )
    test_set = ManifestSegmentationDataset(
        split_file,
        split="test",
        input_channels=input_channels,
        num_classes=num_classes,
        target_size=config.target_size,
        include_empty_slices=config.include_empty_slices,
        max_slices_per_volume=config.max_slices_per_volume,
        normalize=config.normalize,
        cache_dir=config.cache_dir,
    )
    if len(train_set) == 0:
        raise ValueError(f"No labeled training samples found in {split_file}.")
    if len(test_set) == 0:
        raise ValueError(f"No test samples found in {split_file}.")

    print(
        "[train] "
        f"dataset={dataset_name} condition={condition_name} seed={seed} "
        f"device={device} train_samples={len(train_set)} val_samples={len(val_set)} "
        f"test_samples={len(test_set)} epochs={config.epochs} batch_size={config.batch_size} "
        f"num_workers={config.num_workers} cache_dir={config.cache_dir or ''}",
        flush=True,
    )

    loader_kwargs = _loader_kwargs(config, device)
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history: list[dict[str, float | int]] = []
    best_score = -np.inf
    best_state = None
    loss_fn = DiceCrossEntropyLoss(num_classes=num_classes)

    for epoch in range(1, int(config.epochs) + 1):
        epoch_start = time.perf_counter()
        train_loss = _train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_score = _mean_macro_dice(model, val_loader, num_classes, device) if len(val_set) else float("nan")
        selector = val_score if np.isfinite(val_score) else -train_loss
        history.append({"epoch": epoch, "train_loss": float(train_loss), "val_macro_dice": float(val_score)})
        if selector > best_score:
            best_score = float(selector)
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        print(
            "[epoch] "
            f"dataset={dataset_name} condition={condition_name} seed={seed} "
            f"epoch={epoch}/{config.epochs} train_loss={_format_metric(train_loss)} "
            f"val_macro_dice={_format_metric(val_score)} best={_format_metric(best_score)} "
            f"elapsed_s={time.perf_counter() - epoch_start:.1f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    checkpoint = output_path / "checkpoints" / "best.pt"
    torch.save({"model_state": model.state_dict(), "seed": seed, "history": history}, checkpoint)

    history_json = output_path / "history.json"
    history_json.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    metrics_csv = output_path / "metrics.csv"
    rows = evaluate_case_metrics(
        model,
        test_loader,
        dataset_name=dataset_name,
        condition_name=condition_name,
        seed=seed,
        num_classes=num_classes,
        device=device,
    )
    write_metric_rows(metrics_csv, rows)
    return TrainingOutputs(checkpoint=checkpoint, metrics_csv=metrics_csv, history_json=history_json)


def _format_metric(value: float) -> str:
    return "nan" if not np.isfinite(float(value)) else f"{float(value):.6f}"


def _loader_kwargs(config: TrainingConfig, device: torch.device) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "num_workers": int(config.num_workers),
        "pin_memory": bool(config.pin_memory and device.type == "cuda"),
    }
    if int(config.num_workers) > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return kwargs


class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes: int, dice_weight: float = 1.0, ce_weight: float = 1.0) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.dice_weight = float(dice_weight)
        self.ce_weight = float(ce_weight)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = self.ce(logits, target)
        probs = torch.softmax(logits, dim=1)
        target_one_hot = torch.nn.functional.one_hot(target.clamp_min(0), num_classes=self.num_classes).permute(0, 3, 1, 2)
        target_one_hot = target_one_hot.to(dtype=probs.dtype, device=probs.device)
        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_one_hot, dim=dims)
        denominator = torch.sum(probs + target_one_hot, dim=dims)
        dice = (2.0 * intersection + 1e-5) / (denominator + 1e-5)
        foreground = dice[1:] if self.num_classes > 1 else dice
        dice_loss = 1.0 - foreground.mean()
        return self.ce_weight * ce + self.dice_weight * dice_loss


def evaluate_case_metrics(
    model: nn.Module,
    loader: DataLoader,
    *,
    dataset_name: str,
    condition_name: str,
    seed: int,
    num_classes: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device=device, dtype=torch.float32)
            labels = batch["label"].to(device=device, dtype=torch.long)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            targets = labels.cpu().numpy()
            case_ids = list(batch["case_id"])
            slice_indices = batch["slice_index"].cpu().numpy() if torch.is_tensor(batch["slice_index"]) else batch["slice_index"]
            for item_index, case_id in enumerate(case_ids):
                for class_id in range(1, int(num_classes)):
                    pred_mask = preds[item_index] == class_id
                    target_mask = targets[item_index] == class_id
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "condition": condition_name,
                            "seed": int(seed),
                            "case_id": str(case_id),
                            "slice_index": int(slice_indices[item_index]),
                            "class_id": int(class_id),
                            "dice": dice_score(pred_mask, target_mask),
                            "hd95": hd95(pred_mask, target_mask),
                        }
                    )
    return rows


def write_metric_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "condition", "seed", "case_id", "slice_index", "class_id", "dice", "hd95"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)
    denom = int(pred.sum()) + int(target.sum())
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, target).sum() / denom)


def hd95(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        return float("inf")
    pred_surface = pred ^ ndi.binary_erosion(pred)
    target_surface = target ^ ndi.binary_erosion(target)
    target_distance = ndi.distance_transform_edt(~target_surface)
    pred_distance = ndi.distance_transform_edt(~pred_surface)
    distances = np.concatenate([target_distance[pred_surface], pred_distance[target_surface]])
    if distances.size == 0:
        return 0.0
    return float(np.percentile(distances, 95))


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    losses = []
    for batch in loader:
        non_blocking = device.type == "cuda"
        images = batch["image"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
        labels = batch["label"].to(device=device, dtype=torch.long, non_blocking=non_blocking)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def _mean_macro_dice(model: nn.Module, loader: DataLoader, num_classes: int, device: torch.device) -> float:
    if len(loader.dataset) == 0:
        return float("nan")
    rows = evaluate_case_metrics(
        model,
        loader,
        dataset_name="",
        condition_name="",
        seed=0,
        num_classes=num_classes,
        device=device,
    )
    values = [float(row["dice"]) for row in rows if np.isfinite(float(row["dice"]))]
    return float(np.mean(values)) if values else float("nan")


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)
