from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from .utils import sanitize_id


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    name: str
    manifest: Path
    split_unit: str = "patient"
    modality: str = ""
    task_type: str = ""
    labeled_train_units: int | None = None
    expected_labeled_train_units: int | None = None
    input_channels: int | None = None
    num_classes: int | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "DatasetSpec":
        name = str(payload.get("name", "")).strip()
        if not name:
            raise ValueError("Every Phase B dataset entry requires a name.")
        manifest = payload.get("manifest")
        if not manifest:
            raise ValueError(f"Dataset {name} requires a manifest path.")
        split_unit = str(payload.get("split_unit", "patient")).strip().lower()
        if split_unit not in {"patient", "case"}:
            raise ValueError(f"Dataset {name} has unsupported split_unit={split_unit!r}.")
        return cls(
            name=name,
            manifest=Path(str(manifest)),
            split_unit=split_unit,
            modality=str(payload.get("modality", "")),
            task_type=str(payload.get("task_type", "")),
            labeled_train_units=_optional_int(payload.get("labeled_train_units")),
            expected_labeled_train_units=_optional_int(payload.get("expected_labeled_train_units")),
            input_channels=_optional_int(payload.get("input_channels")),
            num_classes=_optional_int(payload.get("num_classes")),
        )


@dataclass(frozen=True, slots=True)
class ConditionSpec:
    name: str
    family: str = ""
    scan: str = ""
    description: str = ""

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | str) -> "ConditionSpec":
        if isinstance(payload, str):
            return cls(
                name=payload,
                family="cnn_baseline" if payload == "CNN" else "mamba",
                scan="" if payload == "CNN" else payload,
            )
        name = str(payload.get("name", "")).strip()
        if not name:
            raise ValueError("Every Phase B condition entry requires a name.")
        return cls(
            name=name,
            family=str(payload.get("family", "")),
            scan=str(payload.get("scan", "")),
            description=str(payload.get("description", "")),
        )


@dataclass(frozen=True, slots=True)
class PhaseBConfig:
    datasets: list[DatasetSpec]
    conditions: list[ConditionSpec]
    seeds: list[int]
    output_dir: Path = Path("results/phase_b")
    phase_a_summary: Path | None = None
    split_seed: int = 2026
    train_fraction: float = 0.70
    val_fraction: float = 0.10
    test_fraction: float = 0.20
    low_label_ratio: float = 0.10
    min_labeled_train_units: int = 10
    model: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PhaseBConfig":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return cls.from_payload(payload)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "PhaseBConfig":
        datasets = [DatasetSpec.from_payload(item) for item in payload.get("datasets", [])]
        if not datasets:
            raise ValueError("Phase B config requires at least one dataset.")
        conditions = [ConditionSpec.from_payload(item) for item in payload.get("conditions", [])]
        if not conditions:
            raise ValueError("Phase B config requires at least one condition.")
        seeds = [int(seed) for seed in payload.get("seeds", [])]
        if not seeds:
            raise ValueError("Phase B config requires at least one seed.")

        split = payload.get("split", {}) or {}
        low_label = payload.get("low_label", {}) or {}
        phase_a_summary = payload.get("phase_a_summary")
        return cls(
            datasets=datasets,
            conditions=conditions,
            seeds=seeds,
            output_dir=Path(str(payload.get("output_dir", "results/phase_b"))),
            phase_a_summary=Path(str(phase_a_summary)) if phase_a_summary else None,
            split_seed=int(split.get("seed", 2026)),
            train_fraction=float(split.get("train", 0.70)),
            val_fraction=float(split.get("val", 0.10)),
            test_fraction=float(split.get("test", 0.20)),
            low_label_ratio=float(low_label.get("ratio", 0.10)),
            min_labeled_train_units=int(low_label.get("min_labeled_train_units", 10)),
            model=dict(payload.get("model", {}) or {}),
            training=dict(payload.get("training", {}) or {}),
        )

    def validate(self) -> None:
        total = self.train_fraction + self.val_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split fractions must sum to 1.0, got {total:.6f}.")
        if not 0.0 < self.low_label_ratio <= 1.0:
            raise ValueError("low_label.ratio must be in (0, 1].")

    def split_dir(self) -> Path:
        return self.output_dir / "splits"

    def run_dir(self) -> Path:
        return self.output_dir / "runs"

    def only_datasets(self, names: list[str] | tuple[str, ...] | None) -> "PhaseBConfig":
        if not names:
            return self
        requested = {_dataset_selector(name) for name in names}
        selected = [
            dataset
            for dataset in self.datasets
            if dataset.name.casefold() in requested or sanitize_id(dataset.name) in requested
        ]
        if len(selected) != len(requested):
            matched = {dataset.name.casefold() for dataset in selected} | {sanitize_id(dataset.name) for dataset in selected}
            missing = sorted(requested - matched)
            available = ", ".join(dataset.name for dataset in self.datasets)
            raise ValueError(f"Unknown Phase B dataset selector(s): {missing}. Available datasets: {available}")
        return replace(self, datasets=selected)

    def only_seeds(self, seeds: list[int] | tuple[int, ...] | None) -> "PhaseBConfig":
        if not seeds:
            return self
        requested = [int(seed) for seed in seeds]
        available = set(self.seeds)
        missing = sorted(set(requested) - available)
        if missing:
            raise ValueError(f"Unknown Phase B seed(s): {missing}. Available seeds: {self.seeds}")
        selected = [seed for seed in self.seeds if seed in set(requested)]
        return replace(self, seeds=selected)

    def protocol_payload(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "phase_a_summary": "" if self.phase_a_summary is None else str(self.phase_a_summary),
            "split": {
                "seed": self.split_seed,
                "train": self.train_fraction,
                "val": self.val_fraction,
                "test": self.test_fraction,
            },
            "low_label": {
                "ratio": self.low_label_ratio,
                "min_labeled_train_units": self.min_labeled_train_units,
            },
            "seeds": self.seeds,
            "model": self.model,
            "training": self.training,
            "conditions": [
                {
                    "name": item.name,
                    "family": item.family,
                    "scan": item.scan,
                    "description": item.description,
                }
                for item in self.conditions
            ],
            "datasets": [
                {
                    "name": item.name,
                    "manifest": str(item.manifest),
                    "split_unit": item.split_unit,
                    "modality": item.modality,
                    "task_type": item.task_type,
                    "labeled_train_units": item.labeled_train_units,
                    "expected_labeled_train_units": item.expected_labeled_train_units,
                    "input_channels": item.input_channels,
                    "num_classes": item.num_classes,
                }
                for item in self.datasets
            ],
        }


def _optional_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    return int(value)


def _dataset_selector(value: str) -> str:
    text = str(value).strip()
    return sanitize_id(text) if sanitize_id(text) != text.casefold() else text.casefold()
