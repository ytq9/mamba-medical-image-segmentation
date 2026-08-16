from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

MANIFEST_COLUMNS = [
    "dataset",
    "case_id",
    "patient_id",
    "image_path",
    "label_path",
    "modality",
    "task_type",
    "split",
    "is_2d",
    "spacing_z",
    "spacing_y",
    "spacing_x",
    "slice_axis",
    "view",
    "phase",
    "label_values",
    "notes",
]

REQUIRED_COLUMNS = {"dataset", "case_id", "patient_id", "image_path", "label_path"}


@dataclass(slots=True)
class ManifestRecord:
    dataset: str
    case_id: str
    patient_id: str
    image_path: str
    label_path: str
    modality: str = ""
    task_type: str = ""
    split: str = ""
    is_2d: bool = True
    spacing_z: float = 1.0
    spacing_y: float = 1.0
    spacing_x: float = 1.0
    slice_axis: int = -1
    view: str = ""
    phase: str = ""
    label_values: str = ""
    notes: str = ""

    @classmethod
    def from_row(cls, row: dict[str, str], manifest_dir: Path) -> "ManifestRecord":
        return cls(
            dataset=row.get("dataset", ""),
            case_id=row.get("case_id", ""),
            patient_id=row.get("patient_id", ""),
            image_path=_resolve_path(row.get("image_path", ""), manifest_dir),
            label_path=_resolve_path(row.get("label_path", ""), manifest_dir),
            modality=row.get("modality", ""),
            task_type=row.get("task_type", ""),
            split=row.get("split", ""),
            is_2d=_parse_bool(row.get("is_2d", "true")),
            spacing_z=_parse_float(row.get("spacing_z", "1.0"), 1.0),
            spacing_y=_parse_float(row.get("spacing_y", "1.0"), 1.0),
            spacing_x=_parse_float(row.get("spacing_x", "1.0"), 1.0),
            slice_axis=int(_parse_float(row.get("slice_axis", "-1"), -1.0)),
            view=row.get("view", ""),
            phase=row.get("phase", ""),
            label_values=row.get("label_values", ""),
            notes=row.get("notes", ""),
        )

    def to_row(self) -> dict[str, str]:
        return {
            "dataset": self.dataset,
            "case_id": self.case_id,
            "patient_id": self.patient_id,
            "image_path": self.image_path,
            "label_path": self.label_path,
            "modality": self.modality,
            "task_type": self.task_type,
            "split": self.split,
            "is_2d": "true" if self.is_2d else "false",
            "spacing_z": str(self.spacing_z),
            "spacing_y": str(self.spacing_y),
            "spacing_x": str(self.spacing_x),
            "slice_axis": str(self.slice_axis),
            "view": self.view,
            "phase": self.phase,
            "label_values": self.label_values,
            "notes": self.notes,
        }


def _parse_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value not in {None, ""} else default
    except ValueError:
        return default


def _resolve_path(value: str, manifest_dir: Path) -> str:
    if not value:
        return ""
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((manifest_dir / path).resolve())


def read_manifest(path: str | Path) -> list[ManifestRecord]:
    manifest_path = Path(path).resolve()
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fields
        if missing:
            raise ValueError(f"{manifest_path} is missing required columns: {sorted(missing)}")
        return [ManifestRecord.from_row(row, manifest_path.parent) for row in reader]


def write_manifest(path: str | Path, records: Iterable[ManifestRecord]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_row())


def notes_with_payload(**payload: object) -> str:
    clean = {
        key: value
        for key, value in payload.items()
        if value is not None and value != "" and value != [] and value != {}
    }
    return json.dumps(clean, sort_keys=True) if clean else ""
