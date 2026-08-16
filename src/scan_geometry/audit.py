from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .data.loaders import label_values_from_mask, load_image_and_label, shapes_compatible
from .data.manifest import MANIFEST_COLUMNS, ManifestRecord, read_manifest
from .geometry.coverage import geometry_pca
from .geometry.descriptors import (
    DESCRIPTOR_COLUMNS,
    aggregate_descriptors,
    axial_angle_mean,
    descriptors_for_mask_stack,
)
from .geometry.scan_profiles import matching_contrast, matching_table
from .geometry.stability import bootstrap_dataset_ci, resize_stability_delta


@dataclass(slots=True)
class AuditConfig:
    manifests: list[Path]
    output_dir: Path = Path("results/dataset_audit")
    target_size: int = 256
    bootstrap_samples: int = 1000
    max_cases: int | None = None
    min_patients: int = 10
    seed: int = 2026

    @classmethod
    def from_args(cls, args: Any) -> "AuditConfig":
        payload: dict[str, Any] = {}
        if getattr(args, "config", None):
            with Path(args.config).open("r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle) or {}
        manifests = getattr(args, "manifest", None) or payload.get("manifests") or []
        if not manifests:
            raise ValueError("Provide --manifest or a YAML config with a manifests list.")
        return cls(
            manifests=[Path(item) for item in manifests],
            output_dir=Path(getattr(args, "output_dir", None) or payload.get("output_dir", "results/dataset_audit")),
            target_size=int(getattr(args, "target_size", None) or payload.get("target_size", 256)),
            bootstrap_samples=int(getattr(args, "bootstrap_samples", None) or payload.get("bootstrap_samples", 1000)),
            max_cases=getattr(args, "max_cases", None) if getattr(args, "max_cases", None) is not None else payload.get("max_cases"),
            min_patients=int(getattr(args, "min_patients", None) or payload.get("min_patients", 10)),
            seed=int(getattr(args, "seed", None) or payload.get("seed", 2026)),
        )


@dataclass(slots=True)
class AuditOutputs:
    output_dir: Path
    suitability_summary: Path


def run_audit(config: AuditConfig) -> AuditOutputs:
    output_dir = config.output_dir

    records = _load_records(config.manifests, max_cases=config.max_cases)
    manifest_rows: list[dict[str, Any]] = []
    case_rows_256: list[dict[str, Any]] = []
    case_rows_512: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        validated, case_256, case_512 = _audit_record(record, config.target_size, alternate_size=512)
        validated["manifest_index"] = index
        manifest_rows.append(validated)
        if case_256 is not None:
            case_rows_256.append(case_256)
        if case_512 is not None:
            case_rows_512.append(case_512)

    dataset_rows = _dataset_geometry(case_rows_256, manifest_rows)
    dataset_rows_512 = _dataset_geometry(case_rows_512, manifest_rows)
    stability_rows = _stability_rows(dataset_rows, dataset_rows_512, case_rows_256, config)
    pca_rows = geometry_pca(dataset_rows)
    matching_rows = _matching_rows(dataset_rows)
    suitability_rows = _suitability_rows(dataset_rows, stability_rows, matching_rows, pca_rows, config)

    _write_csv(output_dir / "manifest_validated.csv", manifest_rows)
    _write_csv(output_dir / "case_geometry.csv", case_rows_256)
    _write_csv(output_dir / "dataset_geometry.csv", _merge_by_dataset(dataset_rows, pca_rows))
    _write_csv(output_dir / "descriptor_stability.csv", stability_rows)
    _write_csv(output_dir / "matching_contrast.csv", matching_rows)
    _write_csv(output_dir / "suitability_summary.csv", suitability_rows)

    return AuditOutputs(
        output_dir=output_dir,
        suitability_summary=output_dir / "suitability_summary.csv",
    )


def _load_records(paths: list[Path], max_cases: int | None) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    for path in paths:
        records.extend(read_manifest(path))
    if max_cases is None:
        return records
    by_dataset: dict[str, list[ManifestRecord]] = {}
    for record in records:
        by_dataset.setdefault(record.dataset, []).append(record)
    capped: list[ManifestRecord] = []
    for dataset_records in by_dataset.values():
        capped.extend(dataset_records[: int(max_cases)])
    return capped


def _audit_record(
    record: ManifestRecord,
    target_size: int,
    alternate_size: int,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    base = record.to_row()
    base.update(
        {
            "load_status": "ok",
            "valid": True,
            "validation_reason": "",
            "image_shape": "",
            "label_shape": "",
            "detected_label_values": "",
        }
    )
    try:
        image, label = load_image_and_label(record)
        base["image_shape"] = "x".join(str(x) for x in np.asarray(image.array).shape)
        base["label_shape"] = "x".join(str(x) for x in np.asarray(label.array).shape)
        base["detected_label_values"] = " ".join(str(value) for value in label_values_from_mask(label.array))
        compatible = shapes_compatible(image.array, label.array)
        if not compatible:
            base["valid"] = False
            base["validation_reason"] = "image_label_shape_mismatch"
        case_256 = _case_geometry_row(record, label.array, analysis_size=target_size)
        case_512 = _case_geometry_row(record, label.array, analysis_size=alternate_size)
        if bool(case_256.get("all_empty", False)):
            base["valid"] = False
            base["validation_reason"] = _append_reason(base["validation_reason"], "empty_mask")
        return base, case_256, case_512
    except Exception as exc:  # noqa: BLE001 - audit reports data errors instead of failing the whole run
        base["load_status"] = "error"
        base["valid"] = False
        base["validation_reason"] = f"{type(exc).__name__}: {exc}"
        return base, None, None


def _case_geometry_row(record: ManifestRecord, mask: np.ndarray, analysis_size: int) -> dict[str, Any]:
    descriptors = descriptors_for_mask_stack(mask, slice_axis=record.slice_axis, analysis_size=analysis_size)
    aggregated = aggregate_descriptors(descriptors)
    empty_count = sum(1 for descriptor in descriptors if descriptor.empty)
    row: dict[str, Any] = {
        "dataset": record.dataset,
        "case_id": record.case_id,
        "patient_id": record.patient_id,
        "analysis_size": analysis_size,
        "n_slices": len(descriptors),
        "empty_slices": empty_count,
        "all_empty": len(descriptors) == 0 or empty_count == len(descriptors),
    }
    row.update(aggregated)
    return row


def _dataset_geometry(case_rows: list[dict[str, Any]], manifest_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    datasets = sorted({str(row["dataset"]) for row in manifest_rows})
    output: list[dict[str, Any]] = []
    for dataset in datasets:
        rows = [row for row in case_rows if row.get("dataset") == dataset and not bool(row.get("all_empty", False))]
        manifest_dataset = [row for row in manifest_rows if row.get("dataset") == dataset]
        valid_manifest = [row for row in manifest_dataset if bool(row.get("valid", False))]
        if not rows:
            output.append(
                {
                    "dataset": dataset,
                    "n_cases": len(manifest_dataset),
                    "n_valid_cases": 0,
                    "n_patients": len({row.get("patient_id", "") for row in valid_manifest}),
                    "D": float("nan"),
                    "A_data": float("nan"),
                    "phi_data": float("nan"),
                    "L": float("nan"),
                    "S": float("nan"),
                    "B": float("nan"),
                    "C": float("nan"),
                    "foreground_ratio": float("nan"),
                    "empty_slice_rate": 1.0,
                }
            )
            continue
        payload: dict[str, Any] = {
            "dataset": dataset,
            "n_cases": len(manifest_dataset),
            "n_valid_cases": len(rows),
            "n_patients": len({row.get("patient_id", "") for row in valid_manifest}),
        }
        for column in ["D", "A_data", "L", "S", "B", "C", "foreground_ratio", "empty_slice_rate"]:
            payload[column] = float(np.nanmean([float(row.get(column, np.nan)) for row in rows]))
        payload["phi_data"] = axial_angle_mean([float(row.get("phi_data", np.nan)) for row in rows])
        output.append(payload)
    return output


def _stability_rows(
    dataset_rows: list[dict[str, Any]],
    dataset_rows_512: list[dict[str, Any]],
    case_rows: list[dict[str, Any]],
    config: AuditConfig,
) -> list[dict[str, Any]]:
    rows_512_by_dataset = {row["dataset"]: row for row in dataset_rows_512}
    output: list[dict[str, Any]] = []
    for row in dataset_rows:
        dataset = str(row["dataset"])
        case_subset = [case for case in case_rows if case.get("dataset") == dataset]
        payload: dict[str, Any] = {"dataset": dataset}
        payload.update(resize_stability_delta(row, rows_512_by_dataset.get(dataset, {})))
        payload.update(bootstrap_dataset_ci(case_subset, config.bootstrap_samples, config.seed))
        output.append(payload)
    return output


def _matching_rows(dataset_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_row in dataset_rows:
        dataset = str(dataset_row["dataset"])
        table = matching_table(dataset, dataset_row)
        contrast = matching_contrast(table)
        for item in table:
            item.update(contrast)
            rows.append(item)
    return rows


def _suitability_rows(
    dataset_rows: list[dict[str, Any]],
    stability_rows: list[dict[str, Any]],
    matching_rows: list[dict[str, Any]],
    pca_rows: list[dict[str, Any]],
    config: AuditConfig,
) -> list[dict[str, Any]]:
    stability_by_dataset = {str(row["dataset"]): row for row in stability_rows}
    pca_by_dataset = {str(row["dataset"]): row for row in pca_rows}
    contrast_by_dataset: dict[str, dict[str, Any]] = {}
    for dataset in {str(row["dataset"]) for row in matching_rows}:
        subset = [row for row in matching_rows if row["dataset"] == dataset]
        contrast_by_dataset[dataset] = matching_contrast(subset)
    output: list[dict[str, Any]] = []
    for row in dataset_rows:
        dataset = str(row["dataset"])
        reasons: list[str] = []
        secondary: list[str] = []
        n_valid = int(row.get("n_valid_cases", 0))
        n_patients = int(row.get("n_patients", 0))
        empty_rate = float(row.get("empty_slice_rate", 1.0))
        if n_valid <= 0:
            reasons.append("no_valid_nonempty_masks")
        if n_patients < config.min_patients:
            reasons.append(f"insufficient_patients:{n_patients}<{config.min_patients}")
        if not np.isfinite(float(row.get("D", np.nan))):
            reasons.append("descriptor_computation_failed")
        stability = stability_by_dataset.get(dataset, {})
        max_delta = float(stability.get("max_resize_delta", np.nan))
        if np.isfinite(max_delta) and max_delta > 0.20:
            secondary.append(f"descriptor_resize_delta_high:{max_delta:.3f}")
        if empty_rate > 0.75:
            secondary.append(f"many_empty_slices:{empty_rate:.3f}")
        contrast = contrast_by_dataset.get(dataset, {})
        matching_value = float(contrast.get("matching_contrast", np.nan))
        if np.isfinite(matching_value) and matching_value < 0.08:
            secondary.append(f"low_matching_contrast:{matching_value:.3f}")
        pca = pca_by_dataset.get(dataset, {})
        nearest = float(pca.get("nearest_neighbor_distance", np.nan))
        if np.isfinite(nearest) and nearest < 0.25:
            secondary.append(f"low_geometry_spread:{nearest:.3f}")
        if reasons:
            status = "reject"
        elif secondary:
            status = "secondary_only"
        else:
            status = "accept"
        output.append(
            {
                "dataset": dataset,
                "status": status,
                "reasons": ";".join(reasons + secondary),
                "n_cases": row.get("n_cases", 0),
                "n_valid_cases": n_valid,
                "n_patients": n_patients,
                "empty_slice_rate": empty_rate,
                "matching_contrast": contrast.get("matching_contrast", ""),
                "best_scan": contrast.get("best_scan", ""),
                "nearest_neighbor_distance": pca.get("nearest_neighbor_distance", ""),
                "max_resize_delta": stability.get("max_resize_delta", ""),
            }
        )
    return output


def _merge_by_dataset(primary: list[dict[str, Any]], extra: list[dict[str, Any]]) -> list[dict[str, Any]]:
    extra_by_dataset = {row["dataset"]: row for row in extra}
    merged: list[dict[str, Any]] = []
    for row in primary:
        payload = dict(row)
        payload.update({key: value for key, value in extra_by_dataset.get(row["dataset"], {}).items() if key != "dataset"})
        merged.append(payload)
    return merged


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _append_reason(existing: str, reason: str) -> str:
    return reason if not existing else f"{existing};{reason}"
