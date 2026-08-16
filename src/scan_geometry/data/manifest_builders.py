from __future__ import annotations

import json
import re
from pathlib import Path

from .manifest import ManifestRecord, notes_with_payload, write_manifest

PRESETS = {"camus", "mnms", "isic2018", "amos22", "drive"}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".gif", ".bmp"}
VOLUME_EXTS = {".nii", ".nii.gz", ".mhd", ".mha", ".npy", ".npz"}


def build_manifest(
    preset: str,
    root: str | Path,
    output: str | Path,
    dataset_name: str | None = None,
    modality: str | None = None,
    task_type: str | None = None,
    split: str = "",
) -> list[ManifestRecord]:
    root_path = Path(root).resolve()
    preset_key = preset.strip().lower()
    if preset_key not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}. Expected one of {sorted(PRESETS)}.")
    builders = {
        "camus": _build_camus,
        "mnms": _build_mnms,
        "isic2018": _build_isic2018,
        "amos22": _build_amos22,
        "drive": _build_drive,
    }
    records = builders[preset_key](root_path, dataset_name, modality, task_type, split)
    if not records:
        raise ValueError(f"No image/mask pairs found for preset={preset_key} under {root_path}")
    write_manifest(output, records)
    return records


def _build_camus(
    root: Path,
    dataset_name: str | None,
    modality: str | None,
    task_type: str | None,
    split: str,
) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    for image_path in _candidate_camus_image_paths(root):
        stem = _strip_known_suffix(image_path.name)
        if _looks_like_mask(stem):
            continue
        if "sequence" in _norm(stem):
            continue
        label_path = _first_existing(
            *[
                image_path.with_name(f"{stem}_{token}{_data_suffix(image_path)}")
                for token in ["gt", "mask", "label", "seg"]
            ]
        )
        if label_path is None:
            continue
        patient = _infer_patient_id(image_path, default=image_path.parent.name)
        view = _infer_token(stem, {"A2C", "A4C", "2CH", "4CH"})
        phase = _infer_token(stem, {"ED", "ES"})
        records.append(
            ManifestRecord(
                dataset=dataset_name or "CAMUS",
                case_id=stem,
                patient_id=patient,
                image_path=str(image_path),
                label_path=str(label_path),
                modality=modality or "ultrasound",
                task_type=task_type or "cardiac_chambers",
                split=split,
                is_2d=True,
                view=_normalize_view(view),
                phase=phase,
            )
        )
    return records


def _build_mnms(
    root: Path,
    dataset_name: str | None,
    modality: str | None,
    task_type: str | None,
    split: str,
) -> list[ManifestRecord]:
    label_paths = _candidate_label_paths(root, volume_only=True)
    label_by_key = {_case_key(path): path for path in label_paths}
    records: list[ManifestRecord] = []
    for image_path in _candidate_image_paths(root, volume_only=True):
        key = _case_key(image_path)
        label_path = label_by_key.get(key)
        if label_path is None:
            continue
        patient = _infer_patient_id(image_path, default=key)
        phase = _infer_token(image_path.stem.upper(), {"ED", "ES"})
        records.append(
            ManifestRecord(
                dataset=dataset_name or "M&Ms",
                case_id=key,
                patient_id=patient,
                image_path=str(image_path),
                label_path=str(label_path),
                modality=modality or "MRI",
                task_type=task_type or "cardiac_chambers",
                split=split,
                is_2d=False,
                slice_axis=-1,
                phase=phase,
            )
        )
    return records


def _build_isic2018(
    root: Path,
    dataset_name: str | None,
    modality: str | None,
    task_type: str | None,
    split: str,
) -> list[ManifestRecord]:
    mask_paths = [
        path
        for path in _iter_files(root, IMAGE_EXTS)
        if _looks_like_mask(path.stem) or "groundtruth" in _norm(path.parent.name) or "mask" in _norm(path.parent.name)
    ]
    mask_by_key = {_isic_key(path): path for path in mask_paths}
    records: list[ManifestRecord] = []
    for image_path in _iter_files(root, IMAGE_EXTS):
        if image_path in mask_paths or _looks_like_mask(image_path.stem):
            continue
        key = _isic_key(image_path)
        label_path = mask_by_key.get(key)
        if label_path is None:
            continue
        records.append(
            ManifestRecord(
                dataset=dataset_name or "ISIC2018",
                case_id=key,
                patient_id=key,
                image_path=str(image_path),
                label_path=str(label_path),
                modality=modality or "dermoscopy",
                task_type=task_type or "skin_lesion",
                split=_infer_split(image_path) or split,
                is_2d=True,
                label_values="1",
            )
        )
    return records


def _build_amos22(
    root: Path,
    dataset_name: str | None,
    modality: str | None,
    task_type: str | None,
    split: str,
) -> list[ManifestRecord]:
    root = _resolve_dataset_root(
        root,
        image_dirs=("imagesTr", "images", "image"),
        label_dirs=("labelsTr", "labels", "label"),
    )
    dataset_json = _read_dataset_json(root / "dataset.json")
    default_modality = modality or _infer_amos_modality(dataset_json) or "CT/MRI"
    labels_dir = _first_existing_dir(root / "labelsTr", root / "labels", root / "label")
    images_dir = _first_existing_dir(root / "imagesTr", root / "images", root / "image")
    if labels_dir is None or images_dir is None:
        return []
    labels = {_strip_nii_suffix(path.name).replace("_gt", ""): path for path in sorted(labels_dir.glob("*.nii*"))}
    records: list[ManifestRecord] = []
    for image_path in sorted(images_dir.glob("*.nii*")):
        key = _strip_nii_suffix(image_path.name).replace("_0000", "")
        label_path = labels.get(key)
        if label_path is None:
            continue
        records.append(
            ManifestRecord(
                dataset=dataset_name or "AMOS2022",
                case_id=key,
                patient_id=key,
                image_path=str(image_path),
                label_path=str(label_path),
                modality=default_modality,
                task_type=task_type or "multi_organ_abdomen",
                split=split or "train",
                is_2d=False,
                slice_axis=-1,
                notes=notes_with_payload(source="amos22_preset"),
            )
        )
    return records


def _build_drive(
    root: Path,
    dataset_name: str | None,
    modality: str | None,
    task_type: str | None,
    split: str,
) -> list[ManifestRecord]:
    masks = [
        path
        for path in _iter_files(root, IMAGE_EXTS)
        if _looks_like_vessel_mask(path) or "manual" in _norm(str(path.parent)) or "vessel" in _norm(str(path.parent))
    ]
    mask_by_key = {_leading_number(path) or path.stem: path for path in masks}
    fov_by_key = {
        _leading_number(path) or path.stem: path
        for path in _iter_files(root, IMAGE_EXTS)
        if "mask" in _norm(path.stem) and not _looks_like_vessel_mask(path)
    }
    records: list[ManifestRecord] = []
    for image_path in _iter_files(root, IMAGE_EXTS):
        if image_path in masks or "mask" in _norm(image_path.stem) or "manual" in _norm(str(image_path.parent)):
            continue
        key = _leading_number(image_path) or image_path.stem
        label_path = mask_by_key.get(key)
        if label_path is None:
            continue
        records.append(
            ManifestRecord(
                dataset=dataset_name or "DRIVE",
                case_id=key,
                patient_id=key,
                image_path=str(image_path),
                label_path=str(label_path),
                modality=modality or "fundus",
                task_type=task_type or "retinal_vessels",
                split=_infer_split(image_path) or split,
                is_2d=True,
                label_values="1",
                notes=notes_with_payload(fov_mask=str(fov_by_key.get(key, ""))),
            )
        )
    return records


def _iter_files(root: Path, extensions: set[str]) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in extensions)


def _candidate_image_paths(root: Path, volume_only: bool) -> list[Path]:
    exts = VOLUME_EXTS if volume_only else VOLUME_EXTS | IMAGE_EXTS
    return [
        path
        for path in _iter_files_multi_suffix(root, exts)
        if not _looks_like_mask(path.stem) and not _path_contains_label_dir(path)
    ]


def _candidate_label_paths(root: Path, volume_only: bool) -> list[Path]:
    exts = VOLUME_EXTS if volume_only else VOLUME_EXTS | IMAGE_EXTS
    return [
        path
        for path in _iter_files_multi_suffix(root, exts)
        if _looks_like_mask(path.stem) or _path_contains_label_dir(path)
    ]


def _iter_files_multi_suffix(root: Path, extensions: set[str]) -> list[Path]:
    output = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suffix = ".nii.gz" if path.name.lower().endswith(".nii.gz") else path.suffix.lower()
        if suffix in extensions:
            output.append(path)
    return sorted(output)


def _candidate_camus_image_paths(root: Path) -> list[Path]:
    exts = {".mhd", ".mha", ".nii", ".nii.gz"}
    return [
        path
        for path in _iter_files_multi_suffix(root, exts)
        if not _looks_like_mask(_strip_known_suffix(path.name))
    ]


def _case_key(path: Path) -> str:
    key = _strip_nii_suffix(path.name)
    for token in ["_segmentation", "_seg", "_label", "_labels", "_mask", "_gt", "_manual"]:
        key = re.sub(f"{token}$", "", key, flags=re.IGNORECASE)
    return key


def _isic_key(path: Path) -> str:
    key = path.stem
    return re.sub(r"_segmentation$", "", key, flags=re.IGNORECASE)


def _looks_like_mask(stem: str) -> bool:
    normalized = _norm(stem)
    return any(token in normalized for token in ["gt", "mask", "seg", "label", "manual"])


def _looks_like_vessel_mask(path: Path) -> bool:
    normalized = _norm(path.stem + "_" + path.parent.name)
    return any(token in normalized for token in ["manual", "vessel", "label", "seg"])


def _path_contains_label_dir(path: Path) -> bool:
    normalized = [_norm(part) for part in path.parts]
    return any(part in {"label", "labels", "labelstr", "groundtruth", "gt", "masks", "mask"} for part in normalized)


def _infer_patient_id(path: Path, default: str) -> str:
    match = re.search(r"(patient[_-]?\d+|sub[_-]?\d+|case[_-]?\d+|\d{3,})", str(path), flags=re.IGNORECASE)
    return match.group(1) if match else default


def _infer_token(text: str, allowed: set[str]) -> str:
    upper = text.upper()
    for token in allowed:
        if token in upper:
            return token
    return ""


def _normalize_view(view: str) -> str:
    return {"2CH": "A2C", "4CH": "A4C"}.get(view, view)


def _infer_split(path: Path) -> str:
    normalized = _norm(str(path.parent))
    if "train" in normalized or "training" in normalized:
        return "train"
    if "test" in normalized:
        return "test"
    if "val" in normalized or "validation" in normalized:
        return "val"
    return ""


def _leading_number(path: Path) -> str:
    match = re.match(r"(\d+)", path.stem)
    return match.group(1) if match else ""


def _strip_nii_suffix(name: str) -> str:
    lowered = name.lower()
    if lowered.endswith(".nii.gz"):
        return name[:-7]
    return Path(name).stem


def _strip_known_suffix(name: str) -> str:
    lowered = name.lower()
    if lowered.endswith(".nii.gz"):
        return name[:-7]
    return Path(name).stem


def _data_suffix(path: Path) -> str:
    return ".nii.gz" if path.name.lower().endswith(".nii.gz") else path.suffix


def _first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _first_existing_dir(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists() and path.is_dir():
            return path
    return None


def _resolve_dataset_root(root: Path, *, image_dirs: tuple[str, ...], label_dirs: tuple[str, ...]) -> Path:
    if _has_any_dir(root, image_dirs) and _has_any_dir(root, label_dirs):
        return root
    for child in sorted(root.iterdir()) if root.exists() else []:
        if not child.is_dir() or child.name.startswith("__"):
            continue
        if _has_any_dir(child, image_dirs) and _has_any_dir(child, label_dirs):
            return child
    return root


def _has_any_dir(root: Path, names: tuple[str, ...]) -> bool:
    return any((root / name).is_dir() for name in names)


def _read_dataset_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return {}


def _infer_amos_modality(payload: dict) -> str:
    modalities = payload.get("modality") or payload.get("channel_names") or {}
    if isinstance(modalities, dict):
        text = " ".join(str(value) for value in modalities.values()).upper()
        if "CT" in text and "MR" in text:
            return "CT/MRI"
        if "CT" in text:
            return "CT"
        if "MR" in text:
            return "MRI"
    return ""


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())
