from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from scan_geometry.data.manifest_builders import build_manifest


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("placeholder", encoding="utf-8")


def _image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(path)


def test_build_isic_manifest_pairs_segmentation_masks(tmp_path: Path) -> None:
    root = tmp_path / "isic"
    _image(root / "images" / "ISIC_0001.jpg")
    _image(root / "masks" / "ISIC_0001_segmentation.png")
    rows = build_manifest("isic2018", root, tmp_path / "isic.csv")
    assert len(rows) == 1
    assert rows[0].case_id == "ISIC_0001"


def test_build_camus_manifest_pairs_gt_mhd(tmp_path: Path) -> None:
    root = tmp_path / "camus"
    _touch(root / "patient0001" / "patient0001_2CH_ED.mhd")
    _touch(root / "patient0001" / "patient0001_2CH_ED_gt.mhd")
    rows = build_manifest("camus", root, tmp_path / "camus.csv")
    assert len(rows) == 1
    assert rows[0].view == "A2C"
    assert rows[0].phase == "ED"


def test_build_camus_manifest_pairs_gt_nifti_and_skips_sequences(tmp_path: Path) -> None:
    root = tmp_path / "camus"
    _touch(root / "CAMUS_public" / "database_nifti" / "patient0001" / "patient0001_2CH_ED.nii.gz")
    _touch(root / "CAMUS_public" / "database_nifti" / "patient0001" / "patient0001_2CH_ED_gt.nii.gz")
    _touch(root / "CAMUS_public" / "database_nifti" / "patient0001" / "patient0001_2CH_half_sequence.nii.gz")
    _touch(root / "CAMUS_public" / "database_nifti" / "patient0001" / "patient0001_2CH_half_sequence_gt.nii.gz")
    rows = build_manifest("camus", root, tmp_path / "camus.csv")
    assert len(rows) == 1
    assert rows[0].case_id == "patient0001_2CH_ED"
    assert rows[0].label_path.endswith("patient0001_2CH_ED_gt.nii.gz")


def test_build_amos_manifest_pairs_imagestr_labelstr(tmp_path: Path) -> None:
    root = tmp_path / "amos"
    _touch(root / "imagesTr" / "amos_0001_0000.nii.gz")
    _touch(root / "labelsTr" / "amos_0001.nii.gz")
    rows = build_manifest("amos22", root, tmp_path / "amos.csv")
    assert len(rows) == 1
    assert rows[0].case_id == "amos_0001"


def test_build_amos_manifest_accepts_nested_amos22_root(tmp_path: Path) -> None:
    root = tmp_path / "AMOS2022"
    _touch(root / "amos22" / "imagesTr" / "amos_0001.nii.gz")
    _touch(root / "amos22" / "labelsTr" / "amos_0001.nii.gz")
    rows = build_manifest("amos22", root, tmp_path / "amos.csv")
    assert len(rows) == 1
    assert rows[0].image_path.endswith("AMOS2022/amos22/imagesTr/amos_0001.nii.gz")


def test_build_drive_manifest_pairs_numeric_prefix(tmp_path: Path) -> None:
    root = tmp_path / "drive"
    _image(root / "training" / "images" / "21_training.tif")
    _image(root / "training" / "1st_manual" / "21_manual1.gif")
    rows = build_manifest("drive", root, tmp_path / "drive.csv")
    assert len(rows) == 1
    assert rows[0].case_id == "21"


def test_build_mnms_manifest_pairs_label_dir(tmp_path: Path) -> None:
    root = tmp_path / "mnms"
    _touch(root / "images" / "patient001_ED.nii.gz")
    _touch(root / "labels" / "patient001_ED.nii.gz")
    rows = build_manifest("mnms", root, tmp_path / "mnms.csv")
    assert len(rows) == 1
    assert rows[0].phase == "ED"
