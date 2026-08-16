from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scan_geometry.visualization import VisualizationConfig, generate_visualizations


def test_visualizations_gracefully_skip_missing_training_data(tmp_path: Path) -> None:
    manifest = generate_visualizations(
        VisualizationConfig(
            output_dir=tmp_path / "figures",
            dataset_geometry=None,
            matching_contrast=None,
            metrics_summary=None,
            case_metrics=None,
        )
    )

    assert (tmp_path / "figures" / "scan_profile_geometry.png").exists()
    assert len(manifest["figures"]) == 1
    assert len(manifest["skipped"]) == 5
    saved = json.loads((tmp_path / "figures" / "visualization_manifest.json").read_text(encoding="utf-8"))
    assert saved["skipped"]


def test_visualizations_generate_all_figures_from_synthetic_csvs(tmp_path: Path) -> None:
    dataset_geometry = tmp_path / "dataset_geometry.csv"
    matching_contrast = tmp_path / "matching_contrast.csv"
    metrics_summary = tmp_path / "metrics_summary.csv"
    case_metrics = tmp_path / "case_metrics.csv"

    pd.DataFrame(
        [
            {"dataset": "CAMUS", "D": 0.78, "L": 0.98, "S": 0.38, "B": 0.46, "C": 1.0},
            {"dataset": "DRIVE", "D": 0.55, "L": 0.40, "S": 0.90, "B": 0.70, "C": 8.0},
        ]
    ).to_csv(dataset_geometry, index=False)
    pd.DataFrame(
        [
            {"dataset": dataset, "scan": condition, "M_primary": 0.50 + 0.05 * index}
            for dataset in ["CAMUS", "DRIVE"]
            for index, condition in enumerate(["Raster-H", "Raster-V", "Hilbert", "LocalWindow", "RandomPermute"])
        ]
    ).to_csv(matching_contrast, index=False)
    pd.DataFrame(
        [
            {
                "dataset": dataset,
                "condition": condition,
                "class_id": "macro",
                "dice_mean": 0.70 + 0.01 * condition_index,
                "dice_seed_std": 0.02,
                "failure_rate": 0.0,
            }
            for dataset in ["CAMUS", "DRIVE"]
            for condition_index, condition in enumerate(["CNN", "Raster-H", "Raster-V", "Hilbert", "LocalWindow", "RandomPermute"])
        ]
    ).to_csv(metrics_summary, index=False)
    pd.DataFrame(
        [
            {
                "dataset": dataset,
                "condition": condition,
                "seed": seed,
                "case_id": f"{dataset}_{seed}_{case}",
                "class_id": "macro",
                "dice": 0.65 + 0.01 * seed + 0.02 * condition_index,
                "hd95": 5.0,
            }
            for dataset in ["CAMUS", "DRIVE"]
            for condition_index, condition in enumerate(["Raster-H", "Raster-V", "Hilbert", "LocalWindow", "RandomPermute"])
            for seed in [1, 2]
            for case in [1, 2]
        ]
    ).to_csv(case_metrics, index=False)

    manifest = generate_visualizations(
        VisualizationConfig(
            output_dir=tmp_path / "figures",
            dataset_geometry=dataset_geometry,
            matching_contrast=matching_contrast,
            metrics_summary=metrics_summary,
            case_metrics=case_metrics,
        )
    )

    expected = {
        "scan_profile_geometry.png",
        "descriptor_correlation.png",
        "performance_heatmap_dice.png",
        "scan_ranking_by_dataset.png",
        "seed_stability.png",
        "p5_matching_association.png",
    }
    assert expected == {Path(path).name for path in manifest["figures"]}
    assert manifest["skipped"] == []
    for name in expected:
        assert (tmp_path / "figures" / name).exists()
