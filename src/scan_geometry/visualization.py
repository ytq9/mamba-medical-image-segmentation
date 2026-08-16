from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .geometry.coverage import PCA_COLUMNS
from .geometry.scan_profiles import DEFAULT_SCAN_PROFILES


CONDITION_ORDER = ["CNN", "Raster-H", "Raster-V", "Hilbert", "LocalWindow", "RandomPermute"]
MAMBA_CONDITIONS = ["Raster-H", "Raster-V", "Hilbert", "LocalWindow", "RandomPermute"]


@dataclass(slots=True)
class VisualizationConfig:
    output_dir: Path = Path("results/visualizations")
    dataset_geometry: Path | None = Path("results/dataset_audit/dataset_geometry.csv")
    matching_contrast: Path | None = Path("results/dataset_audit/matching_contrast.csv")
    metrics_summary: Path | None = Path("results/phase_b/metrics_summary.csv")
    case_metrics: Path | None = None
    prefer_class_id: str = "macro"


def generate_visualizations(config: VisualizationConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"output_dir": str(config.output_dir), "figures": [], "skipped": []}

    dataset_geometry = _read_optional_csv(config.dataset_geometry)
    matching_contrast = _read_optional_csv(config.matching_contrast)
    metrics_summary = _read_optional_csv(config.metrics_summary)
    case_metrics = _read_optional_csv(config.case_metrics)

    _run_plot(manifest, config.output_dir / "scan_profile_geometry.png", plot_scan_profile_geometry)
    _run_plot(
        manifest,
        config.output_dir / "descriptor_correlation.png",
        plot_descriptor_correlation,
        dataset_geometry,
        required=not dataset_geometry.empty,
        skip_reason="dataset_geometry.csv not found or empty",
    )
    _run_plot(
        manifest,
        config.output_dir / "performance_heatmap_dice.png",
        plot_performance_heatmap,
        metrics_summary,
        config.prefer_class_id,
        required=not metrics_summary.empty,
        skip_reason="metrics_summary.csv not found or empty",
    )
    _run_plot(
        manifest,
        config.output_dir / "scan_ranking_by_dataset.png",
        plot_scan_ranking_by_dataset,
        metrics_summary,
        config.prefer_class_id,
        required=not metrics_summary.empty,
        skip_reason="metrics_summary.csv not found or empty",
    )
    _run_plot(
        manifest,
        config.output_dir / "seed_stability.png",
        plot_seed_stability,
        metrics_summary,
        config.prefer_class_id,
        required=not metrics_summary.empty,
        skip_reason="metrics_summary.csv not found or empty",
    )
    _run_plot(
        manifest,
        config.output_dir / "p5_matching_association.png",
        plot_p5_matching_association,
        case_metrics,
        matching_contrast,
        required=not case_metrics.empty and not matching_contrast.empty,
        skip_reason="case metrics and matching_contrast.csv are required",
    )

    manifest_path = config.output_dir / "visualization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def plot_scan_profile_geometry(path: Path) -> None:
    rows = DEFAULT_SCAN_PROFILES
    plt.figure(figsize=(6.5, 4.5))
    x = [profile.A_scan for profile in rows]
    y = [profile.P_loc for profile in rows]
    plt.scatter(x, y, s=70)
    for profile in rows:
        label = profile.name
        if profile.phi_scan is not None:
            label += f"\nphi={profile.phi_scan:.2f}"
        plt.annotate(label, (profile.A_scan, profile.P_loc), xytext=(5, 5), textcoords="offset points", fontsize=8)
    plt.xlim(-0.05, 1.05)
    plt.ylim(0.0, 1.0)
    plt.xlabel("A_scan (directionality)")
    plt.ylabel("P_loc (locality preservation)")
    plt.title("Path-derived Scan Profile Geometry")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_descriptor_correlation(path: Path, dataset_geometry: pd.DataFrame) -> None:
    frame = _numeric_columns(dataset_geometry, PCA_COLUMNS)
    corr = frame.corr(method="spearman")
    plt.figure(figsize=(5.5, 4.8))
    im = plt.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1.0, vmax=1.0)
    plt.xticks(range(len(PCA_COLUMNS)), PCA_COLUMNS)
    plt.yticks(range(len(PCA_COLUMNS)), PCA_COLUMNS)
    for row in range(len(PCA_COLUMNS)):
        for col in range(len(PCA_COLUMNS)):
            value = corr.iloc[row, col]
            label = "" if not np.isfinite(value) else f"{value:.2f}"
            plt.text(col, row, label, ha="center", va="center", fontsize=8)
    plt.colorbar(im, label="Spearman rho")
    plt.title("Descriptor Independence Check")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_performance_heatmap(path: Path, metrics_summary: pd.DataFrame, class_id: str = "macro") -> None:
    frame = _filter_class(metrics_summary, class_id)
    pivot = frame.pivot_table(index="dataset", columns="condition", values="dice_mean", aggfunc="mean")
    datasets = list(pivot.index)
    conditions = [condition for condition in CONDITION_ORDER if condition in pivot.columns]
    values = pivot.reindex(index=datasets, columns=conditions)

    plt.figure(figsize=(max(7, len(conditions) * 1.15), max(3.5, len(datasets) * 0.55 + 1.5)))
    matrix = values.to_numpy(dtype=float)
    im = plt.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    plt.xticks(range(len(conditions)), conditions, rotation=25, ha="right")
    plt.yticks(range(len(datasets)), datasets)
    std_lookup = frame.set_index(["dataset", "condition"]).get("dice_seed_std")
    for i, dataset in enumerate(datasets):
        for j, condition in enumerate(conditions):
            value = values.loc[dataset, condition]
            if pd.isna(value):
                label = ""
            else:
                std = ""
                if std_lookup is not None and (dataset, condition) in std_lookup.index:
                    std_value = std_lookup.loc[(dataset, condition)]
                    if np.isfinite(float(std_value)):
                        std = f"\n+/-{float(std_value):.3f}"
                label = f"{float(value):.3f}{std}"
            plt.text(j, i, label, ha="center", va="center", fontsize=8, color="white")
    plt.colorbar(im, label="Mean Dice")
    plt.title("Phase B Performance Heatmap")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_scan_ranking_by_dataset(path: Path, metrics_summary: pd.DataFrame, class_id: str = "macro") -> None:
    frame = _filter_class(metrics_summary, class_id)
    frame = frame[frame["condition"].isin(MAMBA_CONDITIONS)].copy()
    plt.figure(figsize=(8, 4.5))
    if not frame.empty:
        for dataset, group in frame.groupby("dataset"):
            ordered = group.set_index("condition").reindex(MAMBA_CONDITIONS)
            plt.plot(MAMBA_CONDITIONS, ordered["dice_mean"], marker="o", label=str(dataset))
        plt.legend(fontsize=8)
    plt.ylim(0.0, 1.0)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Mean Dice")
    plt.title("Mamba Scan Ranking by Dataset")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_seed_stability(path: Path, metrics_summary: pd.DataFrame, class_id: str = "macro") -> None:
    frame = _filter_class(metrics_summary, class_id)
    frame = frame.copy()
    frame["label"] = frame["dataset"].astype(str) + "\n" + frame["condition"].astype(str)
    frame = frame.sort_values(["dataset", "condition"])
    plt.figure(figsize=(max(9, len(frame) * 0.35), 4.8))
    x = np.arange(len(frame))
    y = frame["dice_mean"].astype(float).to_numpy()
    err = frame.get("dice_seed_std", pd.Series(np.zeros(len(frame)))).astype(float).fillna(0.0).to_numpy()
    colors = ["#4C78A8" if condition == "CNN" else "#54A24B" for condition in frame["condition"]]
    plt.bar(x, y, yerr=err, capsize=2, color=colors)
    if "failure_rate" in frame.columns:
        for idx, failure in enumerate(frame["failure_rate"].astype(float).fillna(0.0)):
            if failure > 0:
                plt.text(idx, min(1.0, y[idx] + err[idx] + 0.025), f"fail {failure:.2f}", ha="center", fontsize=7, rotation=90)
    plt.ylim(0.0, 1.05)
    plt.xticks(x, frame["label"], rotation=90, fontsize=7)
    plt.ylabel("Mean Dice +/- seed std")
    plt.title("Seed Stability and Failure Rate")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_p5_matching_association(path: Path, case_metrics: pd.DataFrame, matching_contrast: pd.DataFrame) -> None:
    metrics = case_metrics[case_metrics["condition"].isin(MAMBA_CONDITIONS)].copy()
    metrics["dice"] = metrics["dice"].astype(float)
    metrics["within_dataset_z_dice"] = metrics.groupby("dataset")["dice"].transform(_zscore)
    matches = matching_contrast.rename(columns={"scan": "condition"})[["dataset", "condition", "M_primary"]].copy()
    merged = metrics.merge(matches, on=["dataset", "condition"], how="inner")
    plt.figure(figsize=(6.5, 4.8))
    if not merged.empty:
        for dataset, group in merged.groupby("dataset"):
            plt.scatter(group["M_primary"].astype(float), group["within_dataset_z_dice"].astype(float), label=str(dataset), alpha=0.75)
        x = merged["M_primary"].astype(float).to_numpy()
        y = merged["within_dataset_z_dice"].astype(float).to_numpy()
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() >= 2:
            slope, intercept = np.polyfit(x[finite], y[finite], deg=1)
            xs = np.linspace(float(np.nanmin(x[finite])), float(np.nanmax(x[finite])), 100)
            plt.plot(xs, slope * xs + intercept, color="black", linewidth=1.5, label="linear fit")
        plt.legend(fontsize=8)
    plt.xlabel("Pre-registered M_primary(d, s)")
    plt.ylabel("Within-dataset standardized Dice")
    plt.title("P5 Matching Association")
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _run_plot(
    manifest: dict[str, Any],
    path: Path,
    func: Any,
    *args: Any,
    required: bool = True,
    skip_reason: str = "",
) -> None:
    if not required:
        manifest["skipped"].append({"figure": str(path), "reason": skip_reason})
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    func(path, *args)
    manifest["figures"].append(str(path))


def _read_optional_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _numeric_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=columns)
    output = pd.DataFrame()
    for column in columns:
        output[column] = pd.to_numeric(frame.get(column, np.nan), errors="coerce")
    return output


def _filter_class(frame: pd.DataFrame, class_id: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    if "class_id" not in frame.columns:
        return frame
    class_values = frame["class_id"].astype(str)
    if class_id in set(class_values):
        return frame[class_values == class_id].copy()
    return frame.copy()


def _zscore(values: pd.Series) -> pd.Series:
    mean = values.mean()
    std = values.std(ddof=0)
    if not np.isfinite(std) or std < 1e-12:
        return values * 0.0
    return (values - mean) / std
