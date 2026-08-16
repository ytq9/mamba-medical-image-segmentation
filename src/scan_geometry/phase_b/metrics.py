from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"dataset", "condition", "seed", "case_id", "dice", "hd95"}


def metric_files_from_runs(runs_csv: str | Path) -> list[Path]:
    runs_path = Path(runs_csv)
    files: list[Path] = []
    with runs_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            output_dir = row.get("output_dir")
            if not output_dir:
                continue
            for candidate in [Path(output_dir) / "metrics.csv", Path(output_dir) / "test_metrics.csv"]:
                if candidate.exists():
                    files.append(candidate)
                    break
    return files


def read_case_metrics(paths: list[str | Path]) -> pd.DataFrame:
    if not paths:
        raise ValueError("No metric files were provided.")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing required metric columns: {sorted(missing)}")
        if "class_id" not in frame.columns:
            frame["class_id"] = "macro"
        frame["metric_source"] = str(path)
        frames.append(frame)
    metrics = pd.concat(frames, ignore_index=True)
    metrics["seed"] = metrics["seed"].astype(int)
    metrics["dice"] = metrics["dice"].astype(float)
    metrics["hd95"] = metrics["hd95"].astype(float)
    return metrics


def aggregate_case_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    required = REQUIRED_COLUMNS | {"class_id"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Metrics frame is missing required columns: {sorted(missing)}")

    group_cols = ["dataset", "condition", "class_id"]
    seed_cols = [*group_cols, "seed"]
    seed_means = (
        metrics.groupby(seed_cols, dropna=False)
        .agg(
            dice_seed_mean=("dice", "mean"),
            hd95_seed_mean=("hd95", "mean"),
        )
        .reset_index()
    )
    seed_summary = (
        seed_means.groupby(group_cols, dropna=False)
        .agg(
            dice_mean=("dice_seed_mean", "mean"),
            dice_seed_std=("dice_seed_mean", "std"),
            hd95_mean=("hd95_seed_mean", "mean"),
            hd95_seed_std=("hd95_seed_mean", "std"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    case_summary = (
        metrics.groupby(group_cols, dropna=False)
        .agg(
            dice_case_p05=("dice", lambda values: float(values.quantile(0.05))),
            failure_rate=("dice", lambda values: float((values < 0.5).mean())),
            n_observations=("dice", "size"),
            n_cases=("case_id", "nunique"),
        )
        .reset_index()
    )
    summary = seed_summary.merge(case_summary, on=group_cols, how="left")
    for column in ["dice_seed_std", "hd95_seed_std"]:
        summary[column] = summary[column].fillna(0.0)
    return summary.sort_values(group_cols).reset_index(drop=True)


def write_metrics_summary(path: str | Path, summary: pd.DataFrame) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    return output
