#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONDITION_ORDER = ["CNN", "Raster-H", "Raster-V", "Hilbert", "LocalWindow", "RandomPermute"]
MAMBA_CONDITIONS = ["Raster-H", "Raster-V", "Hilbert", "LocalWindow", "RandomPermute"]
DATASET_ORDER = ["CAMUS", "M&Ms", "ISIC2018", "AMOS2022", "DRIVE"]

RUN_RE = re.compile(r"(.+)__(cnn|raster_h|raster_v|hilbert|localwindow|randompermute)__seed(\d+)$")
DATASET_NAME_MAP = {
    "camus": "CAMUS",
    "mnms": "M&Ms",
    "m_ms": "M&Ms",
    "m&ms": "M&Ms",
    "isic2018": "ISIC2018",
    "amos2022": "AMOS2022",
    "amos22": "AMOS2022",
    "drive": "DRIVE",
}
CONDITION_NAME_MAP = {
    "cnn": "CNN",
    "raster_h": "Raster-H",
    "raster_v": "Raster-V",
    "hilbert": "Hilbert",
    "localwindow": "LocalWindow",
    "randompermute": "RandomPermute",
}

P5_LABEL_OFFSETS: dict[tuple[str, str], tuple[int, int, str, str]] = {
    ("CAMUS", "Hilbert"): (-8, 6, "right", "bottom"),
    ("CAMUS", "RandomPermute"): (4, 8, "left", "bottom"),
    ("M&Ms", "Hilbert"): (-8, 6, "right", "bottom"),
    ("M&Ms", "RandomPermute"): (4, 8, "left", "bottom"),
    ("ISIC2018", "Hilbert"): (-8, 4, "right", "bottom"),
    ("ISIC2018", "Raster-H"): (6, -12, "left", "top"),
    ("ISIC2018", "Raster-V"): (6, 6, "left", "bottom"),
    ("AMOS2022", "Raster-H"): (6, -12, "left", "top"),
    ("AMOS2022", "Hilbert"): (-8, 4, "right", "bottom"),
    ("AMOS2022", "LocalWindow"): (6, 8, "left", "bottom"),
    ("DRIVE", "RandomPermute"): (5, 8, "left", "bottom"),
    ("DRIVE", "Raster-H"): (0, 13, "center", "bottom"),
    ("DRIVE", "Raster-V"): (8, -2, "left", "center"),
    ("DRIVE", "LocalWindow"): (0, -14, "center", "top"),
    ("DRIVE", "Hilbert"): (6, 5, "left", "bottom"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate paper-ready Phase B result figures and H1 matching-score analysis. "
            "Run this on the machine containing results/phase_b/runs and, for H1, "
            "results/dataset_audit/matching_contrast.csv."
        )
    )
    parser.add_argument("--runs-dir", default="results/phase_b/runs", help="Directory containing */metrics.csv files.")
    parser.add_argument("--matching-contrast", default="results/dataset_audit/matching_contrast.csv")
    parser.add_argument("--output-dir", default="results/visualizations/paper_analysis")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--permutation-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_level, case_metrics = collect_run_metrics(Path(args.runs_dir))
    if run_level.empty:
        raise SystemExit(f"No metrics.csv files found under {args.runs_dir}")

    fair_summary = make_fair_summary(run_level)
    hypothesis = make_hypothesis_summary(fair_summary)
    paired = paired_vs_cnn(case_metrics, bootstrap_samples=args.bootstrap_samples, seed=args.seed)

    run_level.to_csv(output_dir / "paper_run_level.csv", index=False)
    fair_summary.to_csv(output_dir / "paper_main_result_table.csv", index=False)
    hypothesis.to_csv(output_dir / "paper_ablation_control_table.csv", index=False)
    paired.to_csv(output_dir / "paper_paired_vs_cnn_bootstrap.csv", index=False)

    figures: list[str] = []
    figures.append(str(plot_dice_heatmap(fair_summary, output_dir / "main_result_dice_heatmap.png")))
    figures.append(str(plot_hd95_heatmap(fair_summary, output_dir / "main_result_hd95_heatmap.png")))
    figures.append(str(plot_best_mamba_vs_cnn(hypothesis, output_dir / "main_result_best_mamba_vs_cnn.png")))
    figures.append(str(plot_ablation_groups(hypothesis, output_dir / "ablation_raster_locality_random.png")))
    figures.append(str(plot_random_control(hypothesis, output_dir / "ablation_randompermute_deficit.png")))
    figures.append(str(plot_paired_bootstrap(paired, output_dir / "statistical_paired_vs_cnn.png")))

    matching_path = Path(args.matching_contrast)
    p5_manifest: dict[str, Any]
    if matching_path.exists():
        p5_manifest = run_p5_analysis(
            run_level=run_level,
            fair_summary=fair_summary,
            matching_path=matching_path,
            output_dir=output_dir,
            permutation_samples=args.permutation_samples,
            seed=args.seed,
        )
        figures.extend(p5_manifest.get("figures", []))
    else:
        p5_manifest = {
            "status": "skipped",
            "reason": f"matching_contrast not found: {matching_path}",
            "required_file": str(matching_path),
        }

    manifest = {
        "runs_dir": str(Path(args.runs_dir)),
        "matching_contrast": str(matching_path),
        "output_dir": str(output_dir),
        "n_runs": int(len(run_level)),
        "figures": figures,
        "p5": p5_manifest,
    }
    (output_dir / "paper_analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def collect_run_metrics(runs_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    for metrics_path in sorted(runs_dir.glob("*/metrics.csv")):
        parsed = parse_run_dir(metrics_path.parent.name)
        if parsed is None:
            continue
        dataset, condition, seed = parsed
        metrics = pd.read_csv(metrics_path)
        if metrics.empty or "case_id" not in metrics or "dice" not in metrics:
            continue
        metrics["dice"] = pd.to_numeric(metrics["dice"], errors="coerce")
        metrics["hd95"] = pd.to_numeric(metrics.get("hd95", np.nan), errors="coerce")
        case_summary = (
            metrics.groupby("case_id", dropna=False)
            .agg(case_dice=("dice", "mean"), case_hd95=("hd95", "mean"))
            .reset_index()
        )
        for _, row in case_summary.iterrows():
            case_rows.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "seed": int(seed),
                    "case_id": str(row["case_id"]),
                    "case_dice": float(row["case_dice"]),
                    "case_hd95": float(row["case_hd95"]),
                }
            )
        run_rows.append(
            {
                "dataset": dataset,
                "condition": condition,
                "seed": int(seed),
                "run": metrics_path.parent.name,
                "case_dice_mean": float(case_summary["case_dice"].mean()),
                "case_dice_median": float(case_summary["case_dice"].median()),
                "case_dice_std_cases": float(case_summary["case_dice"].std(ddof=1)),
                "case_hd95_mean": float(case_summary["case_hd95"].mean()),
                "n_cases": int(case_summary["case_id"].nunique()),
                "n_rows": int(len(metrics)),
            }
        )
    return pd.DataFrame(run_rows), pd.DataFrame(case_rows)


def parse_run_dir(name: str) -> tuple[str, str, int] | None:
    match = RUN_RE.match(name)
    if not match:
        return None
    dataset_raw, condition_raw, seed_raw = match.groups()
    dataset = DATASET_NAME_MAP.get(dataset_raw, dataset_raw)
    condition = CONDITION_NAME_MAP[condition_raw]
    return dataset, condition, int(seed_raw)


def make_fair_summary(run_level: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, group in run_level.groupby("dataset"):
        seed_sets = {
            condition: set(group.loc[group["condition"] == condition, "seed"].astype(int))
            for condition in CONDITION_ORDER
            if condition in set(group["condition"])
        }
        if not seed_sets:
            continue
        common = set.intersection(*seed_sets.values())
        if not common:
            common = set.union(*seed_sets.values())
        common = sorted(common)
        for condition in CONDITION_ORDER:
            subset = group[(group["condition"] == condition) & (group["seed"].isin(common))]
            if subset.empty:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "common_seeds": ",".join(str(seed) for seed in sorted(subset["seed"].astype(int).unique())),
                    "n_seeds": int(subset["seed"].nunique()),
                    "mean_case_dice": float(subset["case_dice_mean"].mean()),
                    "std_across_seeds": _finite_or_nan(subset["case_dice_mean"].std(ddof=1)),
                    "mean_hd95": float(subset["case_hd95_mean"].mean()),
                }
            )
    output = pd.DataFrame(rows)
    output["dataset"] = pd.Categorical(output["dataset"], DATASET_ORDER, ordered=True)
    output["condition"] = pd.Categorical(output["condition"], CONDITION_ORDER, ordered=True)
    return output.sort_values(["dataset", "condition"]).reset_index(drop=True)


def make_hypothesis_summary(fair_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, group in fair_summary.groupby("dataset", observed=True):
        values = group.set_index("condition")["mean_case_dice"].to_dict()
        if not values:
            continue
        mamba_values = {key: values[key] for key in MAMBA_CONDITIONS if key in values}
        raster = np.mean([mamba_values[item] for item in ["Raster-H", "Raster-V"] if item in mamba_values])
        locality = np.mean([mamba_values[item] for item in ["Hilbert", "LocalWindow"] if item in mamba_values])
        random_value = mamba_values.get("RandomPermute", np.nan)
        best_mamba = max(mamba_values, key=mamba_values.get) if mamba_values else ""
        best_all = max(values, key=values.get)
        rows.append(
            {
                "dataset": str(dataset),
                "best_all": best_all,
                "best_all_dice": float(values[best_all]),
                "cnn": float(values.get("CNN", np.nan)),
                "best_mamba": best_mamba,
                "best_mamba_dice": float(mamba_values.get(best_mamba, np.nan)) if best_mamba else np.nan,
                "best_mamba_minus_cnn": float(mamba_values.get(best_mamba, np.nan) - values.get("CNN", np.nan))
                if best_mamba and "CNN" in values
                else np.nan,
                "raster_mean": float(raster),
                "locality_mean": float(locality),
                "random": float(random_value),
                "locality_minus_raster": float(locality - raster),
                "locality_minus_random": float(locality - random_value),
                "structured_best_minus_random": float(max(raster, locality) - random_value),
                "mamba_range": float(max(mamba_values.values()) - min(mamba_values.values())) if mamba_values else np.nan,
            }
        )
    output = pd.DataFrame(rows)
    output["dataset"] = pd.Categorical(output["dataset"], DATASET_ORDER, ordered=True)
    return output.sort_values("dataset").reset_index(drop=True)


def paired_vs_cnn(case_metrics: pd.DataFrame, bootstrap_samples: int, seed: int) -> pd.DataFrame:
    if case_metrics.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for dataset, group in case_metrics.groupby("dataset"):
        cnn = group[group["condition"] == "CNN"]
        for condition in MAMBA_CONDITIONS:
            target = group[group["condition"] == condition]
            merged = target.merge(cnn, on=["dataset", "seed", "case_id"], suffixes=("_condition", "_cnn"))
            if merged.empty:
                continue
            diffs = (merged["case_dice_condition"] - merged["case_dice_cnn"]).to_numpy(dtype=float)
            ci_low, ci_high = bootstrap_ci(diffs, rng, bootstrap_samples)
            rows.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "n_pairs": int(len(diffs)),
                    "mean_delta_vs_cnn": float(np.mean(diffs)),
                    "median_delta_vs_cnn": float(np.median(diffs)),
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "sign": "positive" if ci_low > 0 else "negative" if ci_high < 0 else "ns",
                }
            )
    output = pd.DataFrame(rows)
    output["dataset"] = pd.Categorical(output["dataset"], DATASET_ORDER, ordered=True)
    output["condition"] = pd.Categorical(output["condition"], MAMBA_CONDITIONS, ordered=True)
    return output.sort_values(["dataset", "condition"]).reset_index(drop=True)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, samples: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    means = [float(np.mean(rng.choice(values, size=values.size, replace=True))) for _ in range(max(1, samples))]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run_p5_analysis(
    *,
    run_level: pd.DataFrame,
    fair_summary: pd.DataFrame,
    matching_path: Path,
    output_dir: Path,
    permutation_samples: int,
    seed: int,
) -> dict[str, Any]:
    matching = pd.read_csv(matching_path)
    if "scan" in matching.columns and "condition" not in matching.columns:
        matching = matching.rename(columns={"scan": "condition"})
    required = {"dataset", "condition", "M_primary"}
    missing = required - set(matching.columns)
    if missing:
        return {"status": "skipped", "reason": f"matching_contrast missing columns: {sorted(missing)}"}

    points = fair_summary[fair_summary["condition"].isin(MAMBA_CONDITIONS)].merge(
        matching[["dataset", "condition", "M_primary"]],
        on=["dataset", "condition"],
        how="inner",
    )
    points["M_primary"] = pd.to_numeric(points["M_primary"], errors="coerce")
    points["z_dice"] = points.groupby("dataset", observed=True)["mean_case_dice"].transform(_zscore)
    points.to_csv(output_dir / "p5_matching_points.csv", index=False)

    corrs: list[dict[str, Any]] = []
    for dataset, group in points.groupby("dataset", observed=True):
        group = group.dropna(subset=["M_primary", "z_dice"])
        r = pearson_r(group["M_primary"].to_numpy(dtype=float), group["z_dice"].to_numpy(dtype=float))
        corrs.append({"dataset": str(dataset), "n_conditions": int(len(group)), "pearson_r": r})
    corr_frame = pd.DataFrame(corrs)
    corr_frame.to_csv(output_dir / "p5_per_dataset_correlations.csv", index=False)

    pooled = make_pooled_p5(run_level, matching, permutation_samples=permutation_samples, seed=seed)
    pd.DataFrame([pooled]).to_csv(output_dir / "p5_pooled_permutation.csv", index=False)

    figures = [
        str(plot_p5_facets(points, output_dir / "p5_matching_association_facets.png")),
        str(plot_p5_correlation_bars(corr_frame, output_dir / "p5_per_dataset_correlations.png")),
    ]
    return {
        "status": "ok",
        "matching_rows": int(len(matching)),
        "points": int(len(points)),
        "figures": figures,
        "pooled": pooled,
    }


def make_pooled_p5(
    run_level: pd.DataFrame,
    matching: pd.DataFrame,
    permutation_samples: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    rows = run_level[run_level["condition"].isin(MAMBA_CONDITIONS)].merge(
        matching[["dataset", "condition", "M_primary"]],
        on=["dataset", "condition"],
        how="inner",
    )
    rows["z_dice"] = rows.groupby("dataset")["case_dice_mean"].transform(_zscore)
    rows = rows.dropna(subset=["M_primary", "z_dice"])
    if len(rows) < 3:
        return {"n": int(len(rows)), "beta": float("nan"), "permutation_p": float("nan")}
    x = rows["M_primary"].to_numpy(dtype=float)
    y = rows["z_dice"].to_numpy(dtype=float)
    beta = ols_slope(x, y)
    permuted = []
    labels = rows["dataset"].to_numpy()
    y_perm = y.copy()
    for _ in range(max(1, permutation_samples)):
        for dataset in np.unique(labels):
            mask = labels == dataset
            y_perm[mask] = rng.permutation(y[mask])
        permuted.append(ols_slope(x, y_perm))
    p_value = (1.0 + sum(1 for value in permuted if value >= beta)) / (len(permuted) + 1.0)
    return {
        "n": int(len(rows)),
        "beta": float(beta),
        "permutation_p_one_sided_positive": float(p_value),
        "pearson_r": pearson_r(x, y),
    }


def plot_dice_heatmap(summary: pd.DataFrame, path: Path) -> Path:
    pivot = _pivot(summary, "mean_case_dice")
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    im = ax.imshow(pivot.values, cmap="YlGnBu", vmin=0.72, vmax=0.87, aspect="auto")
    decorate_heatmap(ax, pivot, higher_is_better=True, fmt="{:.3f}")
    ax.set_title("Main Performance: Mean Case-level Dice")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Dice")
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def plot_hd95_heatmap(summary: pd.DataFrame, path: Path) -> Path:
    pivot = _pivot(summary, "mean_hd95")
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    im = ax.imshow(pivot.values, cmap="OrRd", aspect="auto")
    decorate_heatmap(ax, pivot, higher_is_better=False, fmt="{:.2f}")
    ax.set_title("Boundary Metric: Mean HD95")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("HD95 (lower is better)")
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def decorate_heatmap(ax: Any, pivot: pd.DataFrame, *, higher_is_better: bool, fmt: str) -> None:
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    threshold = np.nanmedian(pivot.values)
    for i, dataset in enumerate(pivot.index):
        row = pivot.loc[dataset]
        best = row.idxmax() if higher_is_better else row.idxmin()
        for j, condition in enumerate(pivot.columns):
            value = float(row[condition])
            label = fmt.format(value)
            if condition == best:
                label += "*"
            color = "white" if (value > threshold if higher_is_better else value > threshold) else "black"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color=color,
                fontweight="bold" if condition == best else "normal",
            )


def plot_best_mamba_vs_cnn(hypothesis: pd.DataFrame, path: Path) -> Path:
    frame = hypothesis.set_index("dataset").reindex(DATASET_ORDER)
    values = frame["best_mamba_minus_cnn"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.bar(DATASET_ORDER, values, color=["#54A24B" if value > 0 else "#E45756" for value in values], width=0.62)
    ax.axhline(0.0, color="black", linewidth=0.9)
    for idx, value in enumerate(values):
        best = frame.iloc[idx]["best_mamba"]
        ax.text(
            idx,
            value + (0.0007 if value >= 0 else -0.0007),
            f"{best}\n{value:+.4f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
        )
    ax.set_ylabel("Dice difference vs CNN")
    ax.set_title("Best Mamba Scan Compared with CNN")
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def plot_ablation_groups(hypothesis: pd.DataFrame, path: Path) -> Path:
    frame = hypothesis.set_index("dataset").reindex(DATASET_ORDER)
    columns = [("raster_mean", "Raster mean", "#F58518"), ("locality_mean", "Locality mean", "#54A24B"), ("random", "RandomPermute", "#B279A2")]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    x = np.arange(len(frame))
    width = 0.24
    for idx, (column, label, color) in enumerate(columns):
        ax.bar(x + (idx - 1) * width, frame[column].to_numpy(dtype=float), width=width, label=label, color=color)
    ax.set_xticks(x, DATASET_ORDER)
    ax.set_ylabel("Mean case-level Dice")
    ax.set_title("Ablation/Control: Directional, Locality-preserving, and Random Scans")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.27))
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def plot_random_control(hypothesis: pd.DataFrame, path: Path) -> Path:
    frame = hypothesis.set_index("dataset").reindex(DATASET_ORDER)
    values = frame["structured_best_minus_random"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.bar(DATASET_ORDER, values, color="#B279A2", width=0.62)
    ax.axhline(0.0, color="black", linewidth=0.9)
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.0008, f"+{value:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Dice gain over RandomPermute")
    ax.set_title("Negative Control: Structured Scans vs RandomPermute")
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def plot_paired_bootstrap(paired: pd.DataFrame, path: Path) -> Path:
    if paired.empty:
        return path
    frame = paired.copy()
    frame["label"] = frame["dataset"].astype(str) + "\n" + frame["condition"].astype(str)
    x = np.arange(len(frame))
    y = frame["mean_delta_vs_cnn"].to_numpy(dtype=float)
    low = frame["bootstrap_ci_low"].to_numpy(dtype=float)
    high = frame["bootstrap_ci_high"].to_numpy(dtype=float)
    yerr = np.vstack([y - low, high - y])
    colors = ["#54A24B" if lo > 0 else "#E45756" if hi < 0 else "#9D9DA1" for lo, hi in zip(low, high)]
    fig, ax = plt.subplots(figsize=(max(9.0, len(frame) * 0.36), 4.5))
    ax.bar(x, y, color=colors, width=0.72)
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", elinewidth=0.8, capsize=2)
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_xticks(x, frame["label"], rotation=90, fontsize=7)
    ax.set_ylabel("Paired Dice difference vs CNN")
    ax.set_title("Case-level Bootstrap Comparisons Against CNN")
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def plot_p5_facets(points: pd.DataFrame, path: Path) -> Path:
    datasets = [dataset for dataset in DATASET_ORDER if dataset in set(points["dataset"].astype(str))]
    ncols = 3
    nrows = int(math.ceil(len(datasets) / ncols))
    fig, axes_grid = plt.subplots(nrows, ncols, figsize=(5.1 * ncols, 4.15 * nrows), sharey=True)
    axes = np.asarray(axes_grid).reshape(-1)
    y_values = points["z_dice"].to_numpy(dtype=float)
    y_min = min(-2.15, float(np.nanmin(y_values)) - 0.22)
    y_max = max(1.95, float(np.nanmax(y_values)) + 0.30)
    for ax, dataset in zip(axes, datasets):
        group = points[points["dataset"].astype(str) == dataset].dropna(subset=["M_primary", "z_dice"])
        ax.scatter(group["M_primary"], group["z_dice"], color="#4C78A8", s=76)
        for _, row in group.iterrows():
            condition = str(row["condition"])
            dx, dy, ha, va = P5_LABEL_OFFSETS.get((dataset, condition), (5, 5, "left", "bottom"))
            ax.annotate(
                condition,
                (row["M_primary"], row["z_dice"]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=12,
                ha=ha,
                va=va,
                clip_on=True,
                bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.72},
            )
        if len(group) >= 2:
            x = group["M_primary"].to_numpy(dtype=float)
            y = group["z_dice"].to_numpy(dtype=float)
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
            ax.plot(xs, slope * xs + intercept, linestyle="--", color="black", linewidth=1.3)
            title = f"{dataset}\nr={pearson_r(x, y):+.2f}"
        else:
            title = dataset
        ax.set_title(title, fontsize=14, pad=8)
        ax.set_xlabel("M_primary", fontsize=12)
        ax.set_xlim(-0.04, 0.90)
        ax.set_ylim(y_min, y_max)
        ax.axhline(0.0, color="black", alpha=0.25, linewidth=1.0)
        ax.tick_params(axis="both", labelsize=12)
    for ax in axes[len(datasets) :]:
        ax.axis("off")
    axes[0].set_ylabel("Within-dataset standardized Dice")
    axes[0].yaxis.label.set_size(12)
    if len(axes) > ncols:
        axes[ncols].set_ylabel("Within-dataset standardized Dice", fontsize=12)
    fig.suptitle("Geometry-Matching Association for Mamba Scan Strategies", y=0.985, fontsize=15)
    fig.subplots_adjust(left=0.070, right=0.985, bottom=0.090, top=0.865, wspace=0.24, hspace=0.42)
    fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path


def plot_p5_correlation_bars(corr_frame: pd.DataFrame, path: Path) -> Path:
    frame = corr_frame.copy()
    frame["dataset"] = pd.Categorical(frame["dataset"], DATASET_ORDER, ordered=True)
    frame = frame.sort_values("dataset")
    values = frame["pearson_r"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(frame["dataset"].astype(str), values, color=["#54A24B" if value >= 0 else "#E45756" for value in values], width=0.62)
    ax.axhline(0.0, color="black", linewidth=0.9)
    for idx, value in enumerate(values):
        ax.text(idx, value + (0.03 if value >= 0 else -0.03), f"{value:+.2f}", ha="center", va="bottom" if value >= 0 else "top")
    ax.set_ylabel("Pearson r")
    ax.set_title("Per-dataset M_primary vs Standardized Dice Correlation")
    ax.set_ylim(-1.0, 1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def _pivot(summary: pd.DataFrame, value: str) -> pd.DataFrame:
    pivot = summary.pivot(index="dataset", columns="condition", values=value)
    return pivot.reindex(index=DATASET_ORDER, columns=CONDITION_ORDER)


def _zscore(values: pd.Series) -> pd.Series:
    std = values.std(ddof=0)
    if not np.isfinite(std) or std < 1e-12:
        return values * 0.0
    return (values - values.mean()) / std


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_centered = x - np.mean(x)
    denom = float(np.sum(x_centered * x_centered))
    if denom < 1e-12:
        return float("nan")
    return float(np.sum(x_centered * (y - np.mean(y))) / denom)


def _finite_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


if __name__ == "__main__":
    main()
