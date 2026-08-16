from __future__ import annotations

import numpy as np


PCA_COLUMNS = ["D", "L", "S", "B", "C"]


def geometry_pca(dataset_rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    if not dataset_rows:
        return []
    names = [str(row["dataset"]) for row in dataset_rows]
    matrix = np.asarray([[float(row.get(col, float("nan"))) for col in PCA_COLUMNS] for row in dataset_rows], dtype=np.float64)
    col_means = np.nanmean(matrix, axis=0)
    inds = np.where(~np.isfinite(matrix))
    matrix[inds] = np.take(col_means, inds[1])
    if len(dataset_rows) == 1:
        return [{"dataset": names[0], "pc1": 0.0, "pc2": 0.0, "nearest_neighbor_distance": float("nan")}]
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    standardized = centered / std
    _, _, vt = np.linalg.svd(standardized, full_matrices=False)
    components = standardized @ vt[:2].T
    if components.shape[1] == 1:
        components = np.column_stack([components[:, 0], np.zeros(len(dataset_rows))])
    distances = _nearest_distances(components)
    return [
        {
            "dataset": name,
            "pc1": float(components[idx, 0]),
            "pc2": float(components[idx, 1]),
            "nearest_neighbor_distance": float(distances[idx]),
        }
        for idx, name in enumerate(names)
    ]


def _nearest_distances(points: np.ndarray) -> np.ndarray:
    if len(points) <= 1:
        return np.asarray([np.nan])
    distances = np.full(len(points), np.inf)
    for i in range(len(points)):
        for j in range(len(points)):
            if i == j:
                continue
            distances[i] = min(distances[i], float(np.linalg.norm(points[i] - points[j])))
    return distances
