"""Phase B protocol utilities for the scan-strategy experiment."""

from .config import ConditionSpec, DatasetSpec, PhaseBConfig
from .dataset import ManifestSegmentationDataset
from .matrix import generate_run_matrix, write_run_matrix
from .metrics import aggregate_case_metrics, read_case_metrics, write_metrics_summary
from .splits import make_phase_b_splits
from .training import TrainingConfig, run_training

__all__ = [
    "ConditionSpec",
    "DatasetSpec",
    "PhaseBConfig",
    "ManifestSegmentationDataset",
    "TrainingConfig",
    "aggregate_case_metrics",
    "generate_run_matrix",
    "make_phase_b_splits",
    "read_case_metrics",
    "run_training",
    "write_metrics_summary",
    "write_run_matrix",
]
