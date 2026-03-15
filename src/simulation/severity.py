from __future__ import annotations

import numpy as np


SEVERITY_LABELS = {
    1: "Normal",
    2: "Warning",
    3: "Hazardous",
}


def severity_to_norm(severity: int | np.ndarray) -> float | np.ndarray:
    return (np.asarray(severity) - 1.0) / 2.0


def map_pm25_to_severity(values: float | np.ndarray, thresholds: dict[str, float]) -> int | np.ndarray:
    values_array = np.asarray(values, dtype=float)
    normal_max = thresholds["normal_max"]
    warning_max = thresholds["warning_max"]
    severity = np.where(values_array <= normal_max, 1, np.where(values_array <= warning_max, 2, 3))
    if np.isscalar(values):
        return int(severity.item())
    return severity


def severity_label(severity: int) -> str:
    return SEVERITY_LABELS[int(severity)]
