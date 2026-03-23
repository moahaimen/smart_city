from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


def confidence_interval(values: np.ndarray | list[float], confidence_level: float = 0.95) -> tuple[float, float, float]:
    sample = np.asarray(values, dtype=float)
    sample = sample[~np.isnan(sample)]
    if sample.size == 0:
        return float("nan"), float("nan"), float("nan")
    if sample.size == 1:
        value = float(sample[0])
        return value, value, 0.0

    mean = float(sample.mean())
    std = float(sample.std(ddof=1))
    half_width = float(t.ppf((1.0 + confidence_level) / 2.0, df=sample.size - 1) * std / np.sqrt(sample.size))
    return mean - half_width, mean + half_width, half_width


def aggregate_metric_frame(
    frame: pd.DataFrame,
    group_cols: list[str],
    metric_cols: list[str],
    confidence_level: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_rows: list[dict[str, float | str | int]] = []
    wide_rows: list[dict[str, float | str | int]] = []

    for group_values, group_df in frame.groupby(group_cols, sort=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group_payload = dict(zip(group_cols, group_values))
        wide_row = dict(group_payload)
        for metric in metric_cols:
            values = group_df[metric].to_numpy(dtype=float)
            ci_low, ci_high, ci_half = confidence_interval(values, confidence_level)
            row = {
                **group_payload,
                "metric": metric,
                "count": int(values.size),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "ci95_half_width": ci_half,
            }
            long_rows.append(row)
            wide_row[f"{metric}_mean"] = row["mean"]
            wide_row[f"{metric}_count"] = row["count"]
            wide_row[f"{metric}_std"] = row["std"]
            wide_row[f"{metric}_min"] = row["min"]
            wide_row[f"{metric}_max"] = row["max"]
            wide_row[f"{metric}_ci95_low"] = row["ci95_low"]
            wide_row[f"{metric}_ci95_high"] = row["ci95_high"]
            wide_row[f"{metric}_ci95_half_width"] = row["ci95_half_width"]
        wide_rows.append(wide_row)

    return pd.DataFrame.from_records(long_rows), pd.DataFrame.from_records(wide_rows)


def save_markdown_table(frame: pd.DataFrame, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(frame.to_markdown(index=False), encoding="utf-8")
