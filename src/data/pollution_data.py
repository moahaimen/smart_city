from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import ensure_dir


@dataclass
class ScenarioData:
    name: str
    scenario_group: str
    rounds: int
    node_count: int
    area_size: float
    frame: pd.DataFrame
    node_metadata: pd.DataFrame
    feature_columns: list[str]
    target_column: str
    network_overrides: dict[str, float]


def _hotspot_relevance(x: float, y: float, hotspot_centers: list[list[float]], area_size: float) -> float:
    decay = max(area_size * 0.18, 1.0)
    scores = [np.exp(-np.hypot(x - cx, y - cy) / decay) for cx, cy in hotspot_centers]
    return float(np.clip(max(scores), 0.0, 1.0))


def generate_node_metadata(node_count: int, area_size: float, hotspot_centers: list[list[float]], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    for node_id in range(1, node_count + 1):
        x = float(rng.uniform(0.0, area_size))
        y = float(rng.uniform(0.0, area_size))
        hotspot = _hotspot_relevance(x, y, hotspot_centers, area_size)
        records.append(
            {
                "node_id": node_id,
                "x": x,
                "y": y,
                "hotspot_relevance": hotspot,
            }
        )
    return pd.DataFrame.from_records(records)


def _scenario_adjustment(name: str, step: int, total_steps: int, hotspot: float) -> float:
    ratio = step / max(total_steps - 1, 1)
    if name == "normal":
        return 0.0
    if name == "rising_warning":
        return 15.0 + 55.0 * ratio + 10.0 * hotspot
    if name == "hazardous_spike":
        center = 0.58 * total_steps
        width = max(3.0, 0.08 * total_steps)
        spike = 135.0 * (1.0 + 0.7 * hotspot) * np.exp(-((step - center) ** 2) / (2.0 * width ** 2))
        return spike
    if name == "hotspot_heavy":
        return 20.0 + 50.0 * hotspot + 8.0 * np.sin(2.0 * np.pi * step / 48.0)
    return 0.0


def generate_scenario_timeseries(
    scenario_name: str,
    node_metadata: pd.DataFrame,
    total_steps: int,
    area_size: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records: list[dict[str, float | int | str]] = []

    for row in node_metadata.itertuples(index=False):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        pm25_prev = 25.0 + 30.0 * row.hotspot_relevance + rng.normal(0.0, 3.0)
        for step in range(total_steps):
            hour = step % 24
            diurnal = 18.0 + 14.0 * np.sin(2.0 * np.pi * (hour - 7.0) / 24.0 + phase)
            commuter = 8.0 * np.sin(4.0 * np.pi * hour / 24.0 + phase / 2.0)
            scenario_term = _scenario_adjustment(scenario_name, step, total_steps, row.hotspot_relevance)
            trend_target = 28.0 + diurnal + commuter + 22.0 * row.hotspot_relevance + scenario_term
            pm25 = 0.68 * pm25_prev + 0.32 * trend_target + rng.normal(0.0, 4.0)
            pm25 = float(np.clip(pm25, 5.0, 260.0))

            pm10 = float(np.clip(pm25 * 1.32 + rng.normal(0.0, 7.0), 10.0, 380.0))
            co = float(np.clip(0.22 + 0.0085 * pm25 + 0.35 * row.hotspot_relevance + rng.normal(0.0, 0.05), 0.05, 6.0))
            no2 = float(np.clip(10.0 + 0.34 * pm25 + 9.0 * row.hotspot_relevance + rng.normal(0.0, 3.0), 5.0, 220.0))
            temperature = float(23.0 + 7.5 * np.sin(2.0 * np.pi * (hour - 14.0) / 24.0) + rng.normal(0.0, 1.1))
            humidity = float(np.clip(62.0 - 0.75 * (temperature - 23.0) + rng.normal(0.0, 4.5), 18.0, 92.0))
            hour_angle = 2.0 * np.pi * hour / 24.0
            records.append(
                {
                    "scenario": scenario_name,
                    "node_id": int(row.node_id),
                    "step": step,
                    "hour": hour,
                    "x": float(row.x),
                    "y": float(row.y),
                    "hotspot_relevance": float(row.hotspot_relevance),
                    "pm25": pm25,
                    "pm10": pm10,
                    "co": co,
                    "no2": no2,
                    "temperature": temperature,
                    "humidity": humidity,
                    "hour_sin": float(np.sin(hour_angle)),
                    "hour_cos": float(np.cos(hour_angle)),
                }
            )
            pm25_prev = pm25

    frame = pd.DataFrame.from_records(records)
    frame["area_size"] = area_size
    return frame


def prepare_training_dataframe(config: dict, seed: int) -> tuple[pd.DataFrame, str]:
    data_config = config["data"]
    real_csv = Path(data_config["real_csv_path"])
    feature_columns = data_config["feature_columns"]

    if data_config.get("use_real_if_available", True) and real_csv.exists():
        frame = pd.read_csv(real_csv)
        frame = frame.rename(columns={column.lower(): column.lower() for column in frame.columns})
        if "node_id" not in frame.columns:
            if "sensor_id" in frame.columns:
                frame = frame.rename(columns={"sensor_id": "node_id"})
            else:
                frame["node_id"] = 1
        if "step" not in frame.columns:
            frame["step"] = frame.groupby("node_id").cumcount()
        if "scenario" not in frame.columns:
            frame["scenario"] = "real_dataset"
        if "hour" not in frame.columns:
            frame["hour"] = frame["step"] % 24
        if "hour_sin" not in frame.columns:
            angle = 2.0 * np.pi * frame["hour"] / 24.0
            frame["hour_sin"] = np.sin(angle)
            frame["hour_cos"] = np.cos(angle)
        for missing in ("pm10", "co", "no2", "temperature", "humidity"):
            if missing not in frame.columns:
                raise ValueError(f"Real dataset is missing required column: {missing}")
        if "hotspot_relevance" not in frame.columns:
            frame["hotspot_relevance"] = 0.5
        output_path = Path(data_config["synthetic_output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
        return frame[["scenario", "node_id", "step", "hotspot_relevance", *feature_columns]], "real_csv"

    training_frames = []
    hotspot_centers = data_config["hotspot_centers"]
    node_count = config["scenarios"]["evaluation"][0]["node_count"]
    area_size = config["scenarios"]["evaluation"][0]["area_size"]
    total_steps = data_config["training_steps"]
    for index, scenario_name in enumerate(config["scenarios"]["training_mix"]):
        metadata = generate_node_metadata(
            node_count=node_count,
            area_size=area_size,
            hotspot_centers=hotspot_centers,
            seed=seed + 101 * (index + 1),
        )
        scenario_frame = generate_scenario_timeseries(
            scenario_name=scenario_name,
            node_metadata=metadata,
            total_steps=total_steps,
            area_size=area_size,
            seed=seed + 211 * (index + 1),
        )
        training_frames.append(scenario_frame)

    frame = pd.concat(training_frames, ignore_index=True)
    output_path = Path(data_config["synthetic_output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame, "synthetic"


def build_sequence_splits(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    window_size: int,
    horizon: int,
    train_fraction: float,
    val_fraction: float,
) -> dict[str, dict[str, np.ndarray | pd.DataFrame]]:
    buckets: dict[str, list[np.ndarray] | list[dict[str, int | str]]] = {
        "train_x": [],
        "train_y": [],
        "val_x": [],
        "val_y": [],
        "test_x": [],
        "test_y": [],
        "train_meta": [],
        "val_meta": [],
        "test_meta": [],
    }

    grouped = frame.sort_values(["scenario", "node_id", "step"]).groupby(["scenario", "node_id"], sort=False)
    for (scenario_name, node_id), group in grouped:
        features = group[feature_columns].to_numpy(dtype=np.float32)
        targets = group[target_column].to_numpy(dtype=np.float32)
        hotspot = group["hotspot_relevance"].to_numpy(dtype=np.float32)
        total = len(group) - window_size - horizon + 1
        if total <= 0:
            continue

        train_cut = int(total * train_fraction)
        val_cut = int(total * (train_fraction + val_fraction))

        for start in range(total):
            split_name = "train" if start < train_cut else "val" if start < val_cut else "test"
            end = start + window_size
            target_index = end + horizon - 1
            buckets[f"{split_name}_x"].append(features[start:end])
            buckets[f"{split_name}_y"].append(targets[target_index])
            buckets[f"{split_name}_meta"].append(
                {
                    "scenario": scenario_name,
                    "node_id": int(node_id),
                    "target_step": int(group.iloc[target_index]["step"]),
                    "hotspot_relevance": float(hotspot[target_index]),
                }
            )

    outputs: dict[str, dict[str, np.ndarray | pd.DataFrame]] = {}
    for split_name in ("train", "val", "test"):
        x_key = f"{split_name}_x"
        y_key = f"{split_name}_y"
        meta_key = f"{split_name}_meta"
        outputs[split_name] = {
            "x": np.asarray(buckets[x_key], dtype=np.float32),
            "y": np.asarray(buckets[y_key], dtype=np.float32),
            "meta": pd.DataFrame.from_records(buckets[meta_key]),
        }
    return outputs


def build_scenario_bundle(config: dict, seed: int) -> dict[str, ScenarioData]:
    data_config = config["data"]
    scenarios: dict[str, ScenarioData] = {}
    all_scenarios = list(config["scenarios"]["evaluation"]) + list(config["scenarios"]["sensitivity"])

    for index, scenario_cfg in enumerate(all_scenarios):
        scenario_name = scenario_cfg.get("base_scenario", scenario_cfg["name"])
        rounds = int(scenario_cfg["rounds"])
        node_count = int(scenario_cfg["node_count"])
        area_size = float(scenario_cfg["area_size"])
        total_steps = rounds + data_config["window_size"] + data_config["horizon"]
        metadata = generate_node_metadata(
            node_count=node_count,
            area_size=area_size,
            hotspot_centers=data_config["hotspot_centers"],
            seed=seed + 1000 + index * 17,
        )
        frame = generate_scenario_timeseries(
            scenario_name=scenario_name,
            node_metadata=metadata,
            total_steps=total_steps,
            area_size=area_size,
            seed=seed + 2000 + index * 19,
        )
        scenarios[scenario_cfg["name"]] = ScenarioData(
            name=scenario_cfg["name"],
            scenario_group="evaluation" if "base_scenario" not in scenario_cfg else "sensitivity",
            rounds=rounds,
            node_count=node_count,
            area_size=area_size,
            frame=frame,
            node_metadata=metadata,
            feature_columns=list(data_config["feature_columns"]),
            target_column=data_config["target_column"],
            network_overrides={
                key: value
                for key, value in scenario_cfg.items()
                if key in {"initial_energy"}
            },
        )
    return scenarios


def persist_scenarios(scenarios: dict[str, ScenarioData], output_dir: str | Path) -> None:
    base = ensure_dir(output_dir)
    for scenario in scenarios.values():
        scenario.frame.to_csv(base / f"{scenario.name}.csv", index=False)
