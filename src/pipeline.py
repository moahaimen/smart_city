from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.pollution_data import build_scenario_bundle, build_sequence_splits, persist_scenarios, prepare_training_dataframe
from src.metrics.evaluation import as_row
from src.metrics.reporting import plot_alive_nodes, plot_predictions, plot_residual_energy, plot_summary_bar, plot_training_curves
from src.models.tcn_predictor import PredictorBundle, train_tcn_regressor
from src.simulation.engine import run_protocol_simulation
from src.utils.io import prepare_results_dirs, save_json
from src.utils.reproducibility import set_global_seed


def train_predictor(config: dict, results_dirs: dict[str, Path], seed: int) -> dict[str, object]:
    set_global_seed(seed)
    training_frame, data_source = prepare_training_dataframe(config, seed)
    sequence_splits = build_sequence_splits(
        frame=training_frame,
        feature_columns=config["data"]["feature_columns"],
        target_column=config["data"]["target_column"],
        window_size=int(config["data"]["window_size"]),
        horizon=int(config["data"]["horizon"]),
        train_fraction=float(config["model"]["train_fraction"]),
        val_fraction=float(config["model"]["val_fraction"]),
    )
    train_result = train_tcn_regressor(sequence_splits, config, results_dirs, seed)
    plot_training_curves(train_result["history"], results_dirs["figures"] / "tcn_training_curves.png")
    plot_predictions(train_result["predictions"], results_dirs["figures"] / "tcn_predictions.png")
    save_json(
        results_dirs["logs"] / "training_summary.json",
        {
            "model_type": "tcn",
            "data_source": data_source,
            "train_sequences": int(len(sequence_splits["train"]["x"])),
            "val_sequences": int(len(sequence_splits["val"]["x"])),
            "test_sequences": int(len(sequence_splits["test"]["x"])),
            **as_row(train_result["regression_metrics"], prefix="regression"),
            **as_row(train_result["classification_metrics"], prefix="classification"),
        },
    )
    return train_result


def run_experiment_suite(config: dict, predictor: PredictorBundle, results_dirs: dict[str, Path], seed: int) -> dict[str, pd.DataFrame]:
    scenarios = build_scenario_bundle(config, seed)
    persist_scenarios(scenarios, Path("data/processed/scenarios"))

    summary_rows = []
    rounds = []
    for scenario_index, scenario in enumerate(scenarios.values()):
        for protocol_index, protocol_name in enumerate(config["protocols"]):
            summary, rounds_df = run_protocol_simulation(
                scenario=scenario,
                predictor=predictor,
                config=config,
                protocol_name=protocol_name,
                seed=seed + scenario_index * 31 + protocol_index * 7,
            )
            summary_rows.append(summary)
            rounds.append(rounds_df)

    summary_df = pd.DataFrame(summary_rows)
    rounds_df = pd.concat(rounds, ignore_index=True)

    summary_df.to_csv(results_dirs["tables"] / "network_summary.csv", index=False)
    rounds_df.to_csv(results_dirs["logs"] / "network_round_metrics.csv", index=False)
    summary_df[summary_df["scenario_group"] == "evaluation"].to_csv(results_dirs["tables"] / "paper_summary_table.csv", index=False)
    summary_df[summary_df["scenario_group"] == "sensitivity"].to_csv(results_dirs["tables"] / "sensitivity_summary.csv", index=False)

    plot_alive_nodes(rounds_df, results_dirs["figures"] / "alive_nodes_vs_rounds.png")
    plot_residual_energy(rounds_df, results_dirs["figures"] / "residual_energy_vs_rounds.png")
    plot_summary_bar(summary_df, "packet_delivery_ratio", "Packet Delivery Ratio", "PDR", results_dirs["figures"] / "packet_delivery_ratio_comparison.png")
    plot_summary_bar(summary_df, "end_to_end_delay", "End-to-End Delay", "Delay (hops)", results_dirs["figures"] / "delay_comparison.png")
    plot_summary_bar(summary_df, "average_aoi", "Average AoI", "AoI (rounds)", results_dirs["figures"] / "aoi_comparison.png")
    plot_summary_bar(
        summary_df,
        "hazardous_event_delivery_success_rate",
        "Hazardous Event Delivery Success",
        "Success rate",
        results_dirs["figures"] / "hazardous_event_success_comparison.png",
    )

    return {"summary": summary_df, "rounds": rounds_df}


def run_full_pipeline(config: dict) -> dict[str, object]:
    results_dirs = prepare_results_dirs(config.get("output_root", "results"))
    seed = int(config["seed"])
    train_result = train_predictor(config, results_dirs, seed)
    experiment_result = run_experiment_suite(config, train_result["bundle"], results_dirs, seed)

    final_summary = {
        "seed": seed,
        "model_type": "tcn",
        "model_checkpoint": str(train_result["bundle"].checkpoint_path),
        "regression_metrics": train_result["regression_metrics"],
        "classification_metrics": train_result["classification_metrics"],
        "network_summary_table": str(results_dirs["tables"] / "network_summary.csv"),
        "paper_summary_table": str(results_dirs["tables"] / "paper_summary_table.csv"),
        "round_metrics_log": str(results_dirs["logs"] / "network_round_metrics.csv"),
    }
    save_json(results_dirs["logs"] / "pipeline_summary.json", final_summary)
    return {
        "results_dirs": results_dirs,
        "training": train_result,
        "experiments": experiment_result,
        "summary": final_summary,
    }


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML configuration file.")
