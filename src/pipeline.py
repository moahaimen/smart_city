from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.pollution_data import build_sequence_splits, prepare_training_dataframe
from src.metrics.evaluation import as_row
from src.metrics.reporting import plot_predictions, plot_training_curves
from src.models.tcn_predictor import PredictorBundle, train_tcn_regressor
from src.study import (
    generate_ablation_outputs,
    generate_main_study_outputs,
    persist_tcn_validation_artifacts,
    run_multi_seed_study,
    verify_summary_outputs,
    write_reviewer_outputs,
)
from src.utils.io import prepare_results_dirs, save_json
from src.utils.reproducibility import set_global_seed


def _build_sequence_payload(config: dict[str, Any], seed: int) -> tuple[pd.DataFrame, str, dict[str, dict[str, object]], dict[str, int]]:
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
    split_sizes = {
        "train_sequences": int(len(sequence_splits["train"]["x"])),
        "val_sequences": int(len(sequence_splits["val"]["x"])),
        "test_sequences": int(len(sequence_splits["test"]["x"])),
    }
    return training_frame, data_source, sequence_splits, split_sizes


def train_predictor(config: dict[str, Any], results_dirs: dict[str, Path], seed: int) -> dict[str, object]:
    set_global_seed(seed)
    training_frame, data_source, sequence_splits, split_sizes = _build_sequence_payload(config, seed)
    train_result = train_tcn_regressor(sequence_splits, config, results_dirs, seed)

    plot_training_curves(train_result["history"], results_dirs["figures"] / "tcn_training_curves.png")
    plot_predictions(train_result["predictions"], results_dirs["figures"] / "tcn_prediction_samples.png")

    metrics_payload = {
        "model_type": "tcn",
        "data_source": data_source,
        **split_sizes,
        "regression": train_result["regression_metrics"],
        "classification": train_result["classification_metrics"],
    }
    config_payload = {
        "seed": seed,
        "data_source": data_source,
        "data": config["data"],
        "model": config["model"],
        "severity": config["severity"],
        "split_sizes": split_sizes,
        "training_rows": int(len(training_frame)),
    }
    persist_tcn_validation_artifacts(results_dirs, metrics_payload, config_payload)
    save_json(
        results_dirs["logs"] / "training_summary.json",
        {
            "model_type": "tcn",
            "data_source": data_source,
            **split_sizes,
            **as_row(train_result["regression_metrics"], prefix="regression"),
            **as_row(train_result["classification_metrics"], prefix="classification"),
            "checkpoint_path": str(train_result["bundle"].checkpoint_path),
        },
    )
    return {
        **train_result,
        "data_source": data_source,
        "split_sizes": split_sizes,
        "metrics_payload": metrics_payload,
        "config_payload": config_payload,
    }


def load_or_train_predictor(
    config: dict[str, Any],
    results_dirs: dict[str, Path],
    checkpoint: str | Path | None = None,
) -> tuple[PredictorBundle, dict[str, object] | None]:
    checkpoint_path = Path(checkpoint) if checkpoint else None
    if checkpoint_path and checkpoint_path.exists():
        return PredictorBundle.load(checkpoint_path=checkpoint_path), None
    training = train_predictor(config, results_dirs, int(config["seed"]))
    return training["bundle"], training


def run_main_experiments(config: dict[str, Any], predictor: PredictorBundle, results_dirs: dict[str, Path]) -> dict[str, object]:
    study_config = config["study"]
    result = run_multi_seed_study(
        config=config,
        predictor=predictor,
        results_dirs=results_dirs,
        study_name="main_comparisons",
        seeds=[int(seed) for seed in study_config["seeds"]],
        scenario_names=list(study_config["main_scenarios"]),
        protocol_names=list(study_config["main_protocols"]),
    )
    paper_summary_df = generate_main_study_outputs(results_dirs, result["aggregated_wide"], result["rounds"])
    verify_summary_outputs([result["per_seed"], result["aggregated_long"], result["aggregated_wide"], paper_summary_df])
    result["paper_summary"] = paper_summary_df
    return result


def run_ablation_experiments(config: dict[str, Any], predictor: PredictorBundle, results_dirs: dict[str, Path]) -> dict[str, object]:
    study_config = config["study"]
    result = run_multi_seed_study(
        config=config,
        predictor=predictor,
        results_dirs=results_dirs,
        study_name="ablation",
        seeds=[int(seed) for seed in study_config["seeds"]],
        scenario_names=list(study_config["ablation_scenarios"]),
        protocol_names=list(study_config["ablation_protocols"]),
    )
    generate_ablation_outputs(results_dirs, result["aggregated_wide"])
    verify_summary_outputs([result["per_seed"], result["aggregated_long"], result["aggregated_wide"]])
    return result


def regenerate_figures(config: dict[str, Any], results_dirs: dict[str, Path]) -> dict[str, Path]:
    history_path = results_dirs["tables"] / "tcn_training_history.csv"
    predictions_path = results_dirs["tables"] / "tcn_test_predictions.csv"
    summary_path = results_dirs["tables"] / "scenario_protocol_summary.csv"
    rounds_path = results_dirs["logs"] / "per_seed_round_metrics.csv"
    ablation_path = results_dirs["tables"] / "ablation_summary_table.csv"

    history_df = pd.read_csv(history_path)
    predictions_df = pd.read_csv(predictions_path)
    aggregated_df = pd.read_csv(summary_path)
    rounds_df = pd.read_csv(rounds_path)

    plot_training_curves(history_df, results_dirs["figures"] / "tcn_training_curves.png")
    plot_predictions(predictions_df, results_dirs["figures"] / "tcn_prediction_samples.png")
    generate_main_study_outputs(results_dirs, aggregated_df, rounds_df)
    if ablation_path.exists():
        generate_ablation_outputs(results_dirs, pd.read_csv(ablation_path))

    return {
        "training_curves": results_dirs["figures"] / "tcn_training_curves.png",
        "prediction_samples": results_dirs["figures"] / "tcn_prediction_samples.png",
        "alive_nodes": results_dirs["figures"] / "alive_nodes_vs_rounds.png",
        "residual_energy": results_dirs["figures"] / "residual_energy_vs_rounds.png",
        "packet_delivery_ratio": results_dirs["figures"] / "packet_delivery_ratio_comparison.png",
        "delay": results_dirs["figures"] / "delay_comparison.png",
        "average_aoi": results_dirs["figures"] / "average_aoi_comparison.png",
        "hazardous_success": results_dirs["figures"] / "hazardous_event_success_comparison.png",
        "ablation": results_dirs["figures"] / "ablation_comparison.png",
    }


def run_full_pipeline(config: dict[str, Any]) -> dict[str, object]:
    results_dirs = prepare_results_dirs(config.get("output_root", "results"))
    training = train_predictor(config, results_dirs, int(config["seed"]))
    main_result = run_main_experiments(config, training["bundle"], results_dirs)
    ablation_result = run_ablation_experiments(config, training["bundle"], results_dirs)
    write_reviewer_outputs(
        root=Path("."),
        fairness_report_path=results_dirs["logs"] / "fairness_report.json",
        multi_seed_count=len(config["study"]["seeds"]),
        ablation_protocols=list(config["study"]["ablation_protocols"]),
    )

    final_summary = {
        "seed": int(config["seed"]),
        "study_seeds": [int(seed) for seed in config["study"]["seeds"]],
        "model_type": "tcn",
        "proposed_protocol": "tcn_predictive_pollution_aware_leach",
        "paper_short_name": "TCN-PPA-LEACH",
        "model_checkpoint": str(training["bundle"].checkpoint_path),
        "data_source": training["data_source"],
        "regression_metrics": training["regression_metrics"],
        "classification_metrics": training["classification_metrics"],
        "main_per_seed_results": str(results_dirs["tables"] / "per_seed_results.csv"),
        "main_aggregated_results": str(results_dirs["tables"] / "aggregated_results.csv"),
        "main_summary_table": str(results_dirs["tables"] / "scenario_protocol_summary.csv"),
        "ablation_results": str(results_dirs["tables"] / "ablation_results.csv"),
        "ablation_summary_table": str(results_dirs["tables"] / "ablation_summary_table.csv"),
        "paper_summary_table": str(results_dirs["tables"] / "paper_summary_table.csv"),
        "paper_summary_markdown": str(results_dirs["tables"] / "paper_summary_table.md"),
        "main_round_metrics": str(results_dirs["logs"] / "per_seed_round_metrics.csv"),
        "ablation_round_metrics": str(results_dirs["logs"] / "ablation_round_metrics.csv"),
        "fairness_report": str(results_dirs["logs"] / "fairness_report.json"),
        "ablation_fairness_report": str(results_dirs["logs"] / "ablation_fairness_report.json"),
        "tcn_test_metrics": str(results_dirs["logs"] / "test_metrics.json"),
        "tcn_config_used": str(results_dirs["logs"] / "tcn_config_used.json"),
        "reviewer_checklist": str(Path("reviewer_checklist.md")),
        "methods_snapshot": str(Path("methods_snapshot.md")),
        "limitations": str(Path("limitations.md")),
    }
    save_json(results_dirs["logs"] / "pipeline_summary.json", final_summary)
    return {
        "results_dirs": results_dirs,
        "training": training,
        "main": main_result,
        "ablation": ablation_result,
        "summary": final_summary,
    }


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML configuration file.")
