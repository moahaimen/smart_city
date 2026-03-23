from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.pollution_data import build_scenario_bundle, persist_scenarios
from src.metrics.fairness import build_fairness_report
from src.metrics.reporting import plot_ablation_bar, plot_alive_nodes, plot_residual_energy, plot_summary_bar
from src.metrics.statistics import aggregate_metric_frame, save_markdown_table
from src.simulation.engine import run_protocol_simulation
from src.utils.io import save_json
from src.utils.naming import protocol_label, scenario_label
from src.utils.reviewer_docs import build_paper_summary_table, write_limitations, write_methods_snapshot, write_reviewer_checklist


MAIN_METRICS = [
    "fnd",
    "hnd",
    "lnd",
    "average_residual_energy_final_round",
    "packet_delivery_ratio",
    "end_to_end_delay",
    "average_aoi",
    "hazardous_event_delivery_success_rate",
    "packets_generated",
    "packets_delivered",
    "hazardous_packets_generated",
    "hazardous_packets_delivered",
]


def _selected_scenarios(config: dict, seed: int, scenario_names: list[str]) -> dict:
    scenarios = build_scenario_bundle(config, seed)
    return {name: scenarios[name] for name in scenario_names}


def run_multi_seed_study(
    config: dict,
    predictor,
    results_dirs: dict[str, Path],
    study_name: str,
    seeds: list[int],
    scenario_names: list[str],
    protocol_names: list[str],
) -> dict[str, pd.DataFrame]:
    per_seed_rows = []
    round_rows = []
    assumption_rows = []

    for seed in seeds:
        scenarios = _selected_scenarios(config, seed, scenario_names)
        persist_scenarios(scenarios, Path("data/processed/scenarios") / study_name / f"seed_{seed}")
        for scenario in scenarios.values():
            for protocol_name in protocol_names:
                summary, rounds_df, assumptions = run_protocol_simulation(
                    scenario=scenario,
                    predictor=predictor,
                    config=config,
                    protocol_name=protocol_name,
                    seed=seed,
                    study_name=study_name,
                )
                per_seed_rows.append(summary)
                round_rows.append(rounds_df)
                assumption_rows.append(assumptions)

    per_seed_df = pd.DataFrame.from_records(per_seed_rows)
    rounds_df = pd.concat(round_rows, ignore_index=True)
    assumptions_df = pd.DataFrame.from_records(assumption_rows)

    group_cols = ["study_name", "scenario", "scenario_label", "scenario_group", "protocol", "protocol_label"]
    aggregated_long, aggregated_wide = aggregate_metric_frame(
        frame=per_seed_df,
        group_cols=group_cols,
        metric_cols=MAIN_METRICS,
        confidence_level=float(config["study"]["confidence_level"]),
    )
    aggregated_wide = aggregated_wide.sort_values(["scenario", "protocol"]).reset_index(drop=True)

    if study_name == "main_comparisons":
        per_seed_path = results_dirs["tables"] / "per_seed_results.csv"
        aggregated_path = results_dirs["tables"] / "aggregated_results.csv"
        summary_path = results_dirs["tables"] / "scenario_protocol_summary.csv"
        rounds_path = results_dirs["logs"] / "per_seed_round_metrics.csv"
        assumptions_path = results_dirs["logs"] / "run_assumptions.csv"
        fairness_path = results_dirs["logs"] / "fairness_report.json"
    else:
        per_seed_path = results_dirs["tables"] / f"{study_name}_results.csv"
        aggregated_path = results_dirs["tables"] / f"{study_name}_aggregated_results.csv"
        summary_path = results_dirs["tables"] / f"{study_name}_summary_table.csv"
        rounds_path = results_dirs["logs"] / f"{study_name}_round_metrics.csv"
        assumptions_path = results_dirs["logs"] / f"{study_name}_run_assumptions.csv"
        fairness_path = results_dirs["logs"] / f"{study_name}_fairness_report.json"

    per_seed_df.to_csv(per_seed_path, index=False)
    aggregated_long.to_csv(aggregated_path, index=False)
    aggregated_wide.to_csv(summary_path, index=False)
    rounds_df.to_csv(rounds_path, index=False)
    assumptions_df.to_csv(assumptions_path, index=False)
    fairness_report = build_fairness_report(assumptions_df, fairness_path)

    return {
        "per_seed": per_seed_df,
        "aggregated_long": aggregated_long,
        "aggregated_wide": aggregated_wide,
        "rounds": rounds_df,
        "assumptions": assumptions_df,
        "fairness_report": fairness_report,
        "fairness_report_path": fairness_path,
    }


def generate_main_study_outputs(results_dirs: dict[str, Path], aggregated_wide: pd.DataFrame, rounds_df: pd.DataFrame) -> pd.DataFrame:
    evaluation_df = aggregated_wide[aggregated_wide["scenario_group"] == "evaluation"].copy()
    plot_alive_nodes(rounds_df, results_dirs["figures"] / "alive_nodes_vs_rounds.png")
    plot_residual_energy(rounds_df, results_dirs["figures"] / "residual_energy_vs_rounds.png")
    plot_summary_bar(
        evaluation_df,
        "packet_delivery_ratio",
        "Packet Delivery Ratio Across Scenarios",
        "Delivered / Raw Packets Generated",
        results_dirs["figures"] / "packet_delivery_ratio_comparison.png",
    )
    plot_summary_bar(
        evaluation_df,
        "end_to_end_delay",
        "End-to-End Delay Across Scenarios",
        "Delay (hops)",
        results_dirs["figures"] / "delay_comparison.png",
    )
    plot_summary_bar(
        evaluation_df,
        "average_aoi",
        "Average AoI Across Scenarios",
        "AoI (rounds)",
        results_dirs["figures"] / "average_aoi_comparison.png",
    )
    plot_summary_bar(
        evaluation_df,
        "hazardous_event_delivery_success_rate",
        "Hazardous Event Delivery Success Across Scenarios",
        "Success rate",
        results_dirs["figures"] / "hazardous_event_success_comparison.png",
    )

    paper_summary_df = build_paper_summary_table(evaluation_df)
    paper_summary_df.to_csv(results_dirs["tables"] / "paper_summary_table.csv", index=False)
    save_markdown_table(paper_summary_df, results_dirs["tables"] / "paper_summary_table.md")
    return paper_summary_df


def generate_ablation_outputs(results_dirs: dict[str, Path], aggregated_wide: pd.DataFrame) -> None:
    plot_ablation_bar(
        aggregated_wide,
        "packet_delivery_ratio",
        "Ablation Comparison: Packet Delivery Ratio",
        "Delivered / Raw Packets Generated",
        results_dirs["figures"] / "ablation_comparison.png",
    )
    aggregated_wide.to_csv(results_dirs["tables"] / "ablation_summary_table.csv", index=False)


def write_reviewer_outputs(
    root: Path,
    fairness_report_path: Path,
    multi_seed_count: int,
    ablation_protocols: list[str],
) -> None:
    write_reviewer_checklist(root / "reviewer_checklist.md", str(fairness_report_path), multi_seed_count, [protocol_label(name) for name in ablation_protocols])
    write_methods_snapshot(root / "methods_snapshot.md")
    write_limitations(root / "limitations.md")


def verify_summary_outputs(summary_frames: list[pd.DataFrame]) -> None:
    for frame in summary_frames:
        if frame.empty:
            raise RuntimeError("A required summary output is empty.")
        if frame.isna().any().any():
            raise RuntimeError("A required summary output contains NaN values.")


def persist_tcn_validation_artifacts(results_dirs: dict[str, Path], metrics_payload: dict, config_payload: dict) -> None:
    save_json(results_dirs["logs"] / "test_metrics.json", metrics_payload)
    save_json(results_dirs["logs"] / "tcn_config_used.json", config_payload)
