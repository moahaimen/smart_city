from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io import save_json


SHARED_ASSUMPTION_KEYS = [
    "node_count",
    "area_size",
    "initial_energy",
    "tx_energy",
    "rx_energy",
    "fs_energy",
    "mp_energy",
    "aggregation_energy",
    "packet_size_bits",
    "control_packet_bits",
    "cluster_head_ratio",
    "radio_range_factor",
    "round_budget",
    "sink_x",
    "sink_y",
    "scenario_name",
    "severity_threshold_normal_max",
    "severity_threshold_warning_max",
    "pdr_denominator",
    "delay_definition",
    "aoi_definition",
    "sensing_schedule",
    "topology_seed",
    "traffic_seed",
    "simulation_seed",
]


def build_fairness_report(run_assumptions_df: pd.DataFrame, output_path: str | Path) -> dict:
    warnings: list[str] = []
    shared_checks: list[dict] = []
    comparison_notes: list[str] = []
    protocol_specific = run_assumptions_df[
        [
            "study_name",
            "scenario",
            "seed",
            "protocol",
            "protocol_label",
            "cluster_head_rule",
            "join_rule",
            "uses_prediction",
            "uses_aoi_term",
            "uses_suppression",
            "uses_priority_scheduler",
            "priority_scoring_enabled",
            "suppression_description",
        ]
    ].to_dict(orient="records")

    grouped = run_assumptions_df.groupby(["study_name", "scenario", "seed"], sort=False)
    for (study_name, scenario_name, seed), group_df in grouped:
        differences = {}
        for key in SHARED_ASSUMPTION_KEYS:
            distinct_values = group_df[key].astype(str).unique().tolist()
            if len(distinct_values) > 1:
                differences[key] = distinct_values
                warnings.append(
                    f"Shared assumption mismatch for study={study_name}, scenario={scenario_name}, seed={seed}, key={key}: {distinct_values}"
                )
        shared_checks.append(
            {
                "study_name": study_name,
                "scenario": scenario_name,
                "seed": int(seed),
                "protocols": group_df["protocol"].tolist(),
                "shared_assumptions_aligned": len(differences) == 0,
                "differences": differences,
            }
        )
        if group_df["pdr_denominator"].nunique() != 1:
            warnings.append(f"Inconsistent PDR denominator for study={study_name}, scenario={scenario_name}, seed={seed}.")
        if group_df["delay_definition"].nunique() != 1:
            warnings.append(f"Inconsistent delay definition for study={study_name}, scenario={scenario_name}, seed={seed}.")
        if group_df["aoi_definition"].nunique() != 1:
            warnings.append(f"Inconsistent AoI definition for study={study_name}, scenario={scenario_name}, seed={seed}.")
        if group_df["uses_suppression"].nunique() > 1:
            comparison_notes.append(
                f"Suppression differs across protocols in study={study_name}, scenario={scenario_name}, seed={seed}; interpret packet generation and delivery jointly."
            )

    report = {
        "protocols": sorted(run_assumptions_df["protocol"].unique().tolist()),
        "shared_assumption_keys": SHARED_ASSUMPTION_KEYS,
        "shared_checks": shared_checks,
        "protocol_specific_differences": protocol_specific,
        "comparison_notes": sorted(set(comparison_notes)),
        "warnings": warnings,
    }
    save_json(output_path, report)
    return report
