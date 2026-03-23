from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str((ROOT / "results" / "logs" / "mplconfig").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str((ROOT / "results" / "logs" / "cache").resolve()))
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.tcn_predictor import PredictorBundle
from src.study import run_multi_seed_study
from src.utils.config import load_config
from src.utils.io import prepare_results_dirs, save_json

FIGURE_DPI = 1000
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "sensitivity_analysis"
MINIMAL_OUTPUT_ROOT = ROOT / "results" / "sensitivity_analysis_minimal"
CHECKPOINT_PATH = ROOT / "results" / "models" / "tcn_regressor.pt"
BASE_CONFIG_PATH = ROOT / "configs" / "default.yaml"
PROPOSED_PROTOCOL = "tcn_predictive_pollution_aware_leach"

PLOT_METRICS = [
    ("fnd", "FND (rounds)"),
    ("packet_delivery_ratio", "Packet delivery ratio"),
    ("average_aoi", "Average AoI (rounds)"),
    ("hazardous_event_delivery_success_rate", "Hazardous success rate"),
    ("raw_suppression_ratio", "Suppressed/generated packets"),
    ("total_throughput_mbits", "Total throughput (Mbits)"),
]

SUMMARY_METRICS = [
    "fnd",
    "hnd",
    "lnd",
    "average_residual_energy_over_time",
    "average_residual_energy_final_round",
    "packet_delivery_ratio",
    "end_to_end_delay",
    "average_aoi",
    "hazardous_event_delivery_success_rate",
    "packets_delivered",
    "hazardous_packets_delivered",
    "raw_suppression_ratio",
    "total_throughput_mbits",
]

FULL_SWEEP_SPECS = [
    {
        "key": "delta_th",
        "symbol": r"$\Delta_{th}$",
        "label": "Suppression change threshold",
        "config_path": ("network", "suppression", "change_threshold"),
        "values": [4.0, 6.0, 8.0, 10.0, 12.0],
        "formatter": lambda value: f"{value:.1f}",
    },
    {
        "key": "p_th",
        "symbol": r"$P_{th}$",
        "label": "Suppression priority threshold",
        "config_path": ("network", "suppression", "priority_threshold"),
        "values": [0.30, 0.36, 0.42, 0.48, 0.54],
        "formatter": lambda value: f"{value:.2f}",
    },
    {
        "key": "a_th",
        "symbol": r"$A_{th}$",
        "label": "Suppression AoI threshold",
        "config_path": ("network", "suppression", "aoi_threshold"),
        "values": [2, 3, 4, 5, 6],
        "formatter": lambda value: f"{int(value)}",
    },
    {
        "key": "rho",
        "symbol": r"$\rho$",
        "label": "Cluster-head ratio",
        "config_path": ("network", "cluster_head_ratio"),
        "values": [0.08, 0.10, 0.12, 0.14, 0.16],
        "formatter": lambda value: f"{value:.2f}",
    },
]


MINIMAL_SWEEP_SPECS = [
    {
        "key": "delta_th",
        "symbol": r"$\Delta_{th}$",
        "label": "Suppression change threshold",
        "config_path": ("network", "suppression", "change_threshold"),
        "values": [6.0, 8.0, 10.0],
        "formatter": lambda value: f"{value:.1f}",
    },
    {
        "key": "p_th",
        "symbol": r"$P_{th}$",
        "label": "Suppression priority threshold",
        "config_path": ("network", "suppression", "priority_threshold"),
        "values": [0.36, 0.42, 0.48],
        "formatter": lambda value: f"{value:.2f}",
    },
    {
        "key": "a_th",
        "symbol": r"$A_{th}$",
        "label": "Suppression AoI threshold",
        "config_path": ("network", "suppression", "aoi_threshold"),
        "values": [3, 4, 5],
        "formatter": lambda value: f"{int(value)}",
    },
    {
        "key": "rho",
        "symbol": r"$\rho$",
        "label": "Cluster-head ratio",
        "config_path": ("network", "cluster_head_ratio"),
        "values": [0.10, 0.12, 0.14],
        "formatter": lambda value: f"{value:.2f}",
    },
]


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "grid.color": "#cfcfcf",
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def set_nested_value(payload: dict, path: tuple[str, ...], value: float | int) -> None:
    cursor = payload
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value


def safe_value_slug(value: float | int) -> str:
    text = str(value)
    return text.replace(".", "p").replace("-", "m")


def summarize_per_seed(per_seed_df: pd.DataFrame) -> dict[str, float]:
    frame = per_seed_df.copy()
    frame["raw_suppression_ratio"] = frame["packets_suppressed"] / frame["packets_generated"].replace(0, np.nan)
    frame["total_throughput_mbits"] = (frame["packets_delivered"] * 4000.0) / 1e6
    frame["raw_suppression_ratio"] = frame["raw_suppression_ratio"].fillna(0.0)

    row: dict[str, float] = {"run_count": float(len(frame))}
    for metric in SUMMARY_METRICS:
        row[f"{metric}_mean"] = float(frame[metric].mean())
        row[f"{metric}_std"] = float(frame[metric].std(ddof=1)) if len(frame) > 1 else 0.0
        row[f"{metric}_ci95_half_width"] = (
            float(1.96 * row[f"{metric}_std"] / np.sqrt(len(frame))) if len(frame) > 1 else 0.0
        )
    return row


def plot_parameter_sweep(summary_df: pd.DataFrame, spec: dict, figures_dir: Path) -> tuple[Path, Path]:
    fig, axes = plt.subplots(3, 2, figsize=(11.0, 10.0))
    axes = axes.ravel()

    plot_df = summary_df[summary_df["parameter_key"] == spec["key"]].sort_values("parameter_value")
    x_values = plot_df["parameter_value"].to_numpy(dtype=float)
    x_labels = [spec["formatter"](value) for value in x_values]

    for axis, (metric, ylabel) in zip(axes, PLOT_METRICS):
        mean_values = plot_df[f"{metric}_mean"].to_numpy(dtype=float)
        ci_values = plot_df[f"{metric}_ci95_half_width"].to_numpy(dtype=float)
        axis.errorbar(
            x_values,
            mean_values,
            yerr=ci_values,
            color="#1f77b4",
            linewidth=2.0,
            marker="o",
            markersize=5.0,
            capsize=3,
        )
        axis.set_title(ylabel)
        axis.set_xlabel(f"{spec['label']} ({spec['symbol']})")
        axis.set_ylabel(ylabel)
        axis.set_xticks(x_values)
        axis.set_xticklabels(x_labels)
        axis.grid(alpha=0.7)
        axis.set_facecolor("white")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.suptitle(f"Sensitivity Analysis for {spec['label']} ({spec['symbol']})", y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    png_path = figures_dir / f"sensitivity_{spec['key']}.png"
    pdf_path = figures_dir / f"sensitivity_{spec['key']}.pdf"
    fig.savefig(png_path, dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-factor-at-a-time parameter sensitivity sweeps.")
    parser.add_argument(
        "--mode",
        choices=("minimal", "full"),
        default="minimal",
        help="Sweep size. 'minimal' uses low/default/high values around the configured defaults.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional output root. Defaults depend on the selected mode.",
    )
    parser.add_argument(
        "--seed-count",
        type=int,
        default=None,
        help="Optional number of configured seeds to use, taken from the start of the configured seed list.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip parameter values whose per-seed results CSV already exists in the output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Missing required checkpoint for sensitivity sweep: {CHECKPOINT_PATH}")

    output_root = args.output_root or (MINIMAL_OUTPUT_ROOT if args.mode == "minimal" else DEFAULT_OUTPUT_ROOT)
    sweep_specs = MINIMAL_SWEEP_SPECS if args.mode == "minimal" else FULL_SWEEP_SPECS

    config = load_config(BASE_CONFIG_PATH)
    results_dirs = prepare_results_dirs(output_root)
    predictor = PredictorBundle.load(CHECKPOINT_PATH)

    summary_rows: list[dict[str, float | str | int | bool]] = []
    run_manifest: list[dict[str, object]] = []

    seeds = [int(seed) for seed in config["study"]["seeds"]]
    if args.seed_count is not None:
        if args.seed_count <= 0:
            raise ValueError("--seed-count must be positive when provided.")
        seeds = seeds[: args.seed_count]
    scenarios = list(config["study"]["main_scenarios"])

    print(
        f"Starting parameter sensitivity sweep: mode={args.mode}, output_root={output_root}, "
        f"scenarios={len(scenarios)}, seeds={len(seeds)}",
        flush=True,
    )

    for spec in sweep_specs:
        default_value = config
        for key in spec["config_path"]:
            default_value = default_value[key]

        for value in spec["values"]:
            variant_config = copy.deepcopy(config)
            set_nested_value(variant_config, spec["config_path"], value)
            study_name = f"sensitivity_{spec['key']}_{safe_value_slug(value)}"
            per_seed_results_path = results_dirs["tables"] / f"{study_name}_results.csv"

            if args.skip_existing and per_seed_results_path.exists():
                print(f"Skipping existing {study_name}", flush=True)
                frame = pd.read_csv(per_seed_results_path)
                summary = summarize_per_seed(frame)
                summary_rows.append(
                    {
                        "parameter_key": spec["key"],
                        "parameter_symbol": spec["symbol"],
                        "parameter_label": spec["label"],
                        "parameter_value": value,
                        "parameter_value_display": spec["formatter"](value),
                        "is_default": bool(value == default_value),
                        "study_name": study_name,
                        "checkpoint_path": str(CHECKPOINT_PATH),
                        "scenario_count": len(scenarios),
                        "seed_count": len(seeds),
                        **summary,
                    }
                )
                run_manifest.append(
                    {
                        "parameter_key": spec["key"],
                        "parameter_label": spec["label"],
                        "parameter_value": value,
                        "study_name": study_name,
                        "per_seed_results_path": str(per_seed_results_path),
                        "aggregated_results_path": str(results_dirs["tables"] / f"{study_name}_aggregated_results.csv"),
                        "summary_table_path": str(results_dirs["tables"] / f"{study_name}_summary_table.csv"),
                        "round_metrics_path": str(results_dirs["logs"] / f"{study_name}_round_metrics.csv"),
                        "assumptions_path": str(results_dirs["logs"] / f"{study_name}_run_assumptions.csv"),
                        "fairness_path": str(results_dirs["logs"] / f"{study_name}_fairness_report.json"),
                    }
                )
                continue

            print(
                f"Starting {study_name}: {spec['label']}={spec['formatter'](value)} "
                f"across {len(scenarios)} scenarios x {len(seeds)} seeds",
                flush=True,
            )

            result = run_multi_seed_study(
                config=variant_config,
                predictor=predictor,
                results_dirs=results_dirs,
                study_name=study_name,
                seeds=seeds,
                scenario_names=scenarios,
                protocol_names=[PROPOSED_PROTOCOL],
            )

            summary = summarize_per_seed(result["per_seed"])
            summary_rows.append(
                {
                    "parameter_key": spec["key"],
                    "parameter_symbol": spec["symbol"],
                    "parameter_label": spec["label"],
                    "parameter_value": value,
                    "parameter_value_display": spec["formatter"](value),
                    "is_default": bool(value == default_value),
                    "study_name": study_name,
                    "checkpoint_path": str(CHECKPOINT_PATH),
                    "scenario_count": len(scenarios),
                    "seed_count": len(seeds),
                    **summary,
                }
            )
            run_manifest.append(
                {
                    "parameter_key": spec["key"],
                    "parameter_label": spec["label"],
                    "parameter_value": value,
                    "study_name": study_name,
                    "per_seed_results_path": str(per_seed_results_path),
                    "aggregated_results_path": str(results_dirs["tables"] / f"{study_name}_aggregated_results.csv"),
                    "summary_table_path": str(results_dirs["tables"] / f"{study_name}_summary_table.csv"),
                    "round_metrics_path": str(results_dirs["logs"] / f"{study_name}_round_metrics.csv"),
                    "assumptions_path": str(results_dirs["logs"] / f"{study_name}_run_assumptions.csv"),
                    "fairness_path": str(results_dirs["logs"] / f"{study_name}_fairness_report.json"),
                }
            )
            print(f"Completed {study_name}", flush=True)

    summary_df = pd.DataFrame.from_records(summary_rows).sort_values(["parameter_key", "parameter_value"]).reset_index(drop=True)
    manifest_df = pd.DataFrame.from_records(run_manifest).sort_values(["parameter_key", "parameter_value"]).reset_index(drop=True)

    summary_csv = results_dirs["tables"] / "parameter_sensitivity_summary.csv"
    manifest_csv = results_dirs["tables"] / "parameter_sensitivity_manifest.csv"
    summary_md = results_dirs["tables"] / "parameter_sensitivity_summary.md"
    summary_df.to_csv(summary_csv, index=False)
    manifest_df.to_csv(manifest_csv, index=False)
    summary_md.write_text(summary_df.to_markdown(index=False) + "\n", encoding="utf-8")

    plot_rows: list[dict[str, str]] = []
    for spec in sweep_specs:
        png_path, pdf_path = plot_parameter_sweep(summary_df, spec, results_dirs["figures"])
        plot_rows.append(
            {
                "parameter_key": spec["key"],
                "parameter_label": spec["label"],
                "output_png": str(png_path),
                "output_pdf": str(pdf_path),
                "dpi": str(FIGURE_DPI),
                "source_summary_csv": str(summary_csv),
            }
        )

    plots_csv = results_dirs["tables"] / "parameter_sensitivity_plots.csv"
    pd.DataFrame.from_records(plot_rows).to_csv(plots_csv, index=False)

    metadata = {
        "base_config": str(BASE_CONFIG_PATH),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "mode": args.mode,
        "output_root": str(output_root),
        "protocol": PROPOSED_PROTOCOL,
        "scenarios": scenarios,
        "seeds": seeds,
        "sweep_specs": [
            {
                "parameter_key": spec["key"],
                "parameter_label": spec["label"],
                "parameter_symbol": spec["symbol"],
                "config_path": ".".join(spec["config_path"]),
                "values": spec["values"],
            }
            for spec in sweep_specs
        ],
        "summary_csv": str(summary_csv),
        "manifest_csv": str(manifest_csv),
        "plots_csv": str(plots_csv),
    }
    save_json(results_dirs["logs"] / "parameter_sensitivity_metadata.json", metadata)

    print(f"Summary CSV: {summary_csv}", flush=True)
    print(f"Manifest CSV: {manifest_csv}", flush=True)
    print(f"Plots CSV: {plots_csv}", flush=True)
    print(f"Summary Markdown: {summary_md}", flush=True)


if __name__ == "__main__":
    main()
