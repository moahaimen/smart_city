from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str((ROOT / "results" / "logs" / "mplconfig").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str((ROOT / "results" / "logs" / "cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(ROOT))

from src.utils.naming import protocol_label, scenario_label

FIGURE_DPI = 1000
OUTPUT_DIR = ROOT / "results" / "paper_figures_section5"
PROVENANCE_CSV = OUTPUT_DIR / "figure_sources.csv"

MAIN_PROTOCOLS = [
    "standard_leach",
    "energy_aware_leach",
    "tcn_predictive_pollution_aware_leach",
]
MAIN_SCENARIOS = [
    "normal",
    "rising_warning",
    "hazardous_spike",
    "hotspot_heavy",
]
ABLATION_PROTOCOLS = [
    "full_tcn_ppa_leach",
    "no_tcn_prediction",
    "no_aoi_term",
    "no_suppression",
    "no_priority_scheduler",
]

MAIN_COLORS = {
    "standard_leach": "#1f77b4",
    "energy_aware_leach": "#ff7f0e",
    "tcn_predictive_pollution_aware_leach": "#2ca02c",
}
ABLATION_COLORS = {
    "full_tcn_ppa_leach": "#2ca02c",
    "no_tcn_prediction": "#4c78a8",
    "no_aoi_term": "#f58518",
    "no_suppression": "#e45756",
    "no_priority_scheduler": "#72b7b2",
}


@dataclass
class FigureSpec:
    figure_id: str
    file_stem: str
    title: str
    plot_type: str
    x_axis: str
    y_axis: str
    compared_methods: str
    source_files: str
    source_columns: str
    derived_columns: str
    uncertainty_type: str
    purpose: str
    caption_suggestion: str
    generator: Callable[[], None]


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "axes.grid": False,
            "grid.color": "#cfcfcf",
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titlepad": 10,
            "legend.borderaxespad": 0.3,
        }
    )


def ensure_columns(frame: pd.DataFrame, required: list[str], frame_name: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"{frame_name} is missing columns: {missing}")


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required source file: {path}")
    return pd.read_csv(path)


def style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor("white")
    axis.grid(axis="y", alpha=0.45)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#4d4d4d")
    axis.spines["bottom"].set_color("#4d4d4d")
    axis.tick_params(axis="both", length=3.5, color="#4d4d4d")


def save_figure_named(fig: plt.Figure, file_stem: str) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / f"{file_stem}.png"
    pdf_path = OUTPUT_DIR / f"{file_stem}.pdf"
    fig.savefig(png_path, dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def save_figure(fig: plt.Figure, figure_number: int) -> tuple[Path, Path]:
    return save_figure_named(fig, f"figure{figure_number}")


def compute_round_summary(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    grouped = (
        frame.groupby(["scenario", "protocol", "round"], sort=False)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["std"] = grouped["std"].fillna(0.0)
    grouped["ci95_half_width"] = np.where(
        grouped["count"] > 1,
        1.96 * grouped["std"] / np.sqrt(grouped["count"]),
        0.0,
    )
    return grouped


def compute_seed_summary(frame: pd.DataFrame, group_cols: list[str], metric: str) -> pd.DataFrame:
    grouped = frame.groupby(group_cols, sort=False)[metric].agg(["mean", "std", "count"]).reset_index()
    grouped["std"] = grouped["std"].fillna(0.0)
    grouped["ci95_half_width"] = np.where(
        grouped["count"] > 1,
        1.96 * grouped["std"] / np.sqrt(grouped["count"]),
        0.0,
    )
    return grouped


def line_grid_from_rounds(
    frame: pd.DataFrame,
    metric: str,
    ylabel: str,
    figure_number: int,
    title: str,
    colors: dict[str, str],
) -> tuple[Path, Path]:
    summary = compute_round_summary(frame, metric)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2), sharex=False, sharey=False)
    axes = axes.ravel()

    for axis, scenario in zip(axes, MAIN_SCENARIOS):
        scenario_rows = summary[summary["scenario"] == scenario]
        for protocol in MAIN_PROTOCOLS:
            protocol_rows = scenario_rows[scenario_rows["protocol"] == protocol].sort_values("round")
            if protocol_rows.empty:
                continue
            x_values = protocol_rows["round"].to_numpy(dtype=int)
            mean_values = protocol_rows["mean"].to_numpy(dtype=float)
            ci_values = protocol_rows["ci95_half_width"].to_numpy(dtype=float)
            axis.plot(
                x_values,
                mean_values,
                color=colors[protocol],
                linewidth=2.2,
                label=protocol_label(protocol),
            )
            axis.fill_between(
                x_values,
                mean_values - ci_values,
                mean_values + ci_values,
                color=colors[protocol],
                alpha=0.12,
                linewidth=0.0,
            )
        axis.set_title(scenario_label(scenario))
        axis.set_xlabel("Round")
        axis.set_ylabel(ylabel)
        axis.margins(x=0.02)
        style_axis(axis)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(title, y=0.995)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.955),
        columnspacing=1.4,
        handlelength=2.2,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    return save_figure(fig, figure_number)


def bar_from_summary(
    frame: pd.DataFrame,
    metric_mean_col: str,
    metric_ci_col: str,
    ylabel: str,
    figure_number: int,
    title: str,
    protocols: list[str],
    scenarios: list[str],
    colors: dict[str, str],
    output_stem: str | None = None,
    show_legend: bool = True,
) -> tuple[Path, Path]:
    fig, axis = plt.subplots(figsize=(10.8, 5.0))
    scenario_spacing = 1.35
    positions = np.arange(len(scenarios), dtype=float) * scenario_spacing
    protocol_count = max(len(protocols), 1)
    group_span = 0.76 if protocol_count <= 3 else 0.98
    bar_pitch = group_span / protocol_count
    width = bar_pitch * 0.70

    for offset, protocol in enumerate(protocols):
        protocol_rows = frame[frame["protocol"] == protocol].set_index("scenario").reindex(scenarios)
        if protocol_rows[metric_mean_col].isna().any():
            raise ValueError(f"Missing summary rows for protocol={protocol} and metric={metric_mean_col}")
        shift = offset - (len(protocols) - 1) / 2.0
        x_positions = positions + shift * bar_pitch
        axis.bar(
            x_positions,
            protocol_rows[metric_mean_col].to_numpy(dtype=float),
            width=width,
            yerr=protocol_rows[metric_ci_col].to_numpy(dtype=float),
            capsize=3,
            color=colors[protocol],
            edgecolor="#333333",
            linewidth=0.6,
            label=protocol_label(protocol),
        )

    axis.set_xticks(positions)
    axis.set_xticklabels([scenario_label(scenario) for scenario in scenarios], rotation=15)
    if len(positions) > 0:
        axis.set_xlim(positions[0] - group_span * 0.95, positions[-1] + group_span * 0.95)
    axis.set_ylabel(ylabel)
    axis.set_xlabel("Scenario")
    axis.set_title(title)
    style_axis(axis)
    axis.grid(axis="y", alpha=0.7)
    if show_legend:
        handles, labels = axis.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(3, len(protocols)),
            frameon=False,
            bbox_to_anchor=(0.5, 0.99),
            columnspacing=1.4,
            handlelength=1.8,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    else:
        fig.tight_layout()
    if output_stem is not None:
        return save_figure_named(fig, output_stem)
    return save_figure(fig, figure_number)


def plot_figure2(rounds_df: pd.DataFrame) -> tuple[Path, Path]:
    required = ["study_name", "scenario", "protocol", "round", "alive_nodes"]
    ensure_columns(rounds_df, required, "per_seed_round_metrics.csv")
    main_df = rounds_df[
        (rounds_df["study_name"] == "main_comparisons")
        & (rounds_df["scenario"].isin(MAIN_SCENARIOS))
        & (rounds_df["protocol"].isin(MAIN_PROTOCOLS))
    ].copy()
    return line_grid_from_rounds(
        frame=main_df,
        metric="alive_nodes",
        ylabel="Alive nodes (count)",
        figure_number=2,
        title="Alive Nodes Versus Rounds Across Main Evaluation Scenarios",
        colors=MAIN_COLORS,
    )


def plot_figure3(rounds_df: pd.DataFrame) -> tuple[Path, Path]:
    required = ["study_name", "scenario", "protocol", "round", "avg_residual_energy"]
    ensure_columns(rounds_df, required, "per_seed_round_metrics.csv")
    main_df = rounds_df[
        (rounds_df["study_name"] == "main_comparisons")
        & (rounds_df["scenario"].isin(MAIN_SCENARIOS))
        & (rounds_df["protocol"].isin(MAIN_PROTOCOLS))
    ].copy()
    return line_grid_from_rounds(
        frame=main_df,
        metric="avg_residual_energy",
        ylabel="Average residual energy (J)",
        figure_number=3,
        title="Average Residual Energy Versus Rounds",
        colors=MAIN_COLORS,
    )


def plot_figure4(rounds_df: pd.DataFrame) -> tuple[Path, Path]:
    required = ["study_name", "scenario", "protocol", "round", "cluster_heads"]
    ensure_columns(rounds_df, required, "per_seed_round_metrics.csv")
    main_df = rounds_df[
        (rounds_df["study_name"] == "main_comparisons")
        & (rounds_df["scenario"].isin(MAIN_SCENARIOS))
        & (rounds_df["protocol"].isin(MAIN_PROTOCOLS))
    ].copy()
    return line_grid_from_rounds(
        frame=main_df,
        metric="cluster_heads",
        ylabel="Active cluster heads (count)",
        figure_number=4,
        title="Active Cluster Heads Versus Rounds",
        colors=MAIN_COLORS,
    )


def plot_figure5(rounds_df: pd.DataFrame) -> tuple[Path, Path]:
    required = ["study_name", "scenario", "protocol", "seed", "round", "throughput_bits_round"]
    ensure_columns(rounds_df, required, "per_seed_round_metrics.csv")
    main_df = rounds_df[
        (rounds_df["study_name"] == "main_comparisons")
        & (rounds_df["scenario"].isin(MAIN_SCENARIOS))
        & (rounds_df["protocol"].isin(MAIN_PROTOCOLS))
    ].copy()
    main_df["cumulative_throughput_bits"] = main_df.groupby(
        ["scenario", "protocol", "seed"], sort=False
    )["throughput_bits_round"].cumsum()
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2), sharex=False, sharey=False)
    axes = axes.ravel()
    summary = compute_round_summary(main_df, "cumulative_throughput_bits")

    for axis, scenario in zip(axes, MAIN_SCENARIOS):
        scenario_rows = summary[summary["scenario"] == scenario]
        for protocol in MAIN_PROTOCOLS:
            protocol_rows = scenario_rows[scenario_rows["protocol"] == protocol].sort_values("round")
            if protocol_rows.empty:
                continue
            x_values = protocol_rows["round"].to_numpy(dtype=int)
            mean_values = protocol_rows["mean"].to_numpy(dtype=float) / 1e6
            ci_values = protocol_rows["ci95_half_width"].to_numpy(dtype=float) / 1e6
            axis.plot(
                x_values,
                mean_values,
                color=MAIN_COLORS[protocol],
                linewidth=2.2,
                label=protocol_label(protocol),
            )
            axis.fill_between(
                x_values,
                mean_values - ci_values,
                mean_values + ci_values,
                color=MAIN_COLORS[protocol],
                alpha=0.12,
                linewidth=0.0,
            )
        axis.set_title(scenario_label(scenario))
        axis.set_xlabel("Round")
        axis.set_ylabel("Cumulative throughput (Mbits)")
        axis.margins(x=0.02)
        style_axis(axis)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Cumulative Throughput Versus Rounds", y=0.995)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.955),
        columnspacing=1.4,
        handlelength=2.2,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    return save_figure(fig, 5)


def plot_figure6(summary_df: pd.DataFrame) -> tuple[Path, Path]:
    required = ["scenario", "protocol", "packet_delivery_ratio_mean", "packet_delivery_ratio_ci95_half_width"]
    ensure_columns(required=required, frame=summary_df, frame_name="scenario_protocol_summary.csv")
    main_df = summary_df[
        (summary_df["study_name"] == "main_comparisons")
        & (summary_df["scenario"].isin(MAIN_SCENARIOS))
        & (summary_df["protocol"].isin(MAIN_PROTOCOLS))
    ].copy()
    return bar_from_summary(
        frame=main_df,
        metric_mean_col="packet_delivery_ratio_mean",
        metric_ci_col="packet_delivery_ratio_ci95_half_width",
        ylabel="Packet delivery ratio",
        figure_number=6,
        title="Packet Delivery Ratio Across Main Evaluation Scenarios",
        protocols=MAIN_PROTOCOLS,
        scenarios=MAIN_SCENARIOS,
        colors=MAIN_COLORS,
    )


def plot_figure7(summary_df: pd.DataFrame) -> tuple[Path, Path]:
    required = ["scenario", "protocol", "end_to_end_delay_mean", "end_to_end_delay_ci95_half_width"]
    ensure_columns(required=required, frame=summary_df, frame_name="scenario_protocol_summary.csv")
    main_df = summary_df[
        (summary_df["study_name"] == "main_comparisons")
        & (summary_df["scenario"].isin(MAIN_SCENARIOS))
        & (summary_df["protocol"].isin(MAIN_PROTOCOLS))
    ].copy()
    return bar_from_summary(
        frame=main_df,
        metric_mean_col="end_to_end_delay_mean",
        metric_ci_col="end_to_end_delay_ci95_half_width",
        ylabel="Average path length (hops)",
        figure_number=7,
        title="Average Path Length Across Main Evaluation Scenarios",
        protocols=MAIN_PROTOCOLS,
        scenarios=MAIN_SCENARIOS,
        colors=MAIN_COLORS,
    )


def plot_figure8(summary_df: pd.DataFrame) -> tuple[Path, Path]:
    required = ["scenario", "protocol", "average_aoi_mean", "average_aoi_ci95_half_width"]
    ensure_columns(required=required, frame=summary_df, frame_name="scenario_protocol_summary.csv")
    main_df = summary_df[
        (summary_df["study_name"] == "main_comparisons")
        & (summary_df["scenario"].isin(MAIN_SCENARIOS))
        & (summary_df["protocol"].isin(MAIN_PROTOCOLS))
    ].copy()
    return bar_from_summary(
        frame=main_df,
        metric_mean_col="average_aoi_mean",
        metric_ci_col="average_aoi_ci95_half_width",
        ylabel="Average AoI (rounds)",
        figure_number=8,
        title="Average Age of Information Across Main Evaluation Scenarios",
        protocols=MAIN_PROTOCOLS,
        scenarios=MAIN_SCENARIOS,
        colors=MAIN_COLORS,
    )


def plot_figure9(summary_df: pd.DataFrame) -> tuple[Path, Path]:
    required = [
        "scenario",
        "protocol",
        "hazardous_event_delivery_success_rate_mean",
        "hazardous_event_delivery_success_rate_ci95_half_width",
    ]
    ensure_columns(required=required, frame=summary_df, frame_name="scenario_protocol_summary.csv")
    main_df = summary_df[
        (summary_df["study_name"] == "main_comparisons")
        & (summary_df["scenario"].isin(MAIN_SCENARIOS))
        & (summary_df["protocol"].isin(MAIN_PROTOCOLS))
    ].copy()
    return bar_from_summary(
        frame=main_df,
        metric_mean_col="hazardous_event_delivery_success_rate_mean",
        metric_ci_col="hazardous_event_delivery_success_rate_ci95_half_width",
        ylabel="Hazardous-event delivery success rate",
        figure_number=9,
        title="Hazardous-Event Delivery Success Across Main Evaluation Scenarios",
        protocols=MAIN_PROTOCOLS,
        scenarios=MAIN_SCENARIOS,
        colors=MAIN_COLORS,
    )


def plot_figure10(per_seed_df: pd.DataFrame) -> tuple[Path, Path]:
    required = ["study_name", "scenario", "protocol", "packets_suppressed", "packets_generated"]
    ensure_columns(per_seed_df, required, "per_seed_results.csv")
    main_df = per_seed_df[
        (per_seed_df["study_name"] == "main_comparisons")
        & (per_seed_df["scenario"].isin(MAIN_SCENARIOS))
        & (per_seed_df["protocol"].isin(MAIN_PROTOCOLS))
    ].copy()
    generated = main_df["packets_generated"].replace(0, np.nan)
    main_df["suppression_ratio_raw"] = main_df["packets_suppressed"] / generated
    summary = compute_seed_summary(main_df, ["scenario", "protocol"], "suppression_ratio_raw")
    return bar_from_summary(
        frame=summary,
        metric_mean_col="mean",
        metric_ci_col="ci95_half_width",
        ylabel="Suppressed-packet fraction",
        figure_number=10,
        title="Suppressed-Packet Fraction Across Main Evaluation Scenarios",
        protocols=MAIN_PROTOCOLS,
        scenarios=MAIN_SCENARIOS,
        colors=MAIN_COLORS,
    )


def plot_figure10_tcn_only(per_seed_df: pd.DataFrame) -> tuple[Path, Path]:
    required = ["study_name", "scenario", "protocol", "packets_suppressed", "packets_generated"]
    ensure_columns(per_seed_df, required, "per_seed_results.csv")
    main_df = per_seed_df[
        (per_seed_df["study_name"] == "main_comparisons")
        & (per_seed_df["scenario"].isin(MAIN_SCENARIOS))
        & (per_seed_df["protocol"] == "tcn_predictive_pollution_aware_leach")
    ].copy()
    generated = main_df["packets_generated"].replace(0, np.nan)
    main_df["suppression_ratio_raw"] = main_df["packets_suppressed"] / generated
    summary = compute_seed_summary(main_df, ["scenario", "protocol"], "suppression_ratio_raw")
    return bar_from_summary(
        frame=summary,
        metric_mean_col="mean",
        metric_ci_col="ci95_half_width",
        ylabel="Suppressed-packet fraction",
        figure_number=10,
        title="Suppressed-Packet Fraction Across Main Evaluation Scenarios (TCN-PPA-LEACH Only)",
        protocols=["tcn_predictive_pollution_aware_leach"],
        scenarios=MAIN_SCENARIOS,
        colors=MAIN_COLORS,
        output_stem="figure10_tcn_only",
        show_legend=False,
    )


def plot_figure11(ablation_df: pd.DataFrame) -> tuple[Path, Path]:
    required = ["scenario", "protocol", "packet_delivery_ratio_mean", "packet_delivery_ratio_ci95_half_width"]
    ensure_columns(required=required, frame=ablation_df, frame_name="ablation_summary_table.csv")
    main_df = ablation_df[
        ablation_df["scenario"].isin(["rising_warning", "hazardous_spike", "hotspot_heavy"])
        & ablation_df["protocol"].isin(ABLATION_PROTOCOLS)
    ].copy()
    scenarios = ["rising_warning", "hazardous_spike", "hotspot_heavy"]
    return bar_from_summary(
        frame=main_df,
        metric_mean_col="packet_delivery_ratio_mean",
        metric_ci_col="packet_delivery_ratio_ci95_half_width",
        ylabel="Packet delivery ratio",
        figure_number=11,
        title="Ablation Study: Packet Delivery Ratio Across Stress Scenarios",
        protocols=ABLATION_PROTOCOLS,
        scenarios=scenarios,
        colors=ABLATION_COLORS,
    )


def plot_figure12(ablation_df: pd.DataFrame) -> tuple[Path, Path]:
    required = ["scenario", "protocol", "average_aoi_mean", "average_aoi_ci95_half_width"]
    ensure_columns(required=required, frame=ablation_df, frame_name="ablation_summary_table.csv")
    main_df = ablation_df[
        ablation_df["scenario"].isin(["rising_warning", "hazardous_spike", "hotspot_heavy"])
        & ablation_df["protocol"].isin(ABLATION_PROTOCOLS)
    ].copy()
    scenarios = ["rising_warning", "hazardous_spike", "hotspot_heavy"]
    return bar_from_summary(
        frame=main_df,
        metric_mean_col="average_aoi_mean",
        metric_ci_col="average_aoi_ci95_half_width",
        ylabel="Average AoI (rounds)",
        figure_number=12,
        title="Ablation Study: Average Age of Information Across Stress Scenarios",
        protocols=ABLATION_PROTOCOLS,
        scenarios=scenarios,
        colors=ABLATION_COLORS,
    )


def build_specs(
    rounds_df: pd.DataFrame,
    per_seed_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
) -> list[FigureSpec]:
    return [
        FigureSpec(
            figure_id="Figure 2",
            file_stem="figure2",
            title="Alive Nodes Versus Rounds Across Main Evaluation Scenarios",
            plot_type="line plot (2x2 scenario grid)",
            x_axis="Round",
            y_axis="Alive nodes (count)",
            compared_methods="LEACH; EA-LEACH; TCN-PPA-LEACH",
            source_files="results/logs/per_seed_round_metrics.csv",
            source_columns="study_name, scenario, protocol, seed, round, alive_nodes",
            derived_columns="Per-round mean and 95% CI over available seeds",
            uncertainty_type="Shaded band: mean ± 1.96*SD/sqrt(n) over available seeds at each round",
            purpose="Show network lifetime progression under the four main scenarios.",
            caption_suggestion="Alive-node trajectories over rounds for the three compared protocols under the four main evaluation scenarios.",
            generator=lambda: plot_figure2(rounds_df),
        ),
        FigureSpec(
            figure_id="Figure 3",
            file_stem="figure3",
            title="Average Residual Energy Versus Rounds",
            plot_type="line plot (2x2 scenario grid)",
            x_axis="Round",
            y_axis="Average residual energy (J)",
            compared_methods="LEACH; EA-LEACH; TCN-PPA-LEACH",
            source_files="results/logs/per_seed_round_metrics.csv",
            source_columns="study_name, scenario, protocol, seed, round, avg_residual_energy",
            derived_columns="Per-round mean and 95% CI over available seeds",
            uncertainty_type="Shaded band: mean ± 1.96*SD/sqrt(n) over available seeds at each round",
            purpose="Show depletion rate and residual energy preservation over time.",
            caption_suggestion="Average residual energy over simulation rounds for the three compared protocols.",
            generator=lambda: plot_figure3(rounds_df),
        ),
        FigureSpec(
            figure_id="Figure 4",
            file_stem="figure4",
            title="Active Cluster Heads Versus Rounds",
            plot_type="line plot (2x2 scenario grid)",
            x_axis="Round",
            y_axis="Active cluster heads (count)",
            compared_methods="LEACH; EA-LEACH; TCN-PPA-LEACH",
            source_files="results/logs/per_seed_round_metrics.csv",
            source_columns="study_name, scenario, protocol, seed, round, cluster_heads",
            derived_columns="Per-round mean and 95% CI over available seeds",
            uncertainty_type="Shaded band: mean ± 1.96*SD/sqrt(n) over available seeds at each round",
            purpose="Show the only stored CH-burden proxy in the simulator outputs.",
            caption_suggestion="Mean number of active cluster heads over rounds across the four main scenarios.",
            generator=lambda: plot_figure4(rounds_df),
        ),
        FigureSpec(
            figure_id="Figure 5",
            file_stem="figure5",
            title="Cumulative Throughput Versus Rounds",
            plot_type="line plot (2x2 scenario grid)",
            x_axis="Round",
            y_axis="Cumulative throughput (Mbits)",
            compared_methods="LEACH; EA-LEACH; TCN-PPA-LEACH",
            source_files="results/logs/per_seed_round_metrics.csv",
            source_columns="study_name, scenario, protocol, seed, round, throughput_bits_round",
            derived_columns="Per-seed cumulative sum of throughput_bits_round; per-round mean and 95% CI over available seeds; plotted in Mbits by dividing mean and CI by 1e6",
            uncertainty_type="Shaded band: mean ± 1.96*SD/sqrt(n) over available seeds at each round",
            purpose="Show sink-side data accumulation over time.",
            caption_suggestion="Cumulative throughput accumulated at the base station over rounds for each compared protocol.",
            generator=lambda: plot_figure5(rounds_df),
        ),
        FigureSpec(
            figure_id="Figure 6",
            file_stem="figure6",
            title="Packet Delivery Ratio Across Main Evaluation Scenarios",
            plot_type="grouped bar chart",
            x_axis="Scenario",
            y_axis="Packet delivery ratio",
            compared_methods="LEACH; EA-LEACH; TCN-PPA-LEACH",
            source_files="results/tables/scenario_protocol_summary.csv",
            source_columns="scenario, protocol, packet_delivery_ratio_mean, packet_delivery_ratio_ci95_half_width",
            derived_columns="None",
            uncertainty_type="Error bar: stored 95% CI half-width from scenario_protocol_summary.csv",
            purpose="Compare delivery efficiency under each scenario.",
            caption_suggestion="Packet delivery ratio with 95% confidence intervals across the four main evaluation scenarios.",
            generator=lambda: plot_figure6(summary_df),
        ),
        FigureSpec(
            figure_id="Figure 7",
            file_stem="figure7",
            title="Average Path Length Across Main Evaluation Scenarios",
            plot_type="grouped bar chart",
            x_axis="Scenario",
            y_axis="Average path length (hops)",
            compared_methods="LEACH; EA-LEACH; TCN-PPA-LEACH",
            source_files="results/tables/scenario_protocol_summary.csv",
            source_columns="scenario, protocol, end_to_end_delay_mean, end_to_end_delay_ci95_half_width",
            derived_columns="None",
            uncertainty_type="Error bar: stored 95% CI half-width from scenario_protocol_summary.csv",
            purpose="Compare route-delay cost using the logged hop-count delay definition.",
            caption_suggestion="Average path length, measured in hops to sink, across the four main evaluation scenarios.",
            generator=lambda: plot_figure7(summary_df),
        ),
        FigureSpec(
            figure_id="Figure 8",
            file_stem="figure8",
            title="Average Age of Information Across Main Evaluation Scenarios",
            plot_type="grouped bar chart",
            x_axis="Scenario",
            y_axis="Average AoI (rounds)",
            compared_methods="LEACH; EA-LEACH; TCN-PPA-LEACH",
            source_files="results/tables/scenario_protocol_summary.csv",
            source_columns="scenario, protocol, average_aoi_mean, average_aoi_ci95_half_width",
            derived_columns="None",
            uncertainty_type="Error bar: stored 95% CI half-width from scenario_protocol_summary.csv",
            purpose="Compare information freshness at the base station.",
            caption_suggestion="Average Age of Information across the four main evaluation scenarios.",
            generator=lambda: plot_figure8(summary_df),
        ),
        FigureSpec(
            figure_id="Figure 9",
            file_stem="figure9",
            title="Hazardous-Event Delivery Success Across Main Evaluation Scenarios",
            plot_type="grouped bar chart",
            x_axis="Scenario",
            y_axis="Hazardous-event delivery success rate",
            compared_methods="LEACH; EA-LEACH; TCN-PPA-LEACH",
            source_files="results/tables/scenario_protocol_summary.csv",
            source_columns="scenario, protocol, hazardous_event_delivery_success_rate_mean, hazardous_event_delivery_success_rate_ci95_half_width",
            derived_columns="None",
            uncertainty_type="Error bar: stored 95% CI half-width from scenario_protocol_summary.csv",
            purpose="Compare urgent-traffic reliability under hazardous conditions.",
            caption_suggestion="Hazardous-event delivery success rate with 95% confidence intervals across the four main evaluation scenarios.",
            generator=lambda: plot_figure9(summary_df),
        ),
        FigureSpec(
            figure_id="Figure 10",
            file_stem="figure10",
            title="Suppressed-Packet Fraction Across Main Evaluation Scenarios",
            plot_type="grouped bar chart",
            x_axis="Scenario",
            y_axis="Suppressed-packet fraction",
            compared_methods="LEACH; EA-LEACH; TCN-PPA-LEACH",
            source_files="results/tables/per_seed_results.csv",
            source_columns="study_name, scenario, protocol, seed, packets_suppressed, packets_generated",
            derived_columns="suppression_ratio_raw = packets_suppressed / packets_generated; per-scenario mean and 95% CI over seeds",
            uncertainty_type="Error bar: mean ± 1.96*SD/sqrt(n) over per-seed raw suppression ratio",
            purpose="Quantify the actually logged suppression behavior.",
            caption_suggestion="Suppressed-packet fraction, computed as suppressed packets divided by raw generated packets, across the four main scenarios.",
            generator=lambda: plot_figure10(per_seed_df),
        ),
        FigureSpec(
            figure_id="Figure 11",
            file_stem="figure11",
            title="Ablation Study: Packet Delivery Ratio Across Stress Scenarios",
            plot_type="grouped bar chart",
            x_axis="Scenario",
            y_axis="Packet delivery ratio",
            compared_methods="TCN-PPA-LEACH; No Prediction; No AoI; No Suppression; No Priority Scheduler",
            source_files="results/tables/ablation_summary_table.csv",
            source_columns="scenario, protocol, packet_delivery_ratio_mean, packet_delivery_ratio_ci95_half_width",
            derived_columns="None",
            uncertainty_type="Error bar: stored 95% CI half-width from ablation_summary_table.csv",
            purpose="Show how the ablated design variants affect delivery efficiency.",
            caption_suggestion="Ablation comparison of packet delivery ratio across the three stress scenarios.",
            generator=lambda: plot_figure11(ablation_df),
        ),
        FigureSpec(
            figure_id="Figure 12",
            file_stem="figure12",
            title="Ablation Study: Average Age of Information Across Stress Scenarios",
            plot_type="grouped bar chart",
            x_axis="Scenario",
            y_axis="Average AoI (rounds)",
            compared_methods="TCN-PPA-LEACH; No Prediction; No AoI; No Suppression; No Priority Scheduler",
            source_files="results/tables/ablation_summary_table.csv",
            source_columns="scenario, protocol, average_aoi_mean, average_aoi_ci95_half_width",
            derived_columns="None",
            uncertainty_type="Error bar: stored 95% CI half-width from ablation_summary_table.csv",
            purpose="Show how the ablated design variants affect information freshness.",
            caption_suggestion="Ablation comparison of Average Age of Information across the three stress scenarios.",
            generator=lambda: plot_figure12(ablation_df),
        ),
    ]


def main() -> None:
    configure_matplotlib()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rounds_df = load_csv(ROOT / "results" / "logs" / "per_seed_round_metrics.csv")
    per_seed_df = load_csv(ROOT / "results" / "tables" / "per_seed_results.csv")
    summary_df = load_csv(ROOT / "results" / "tables" / "scenario_protocol_summary.csv")
    ablation_df = load_csv(ROOT / "results" / "tables" / "ablation_summary_table.csv")

    provenance_rows: list[dict[str, str]] = []
    all_specs = build_specs(rounds_df, per_seed_df, summary_df, ablation_df)
    extra_specs = [
        FigureSpec(
            figure_id="Figure 10 (TCN only)",
            file_stem="figure10_tcn_only",
            title="Suppressed-Packet Fraction Across Main Evaluation Scenarios (TCN-PPA-LEACH Only)",
            plot_type="grouped bar chart",
            x_axis="Scenario",
            y_axis="Suppressed-packet fraction",
            compared_methods="TCN-PPA-LEACH",
            source_files="results/tables/per_seed_results.csv",
            source_columns="study_name, scenario, protocol, seed, packets_suppressed, packets_generated",
            derived_columns="suppression_ratio_raw = packets_suppressed / packets_generated; per-scenario mean and 95% CI over seeds",
            uncertainty_type="Error bar: mean ± 1.96*SD/sqrt(n) over per-seed raw suppression ratio",
            purpose="Provide a cleaner optional suppression-only view for the proposed method.",
            caption_suggestion="Suppressed-packet fraction for TCN-PPA-LEACH alone across the four main evaluation scenarios.",
            generator=lambda: plot_figure10_tcn_only(per_seed_df),
        )
    ]

    for spec in [*all_specs, *extra_specs]:
        try:
            png_path, pdf_path = spec.generator()
            status = "generated"
            reason = ""
        except (FileNotFoundError, KeyError, ValueError) as exc:
            png_path = OUTPUT_DIR / f"{spec.file_stem}.png"
            pdf_path = OUTPUT_DIR / f"{spec.file_stem}.pdf"
            status = "skipped"
            reason = str(exc)

        provenance_rows.append(
            {
                "figure_id": spec.figure_id,
                "file_stem": spec.file_stem,
                "status": status,
                "skip_reason": reason,
                "title": spec.title,
                "plot_type": spec.plot_type,
                "x_axis": spec.x_axis,
                "y_axis": spec.y_axis,
                "axis_labels_used": f"x={spec.x_axis}; y={spec.y_axis}",
                "compared_methods": spec.compared_methods,
                "plotted_methods": spec.compared_methods,
                "uncertainty_type": spec.uncertainty_type,
                "scientific_purpose": spec.purpose,
                "caption_suggestion": spec.caption_suggestion,
                "source_files": spec.source_files,
                "source_columns": spec.source_columns,
                "derived_columns": spec.derived_columns,
                "output_png": str(png_path),
                "output_pdf": str(pdf_path),
                "output_paths": f"{png_path}; {pdf_path}",
                "dpi_png": str(FIGURE_DPI),
            }
        )

    provenance_df = pd.DataFrame.from_records(provenance_rows)
    provenance_df.to_csv(PROVENANCE_CSV, index=False)

    generated = provenance_df[provenance_df["status"] == "generated"]
    skipped = provenance_df[provenance_df["status"] == "skipped"]

    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Provenance CSV: {PROVENANCE_CSV}")
    print("Generated figures:")
    for row in generated.itertuples(index=False):
        print(f"- {row.figure_id}: {row.output_png}")
        print(f"  {row.output_pdf}")

    if skipped.empty:
        print("Skipped figures: none")
    else:
        print("Skipped figures:")
        for row in skipped.itertuples(index=False):
            print(f"- {row.figure_id}: {row.skip_reason}")


if __name__ == "__main__":
    main()
