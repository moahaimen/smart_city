from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("results/logs/mplconfig").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("results/logs/cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.naming import protocol_label, scenario_label

FIGURE_DPI = 1000


def _save_figure(fig: plt.Figure, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight")
    if output.suffix.lower() == ".png":
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")


def plot_training_curves(history_df: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history_df["epoch"], history_df["train_loss"], label="Train")
    ax.plot(history_df["epoch"], history_df["val_loss"], label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("TCN Training Curves")
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)


def plot_predictions(predictions_df: pd.DataFrame, output_path: str | Path, max_points: int = 250) -> None:
    plot_df = predictions_df.head(max_points)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(plot_df.index, plot_df["y_true"], label="True PM2.5")
    ax.plot(plot_df.index, plot_df["y_pred"], label="Predicted PM2.5")
    ax.set_xlabel("Sample")
    ax.set_ylabel("PM2.5")
    ax.set_title("TCN Prediction Samples")
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)


def _plot_metric_grid(rounds_df: pd.DataFrame, metric: str, ylabel: str, title: str, output_path: str | Path) -> None:
    evaluation_df = rounds_df[rounds_df["scenario_group"] == "evaluation"]
    scenarios = list(dict.fromkeys(evaluation_df["scenario"]))
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=False)
    axes = axes.ravel()
    for axis, scenario_name in zip(axes, scenarios):
        subset = evaluation_df[evaluation_df["scenario"] == scenario_name]
        grouped = subset.groupby(["protocol", "protocol_label", "round"], as_index=False)[metric].agg(["mean", "std", "count"]).reset_index()
        for protocol_name, protocol_df in grouped.groupby("protocol_label"):
            mean_values = protocol_df["mean"].to_numpy(dtype=float)
            std_values = protocol_df["std"].fillna(0.0).to_numpy(dtype=float)
            count_values = protocol_df["count"].to_numpy(dtype=float)
            ci_half = np.where(count_values > 1, 1.96 * std_values / np.sqrt(count_values), 0.0)
            rounds = protocol_df["round"].to_numpy(dtype=int)
            axis.plot(rounds, mean_values, label=protocol_name)
            axis.fill_between(rounds, mean_values - ci_half, mean_values + ci_half, alpha=0.15)
        axis.set_title(scenario_label(scenario_name))
        axis.set_xlabel("Round")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    for axis in axes[len(scenarios) :]:
        axis.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(title)
    fig.legend(handles, labels, loc="upper center", ncol=min(3, len(labels)))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save_figure(fig, output_path)
    plt.close(fig)


def plot_alive_nodes(rounds_df: pd.DataFrame, output_path: str | Path) -> None:
    _plot_metric_grid(rounds_df, metric="alive_nodes", ylabel="Alive nodes", title="Alive Nodes vs Rounds", output_path=output_path)


def plot_residual_energy(rounds_df: pd.DataFrame, output_path: str | Path) -> None:
    _plot_metric_grid(
        rounds_df,
        metric="avg_residual_energy",
        ylabel="Average residual energy (J)",
        title="Residual Energy vs Rounds",
        output_path=output_path,
    )


def plot_summary_bar(summary_df: pd.DataFrame, metric: str, title: str, ylabel: str, output_path: str | Path) -> None:
    if "scenario_group" in summary_df.columns:
        evaluation_df = summary_df[summary_df["scenario_group"] == "evaluation"]
    else:
        evaluation_df = summary_df
    scenarios = list(dict.fromkeys(evaluation_df["scenario"]))
    protocols = list(dict.fromkeys(evaluation_df["protocol"]))
    width = 0.8 / max(len(protocols), 1)
    positions = range(len(scenarios))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for offset, protocol_name in enumerate(protocols):
        values = []
        errors = []
        for scenario_name in scenarios:
            row = evaluation_df[(evaluation_df["scenario"] == scenario_name) & (evaluation_df["protocol"] == protocol_name)].iloc[0]
            values.append(row[f"{metric}_mean"])
            errors.append(row[f"{metric}_ci95_half_width"])
        shift = offset - (len(protocols) - 1) / 2.0
        x_positions = [position + shift * width for position in positions]
        ax.bar(x_positions, values, width=width, yerr=errors, capsize=3, label=protocol_label(protocol_name))

    ax.set_xticks(list(positions))
    ax.set_xticklabels([scenario_label(scenario) for scenario in scenarios], rotation=15)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)


def plot_ablation_bar(summary_df: pd.DataFrame, metric: str, title: str, ylabel: str, output_path: str | Path) -> None:
    scenarios = list(dict.fromkeys(summary_df["scenario"]))
    protocols = list(dict.fromkeys(summary_df["protocol"]))
    width = 0.16
    positions = range(len(scenarios))

    fig, ax = plt.subplots(figsize=(11, 4.8))
    for offset, protocol_name in enumerate(protocols):
        values = []
        errors = []
        for scenario_name in scenarios:
            row = summary_df[(summary_df["scenario"] == scenario_name) & (summary_df["protocol"] == protocol_name)].iloc[0]
            values.append(row[f"{metric}_mean"])
            errors.append(row[f"{metric}_ci95_half_width"])
        shift = offset - (len(protocols) - 1) / 2.0
        x_positions = [position + shift * width for position in positions]
        ax.bar(x_positions, values, width=width, yerr=errors, capsize=3, label=protocol_label(protocol_name))

    ax.set_xticks(list(positions))
    ax.set_xticklabels([scenario_label(scenario) for scenario in scenarios], rotation=15)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)
