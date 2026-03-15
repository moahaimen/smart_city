from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("results/logs/mplconfig").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("results/logs/cache").resolve()))

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_curves(history_df: pd.DataFrame, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history_df["epoch"], history_df["train_loss"], label="Train")
    ax.plot(history_df["epoch"], history_df["val_loss"], label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("TCN Training Curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_predictions(predictions_df: pd.DataFrame, output_path: str | Path, max_points: int = 250) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_df = predictions_df.head(max_points)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(plot_df.index, plot_df["y_true"], label="True PM2.5")
    ax.plot(plot_df.index, plot_df["y_pred"], label="Predicted PM2.5")
    ax.set_xlabel("Sample")
    ax.set_ylabel("PM2.5")
    ax.set_title("TCN Prediction Samples")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _plot_metric_grid(rounds_df: pd.DataFrame, metric: str, ylabel: str, title: str, output_path: str | Path) -> None:
    evaluation_df = rounds_df[rounds_df["scenario_group"] == "evaluation"]
    scenarios = list(dict.fromkeys(evaluation_df["scenario"]))
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=False)
    axes = axes.ravel()
    for axis, scenario_name in zip(axes, scenarios):
        subset = evaluation_df[evaluation_df["scenario"] == scenario_name]
        for protocol_name, protocol_df in subset.groupby("protocol"):
            axis.plot(protocol_df["round"], protocol_df[metric], label=protocol_name)
        axis.set_title(scenario_name)
        axis.set_xlabel("Round")
        axis.set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(title)
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
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
    evaluation_df = summary_df[summary_df["scenario_group"] == "evaluation"]
    scenarios = list(dict.fromkeys(evaluation_df["scenario"]))
    protocols = list(dict.fromkeys(evaluation_df["protocol"]))
    width = 0.25
    positions = range(len(scenarios))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for offset, protocol_name in enumerate(protocols):
        values = []
        for scenario_name in scenarios:
            row = evaluation_df[(evaluation_df["scenario"] == scenario_name) & (evaluation_df["protocol"] == protocol_name)].iloc[0]
            values.append(row[metric])
        x_positions = [position + (offset - 1) * width for position in positions]
        ax.bar(x_positions, values, width=width, label=protocol_name)

    ax.set_xticks(list(positions))
    ax.set_xticklabels(scenarios, rotation=15)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
