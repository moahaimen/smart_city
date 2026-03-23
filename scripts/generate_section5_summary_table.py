from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str((ROOT / "results" / "logs" / "mplconfig").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str((ROOT / "results" / "logs" / "cache").resolve()))
sys.path.insert(0, str(ROOT))

MAIN_PROTOCOLS = [
    ("standard_leach", "LEACH"),
    ("energy_aware_leach", "EA-LEACH"),
    ("tcn_predictive_pollution_aware_leach", "TCN-PPA-LEACH"),
]
DISPLAY_ORDER = [label for _, label in MAIN_PROTOCOLS]
UNAVAILABLE = "Unavailable"


def load_main_results() -> pd.DataFrame:
    path = ROOT / "results" / "tables" / "per_seed_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing required results file: {path}")

    frame = pd.read_csv(path)
    protocol_ids = [protocol for protocol, _ in MAIN_PROTOCOLS]
    frame = frame[
        (frame["study_name"] == "main_comparisons")
        & (frame["scenario_group"] == "evaluation")
        & (frame["protocol"].isin(protocol_ids))
    ].copy()
    frame["Method"] = frame["protocol_label"]
    frame["Total throughput (Mbits)"] = (frame["packets_delivered"] * 4000.0) / 1e6
    frame["Raw suppression ratio"] = frame["packets_suppressed"] / frame["packets_generated"]
    return frame


def mean_std(frame: pd.DataFrame, method: str, column: str) -> tuple[float, float]:
    series = frame.loc[frame["Method"] == method, column]
    return float(series.mean()), float(series.std(ddof=1))


def format_mean_std(mean: float, std: float, decimals: int) -> str:
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def build_display_table(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    numeric_cache: dict[str, dict[str, float]] = {}
    rows: list[dict[str, str]] = []

    metric_specs = [
        ("FND", "fnd", 1),
        ("HND", "hnd", 1),
        ("LND", "lnd", 1),
        ("Avg residual energy (J)", "average_residual_energy_over_time", 3),
        ("Throughput (Mbits)", "Total throughput (Mbits)", 3),
        ("Delivered packets", "packets_delivered", 1),
        ("PDR", "packet_delivery_ratio", 3),
        ("Delay (hops)", "end_to_end_delay", 3),
        ("Avg AoI (rounds)", "average_aoi", 3),
        ("Hazardous success rate", "hazardous_event_delivery_success_rate", 3),
        ("Hazardous packets delivered", "hazardous_packets_delivered", 1),
        ("Raw suppression ratio", "Raw suppression ratio", 3),
    ]

    for method in DISPLAY_ORDER:
        row = {"Method": method}
        numeric_cache[method] = {}
        for display_name, source_name, decimals in metric_specs:
            mean, std = mean_std(frame, method, source_name)
            row[display_name] = format_mean_std(mean, std, decimals)
            numeric_cache[method][f"{display_name}_mean"] = mean
            numeric_cache[method][f"{display_name}_std"] = std

        row["Routine suppression ratio"] = UNAVAILABLE
        row["CH energy burden"] = UNAVAILABLE
        row["Fairness metric"] = UNAVAILABLE
        rows.append(row)

    return pd.DataFrame.from_records(rows), numeric_cache


def build_improvement_table(numeric_cache: dict[str, dict[str, float]]) -> pd.DataFrame:
    baseline_methods = ["LEACH", "EA-LEACH"]
    tcn_method = "TCN-PPA-LEACH"

    improvement_specs = [
        ("FND", "higher"),
        ("HND", "higher"),
        ("LND", "higher"),
        ("Avg residual energy (J)", "higher"),
        ("Throughput (Mbits)", "higher"),
        ("Delivered packets", "higher"),
        ("PDR", "higher"),
        ("Delay (hops)", "lower"),
        ("Avg AoI (rounds)", "lower"),
        ("Hazardous success rate", "higher"),
        ("Hazardous packets delivered", "higher"),
    ]

    rows: list[dict[str, str]] = []
    for metric, direction in improvement_specs:
        baseline_values = {method: numeric_cache[method][f"{metric}_mean"] for method in baseline_methods}
        if direction == "higher":
            strongest_baseline = max(baseline_values, key=baseline_values.get)
            baseline_value = baseline_values[strongest_baseline]
            tcn_value = numeric_cache[tcn_method][f"{metric}_mean"]
            change_pct = ((tcn_value - baseline_value) / baseline_value) * 100.0 if baseline_value else float("nan")
        else:
            strongest_baseline = min(baseline_values, key=baseline_values.get)
            baseline_value = baseline_values[strongest_baseline]
            tcn_value = numeric_cache[tcn_method][f"{metric}_mean"]
            change_pct = ((baseline_value - tcn_value) / baseline_value) * 100.0 if baseline_value else float("nan")

        rows.append(
            {
                "Metric": metric,
                "Optimization direction": "Higher is better" if direction == "higher" else "Lower is better",
                "Strongest baseline": strongest_baseline,
                "Baseline mean": f"{baseline_value:.3f}",
                "TCN-PPA-LEACH mean": f"{tcn_value:.3f}",
                "TCN vs strongest baseline (%)": f"{change_pct:.3f}",
            }
        )

    return pd.DataFrame.from_records(rows)


def dataframe_to_latex(display_df: pd.DataFrame) -> str:
    column_format = "l" + "c" * (len(display_df.columns) - 1)
    latex_body = display_df.to_latex(index=False, escape=False, column_format=column_format)
    return (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\caption{Section 5 summary table aggregated over the full main evaluation bundle "
        "(4 scenarios $\\times$ 10 seeds = 40 runs per method). Values are reported as mean $\\pm$ standard deviation. "
        "Routine-only suppression ratio, CH energy burden, and fairness metric are unavailable in the saved simulator outputs.}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        f"{latex_body}"
        "}\n"
        "\\end{table*}\n"
    )


def main() -> None:
    output_dir = ROOT / "results" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    display_csv_path = output_dir / "section5_final_summary_table.csv"
    latex_path = output_dir / "section5_final_summary_table.tex"
    markdown_path = output_dir / "section5_final_summary_table.md"
    improvement_csv_path = output_dir / "section5_tcn_improvement_summary.csv"

    frame = load_main_results()
    display_df, numeric_cache = build_display_table(frame)
    improvement_df = build_improvement_table(numeric_cache)

    display_df.to_csv(display_csv_path, index=False)
    improvement_df.to_csv(improvement_csv_path, index=False)
    markdown_path.write_text(display_df.to_markdown(index=False) + "\n", encoding="utf-8")
    latex_path.write_text(dataframe_to_latex(display_df), encoding="utf-8")

    print(f"Main summary CSV: {display_csv_path}")
    print(f"LaTeX table: {latex_path}")
    print(f"Markdown table: {markdown_path}")
    print(f"Improvement CSV: {improvement_csv_path}")
    print("\nDisplay table:")
    print(display_df.to_markdown(index=False))
    print("\nImprovement table:")
    print(improvement_df.to_markdown(index=False))


if __name__ == "__main__":
    main()
