from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import add_common_args, load_or_train_predictor, run_main_experiments
from src.utils.config import load_config
from src.utils.io import prepare_results_dirs


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the main multi-seed LEACH protocol comparisons.")
    add_common_args(parser)
    parser.add_argument("--checkpoint", default="", help="Existing TCN checkpoint to load. If omitted, the script trains one.")
    args = parser.parse_args()

    config = load_config(args.config)
    results_dirs = prepare_results_dirs(config.get("output_root", "results"))
    bundle, _ = load_or_train_predictor(config, results_dirs, args.checkpoint)
    experiment_result = run_main_experiments(config, bundle, results_dirs)

    print(f"Saved per-seed results to {results_dirs['tables'] / 'per_seed_results.csv'}")
    print(f"Saved aggregated summary to {results_dirs['tables'] / 'scenario_protocol_summary.csv'}")
    print(f"Saved fairness report to {results_dirs['logs'] / 'fairness_report.json'}")
    print(
        experiment_result["paper_summary"][
            ["Scenario", "Protocol", "PDR mean±std", "AoI mean±std", "Hazardous success mean±std"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
