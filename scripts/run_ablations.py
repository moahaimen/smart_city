from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import add_common_args, load_or_train_predictor, run_ablation_experiments
from src.utils.config import load_config
from src.utils.io import prepare_results_dirs


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the multi-seed ablation study for TCN-PPA-LEACH.")
    add_common_args(parser)
    parser.add_argument("--checkpoint", default="", help="Existing TCN checkpoint to load. If omitted, the script trains one.")
    args = parser.parse_args()

    config = load_config(args.config)
    results_dirs = prepare_results_dirs(config.get("output_root", "results"))
    bundle, _ = load_or_train_predictor(config, results_dirs, args.checkpoint)
    run_ablation_experiments(config, bundle, results_dirs)

    print(f"Saved ablation per-seed results to {results_dirs['tables'] / 'ablation_results.csv'}")
    print(f"Saved ablation summary table to {results_dirs['tables'] / 'ablation_summary_table.csv'}")
    print(f"Saved ablation fairness report to {results_dirs['logs'] / 'ablation_fairness_report.json'}")


if __name__ == "__main__":
    main()
