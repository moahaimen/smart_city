from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.tcn_predictor import PredictorBundle
from src.pipeline import add_common_args, run_experiment_suite, train_predictor
from src.utils.config import load_config
from src.utils.io import prepare_results_dirs


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run LEACH protocol experiments.")
    add_common_args(parser)
    parser.add_argument("--checkpoint", default="", help="Existing TCN checkpoint to load. If omitted, the script trains one.")
    args = parser.parse_args()

    config = load_config(args.config)
    results_dirs = prepare_results_dirs(config.get("output_root", "results"))

    if args.checkpoint and Path(args.checkpoint).exists():
        bundle = PredictorBundle.load(checkpoint_path=args.checkpoint)
    else:
        train_result = train_predictor(config, results_dirs, int(config["seed"]))
        bundle = train_result["bundle"]

    experiment_result = run_experiment_suite(config, bundle, results_dirs, int(config["seed"]))
    print(f"Saved network summary to {results_dirs['tables'] / 'network_summary.csv'}")
    print(experiment_result["summary"][["scenario", "protocol", "packet_delivery_ratio", "average_aoi"]].head().to_string(index=False))


if __name__ == "__main__":
    main()
