from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import add_common_args, train_predictor
from src.utils.config import load_config
from src.utils.io import prepare_results_dirs


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train the TCN pollution predictor.")
    add_common_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    results_dirs = prepare_results_dirs(config.get("output_root", "results"))
    result = train_predictor(config, results_dirs, int(config["seed"]))
    print(f"Saved TCN checkpoint to {result['bundle'].checkpoint_path}")
    print(f"Regression metrics: {result['regression_metrics']}")
    print(f"Classification metrics: {result['classification_metrics']}")


if __name__ == "__main__":
    main()
