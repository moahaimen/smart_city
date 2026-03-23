from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import add_common_args, regenerate_figures
from src.utils.config import load_config
from src.utils.io import prepare_results_dirs


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Regenerate figures from saved TCN-PPA-LEACH results.")
    add_common_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    results_dirs = prepare_results_dirs(config.get("output_root", "results"))
    figure_paths = regenerate_figures(config, results_dirs)
    for name, path in figure_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
