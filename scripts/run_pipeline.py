from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import add_common_args, run_full_pipeline
from src.utils.config import load_config


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the full predictive pollution-aware LEACH pipeline.")
    add_common_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    result = run_full_pipeline(config)
    print("Pipeline completed successfully.")
    print(f"Model checkpoint: {result['summary']['model_checkpoint']}")
    print(f"Network summary table: {result['summary']['network_summary_table']}")
    print(f"Paper summary table: {result['summary']['paper_summary_table']}")


if __name__ == "__main__":
    main()
