from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Print the final TCN test metrics in a compact table.")
    parser.add_argument("--results-root", default="results", help="Results root directory.")
    args = parser.parse_args()

    metrics_path = Path(args.results_root) / "logs" / "test_metrics.json"
    with metrics_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    regression = payload["regression"]
    classification = payload["classification"]
    print("TCN Test Metrics")
    print("================")
    print(f"Data source      : {payload['data_source']}")
    print(f"Train/Val/Test   : {payload['train_sequences']}/{payload['val_sequences']}/{payload['test_sequences']}")
    print(f"RMSE             : {regression['rmse']:.4f}")
    print(f"MAE              : {regression['mae']:.4f}")
    print(f"MAPE             : {regression['mape']:.4f}")
    print(f"R^2              : {regression['r2']:.4f}")
    print(f"Accuracy         : {classification['accuracy']:.4f}")
    print(f"Precision macro  : {classification['precision_macro']:.4f}")
    print(f"Recall macro     : {classification['recall_macro']:.4f}")
    print(f"F1 macro         : {classification['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
