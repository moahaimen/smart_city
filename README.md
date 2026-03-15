# Predictive Pollution-Aware LEACH Research Prototype

This repository contains a runnable research prototype for:

“Predictive Pollution-Aware LEACH: A TCN-Based Deep Learning Framework for Smart-City IoT Sensor Networks”

The project is specific to smart-city air-pollution monitoring. It combines:

- a synthetic or real pollution time-series data pipeline
- a lightweight PyTorch TCN predictor
- AQI-style severity mapping
- sink-side Age of Information tracking
- a configurable predictive priority score
- three routing protocols:
  - `standard_leach`
  - `energy_aware_leach`
  - `predictive_pollution_aware_leach`
- automated experiment execution
- publication-style figures and CSV summary tables

## Repository Layout

```text
configs/
data/
notebooks/
results/
scripts/
src/
tests/
```

## Requirements

- Python 3.12+
- PyTorch
- NumPy
- pandas
- SciPy
- matplotlib
- scikit-learn
- PyYAML

Install them with:

```bash
python3 -m pip install -r requirements.txt
```

## Main Reproduction Commands

Train the TCN only:

```bash
python3 scripts/train_tcn.py --config configs/default.yaml
```

Run only the network experiments. If no checkpoint exists, this trains one automatically:

```bash
python3 scripts/run_experiments.py --config configs/default.yaml
```

Run the full end-to-end pipeline:

```bash
python3 scripts/run_pipeline.py --config configs/default.yaml
```

## Dataset Behavior

The code first checks `data/raw/air_quality.csv`.

- If a real CSV exists, it is used for TCN training.
- If no real CSV exists, the code generates a deterministic synthetic smart-city pollution dataset with:
  - `PM2.5`
  - `PM10`
  - `CO`
  - `NO2`
  - temperature
  - humidity
  - time-of-day features

Synthetic training data is saved to:

```text
data/processed/synthetic_pollution_timeseries.csv
```

Scenario-specific simulation traces are saved under:

```text
data/processed/scenarios/
```

## What the Pipeline Produces

Model outputs:

- `results/models/tcn_regressor.pt`
- `results/tables/tcn_training_history.csv`
- `results/tables/tcn_test_predictions.csv`
- `results/tables/tcn_regression_metrics.csv`
- `results/tables/tcn_classification_metrics.csv`
- `results/figures/tcn_training_curves.png`
- `results/figures/tcn_predictions.png`

Network outputs:

- `results/tables/network_summary.csv`
- `results/tables/paper_summary_table.csv`
- `results/tables/sensitivity_summary.csv`
- `results/logs/network_round_metrics.csv`
- `results/figures/alive_nodes_vs_rounds.png`
- `results/figures/residual_energy_vs_rounds.png`
- `results/figures/packet_delivery_ratio_comparison.png`
- `results/figures/delay_comparison.png`
- `results/figures/aoi_comparison.png`
- `results/figures/hazardous_event_success_comparison.png`

Run metadata:

- `results/logs/training_summary.json`
- `results/logs/pipeline_summary.json`

## Scientific Defaults

The default config runs:

- 4 core pollution scenarios:
  - `normal`
  - `rising_warning`
  - `hazardous_spike`
  - `hotspot_heavy`
- 4 sensitivity cases:
  - node-count variation
  - area-size variation
  - initial-energy variation

The TCN predicts the next-step `PM2.5` value. Severity classes are then derived from configurable thresholds:

- Class 1: Normal
- Class 2: Warning
- Class 3: Hazardous

## Testing

Run all tests with:

```bash
python3 -m unittest discover -s tests -v
```

The tests cover:

- severity mapping
- AoI updates
- priority scoring
- cluster-head scoring
- deterministic TCN/simulation smoke checks

## Notes

- The old MATLAB files are kept for historical reference, but the reproducible research workflow is the Python project under `src/` and `scripts/`.
- All random seeds are fixed through the YAML config and the pipeline utilities.
