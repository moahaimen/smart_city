# TCN-Assisted Predictive Pollution-Aware LEACH for Smart-City IoT Sensor Networks

This repository contains a runnable, TCN-only research prototype for smart-city air-pollution monitoring over LEACH-family wireless sensor networks.

The proposed method is:

- Code identifier: `tcn_predictive_pollution_aware_leach`
- Paper-facing short name: `TCN-PPA-LEACH`

The repository includes:

- a pollution time-series pipeline with real-CSV support and a documented synthetic fallback
- a lightweight PyTorch TCN predictor for next-step `PM2.5`
- severity mapping (`Normal`, `Warning`, `Hazardous`)
- sink-side Age of Information tracking
- predictive priority scoring
- multi-seed LEACH-family simulation
- ablation studies
- fairness logging
- publication-ready figures and summary tables

## Repository Layout

```text
configs/
data/
notebooks/
results/
scripts/
src/
tests/
reviewer_checklist.md
methods_snapshot.md
limitations.md
```

## Environment Setup

Requirements:

- Python 3.12+
- CPU execution is sufficient

Install dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Data Behavior

The training pipeline first checks:

```text
data/raw/air_quality.csv
```

If the file exists, it is used as the TCN training source. The CSV must provide the required pollution and environmental columns or be compatible with the loader in [src/data/pollution_data.py](/Users/moahaimentalib/Documents/scientfic_papers/leach_smart_city/LEACH-fullCode/src/data/pollution_data.py).

If no real CSV is present, the repository generates a deterministic synthetic smart-city dataset and saves it to:

```text
data/processed/synthetic_pollution_timeseries.csv
```

This synthetic fallback is useful for software validation, ablations, and manuscript drafting, but it must be described honestly as simulation data in the paper.

## Main Commands

Train the TCN only:

```bash
python3 scripts/train_tcn.py --config configs/default.yaml
```

Run the main multi-seed comparison study:

```bash
python3 scripts/run_experiments.py --config configs/default.yaml
```

Run the ablation study:

```bash
python3 scripts/run_ablations.py --config configs/default.yaml
```

Regenerate figures from saved CSV outputs:

```bash
python3 scripts/generate_figures.py --config configs/default.yaml
```

Print the final TCN metrics in a compact table:

```bash
python3 scripts/print_tcn_metrics.py --results-root results
```

Run the complete end-to-end pipeline:

```bash
python3 scripts/run_pipeline.py --config configs/default.yaml
```

## Default Study Design

Main comparison protocols:

- `standard_leach`
- `energy_aware_leach`
- `tcn_predictive_pollution_aware_leach`

Main scenarios:

- `normal`
- `rising_warning`
- `hazardous_spike`
- `hotspot_heavy`

Ablation variants:

- `full_tcn_ppa_leach`
- `no_tcn_prediction`
- `no_aoi_term`
- `no_suppression`
- `no_priority_scheduler`

Default reproducibility settings:

- deterministic seeds from [configs/default.yaml](/Users/moahaimentalib/Documents/scientfic_papers/leach_smart_city/LEACH-fullCode/configs/default.yaml)
- 10 seeds for main comparisons and ablations
- shared radio model, sink location, packet sizes, and scenario generation across protocols
- fairness report generated for both the main study and the ablation study

## Outputs

### TCN validation outputs

- [results/models/tcn_regressor.pt](/Users/moahaimentalib/Documents/scientfic_papers/leach_smart_city/LEACH-fullCode/results/models/tcn_regressor.pt)
- [results/tables/tcn_training_history.csv](/Users/moahaimentalib/Documents/scientfic_papers/leach_smart_city/LEACH-fullCode/results/tables/tcn_training_history.csv)
- [results/tables/tcn_test_predictions.csv](/Users/moahaimentalib/Documents/scientfic_papers/leach_smart_city/LEACH-fullCode/results/tables/tcn_test_predictions.csv)
- [results/tables/tcn_regression_metrics.csv](/Users/moahaimentalib/Documents/scientfic_papers/leach_smart_city/LEACH-fullCode/results/tables/tcn_regression_metrics.csv)
- [results/tables/tcn_classification_metrics.csv](/Users/moahaimentalib/Documents/scientfic_papers/leach_smart_city/LEACH-fullCode/results/tables/tcn_classification_metrics.csv)
- [results/logs/test_metrics.json](/Users/moahaimentalib/Documents/scientfic_papers/leach_smart_city/LEACH-fullCode/results/logs/test_metrics.json)
- [results/logs/tcn_config_used.json](/Users/moahaimentalib/Documents/scientfic_papers/leach_smart_city/LEACH-fullCode/results/logs/tcn_config_used.json)
- [results/figures/tcn_training_curves.png](/Users/moahaimentalib/Documents/scientfic_papers/leach_smart_city/LEACH-fullCode/results/figures/tcn_training_curves.png)
- [results/figures/tcn_prediction_samples.png](/Users/moahaimentalib/Documents/scientfic_papers/leach_smart_city/LEACH-fullCode/results/figures/tcn_prediction_samples.png)

### Main study outputs

- [results/tables/per_seed_results.csv](/Users/moahaimentalib/Documents/scientfic_papers/leach_smart_city/LEACH-fullCode/results/tables/per_seed_results.csv)
- [results/tables/aggregated_results.csv](/Users/moahaimentalib/Documents/scientfic_papers/leach_smart_city/LEACH-fullCode/results/tables/aggregated_results.csv)
- [results/tables/scenario_protocol_summary.csv](/Users/moahaimentalib/Documents/scientfic_papers/leach_smart_city/LEACH-fullCode/results/tables/scenario_protocol_summary.csv)
- [results/tables/paper_summary_table.csv](/Users/moahaimentalib/Documents/scientfic_papers/leach_smart_city/LEACH-fullCode/results/tables/paper_summary_table.csv)
- [results/tables/paper_summary_table.md](/Users/moahaimentalib/Documents/scientfic_papers/leach_smart_city/LEACH-fullCode/results/tables/paper_summary_table.md)
- [results/logs/per_seed_round_metrics.csv](/Users/moahaimentalib/Documents/scientfic_papers/leach_smart_city/LEACH-fullCode/results/logs/per_seed_round_metrics.csv)
- [results/logs/run_assumptions.csv](/Users/moahaimentalib/Documents/scientfic_papers/leach_smart_city/LEACH-fullCode/results/logs/run_assumptions.csv)
- [results/logs/fairness_report.json](/Users/moahaimentalib/Documents/scientfic_papers/leach_smart_city/LEACH-fullCode/results/logs/fairness_report.json)

### Ablation outputs

- [results/tables/ablation_results.csv](/Users/moahaimentalib/Documents/scientfic_papers/leach_smart_city/LEACH-fullCode/results/tables/ablation_results.csv)
- [results/tables/ablation_aggregated_results.csv](/Users/moahaimentalib/Documents/scientfic_papers/leach_smart_city/LEACH-fullCode/results/tables/ablation_aggregated_results.csv)
- [results/tables/ablation_summary_table.csv](/Users/moahaimentalib/Documents/scientfic_papers/leach_smart_city/LEACH-fullCode/results/tables/ablation_summary_table.csv)
- [results/logs/ablation_round_metrics.csv](/Users/moahaimentalib/Documents/scientfic_papers/leach_smart_city/LEACH-fullCode/results/logs/ablation_round_metrics.csv)
- [results/logs/ablation_run_assumptions.csv](/Users/moahaimentalib/Documents/scientfic_papers/leach_smart_city/LEACH-fullCode/results/logs/ablation_run_assumptions.csv)
- [results/logs/ablation_fairness_report.json](/Users/moahaimentalib/Documents/scientfic_papers/leach_smart_city/LEACH-fullCode/results/logs/ablation_fairness_report.json)

### Figures

- [results/figures/alive_nodes_vs_rounds.png](/Users/moahaimentalib/Documents/scientfic_papers/leach_smart_city/LEACH-fullCode/results/figures/alive_nodes_vs_rounds.png)
- [results/figures/residual_energy_vs_rounds.png](/Users/moahaimentalib/Documents/scientfic_papers/leach_smart_city/LEACH-fullCode/results/figures/residual_energy_vs_rounds.png)
- [results/figures/packet_delivery_ratio_comparison.png](/Users/moahaimentalib/Documents/scientfic_papers/leach_smart_city/LEACH-fullCode/results/figures/packet_delivery_ratio_comparison.png)
- [results/figures/delay_comparison.png](/Users/moahaimentalib/Documents/scientfic_papers/leach_smart_city/LEACH-fullCode/results/figures/delay_comparison.png)
- [results/figures/average_aoi_comparison.png](/Users/moahaimentalib/Documents/scientfic_papers/leach_smart_city/LEACH-fullCode/results/figures/average_aoi_comparison.png)
- [results/figures/hazardous_event_success_comparison.png](/Users/moahaimentalib/Documents/scientfic_papers/leach_smart_city/LEACH-fullCode/results/figures/hazardous_event_success_comparison.png)
- [results/figures/ablation_comparison.png](/Users/moahaimentalib/Documents/scientfic_papers/leach_smart_city/LEACH-fullCode/results/figures/ablation_comparison.png)

### Reviewer-facing documents

- [reviewer_checklist.md](/Users/moahaimentalib/Documents/scientfic_papers/leach_smart_city/LEACH-fullCode/reviewer_checklist.md)
- [methods_snapshot.md](/Users/moahaimentalib/Documents/scientfic_papers/leach_smart_city/LEACH-fullCode/methods_snapshot.md)
- [limitations.md](/Users/moahaimentalib/Documents/scientfic_papers/leach_smart_city/LEACH-fullCode/limitations.md)

## Expected Runtime

On CPU, the full default pipeline usually completes in a few minutes. Runtime depends on:

- Python environment speed
- whether the TCN must be retrained
- the number of seeds in [configs/default.yaml](/Users/moahaimentalib/Documents/scientfic_papers/leach_smart_city/LEACH-fullCode/configs/default.yaml)

If you need a faster smoke run, reduce:

- `study.seeds`
- `model.epochs`
- scenario round counts

Use those smaller settings only for debugging, not for the final reported paper outputs.

## Tests

Run the lightweight test suite:

```bash
python3 -m unittest discover -s tests -v
```

The tests cover:

- severity mapping
- AoI updates
- priority scoring
- cluster-head scoring
- confidence interval logic
- per-seed aggregation logic
- TCN checkpoint round-trip
- legacy standalone LEACH smoke checks

## Troubleshooting

If matplotlib fails because of cache or config permissions, the plotting module already redirects its cache into `results/logs/`.

If the full study takes too long on your machine:

- reduce the seed list for a smoke run
- confirm that the TCN checkpoint already exists
- run `scripts/run_experiments.py` and `scripts/run_ablations.py` separately

If `data/raw/air_quality.csv` is not present, synthetic data is expected. Do not describe the outputs as real-data results in that case.
