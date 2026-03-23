# Reviewer Checklist

## Model Consistency Fixed
- The repository is standardized to TCN terminology and files only.
- The proposed method identifier is `tcn_predictive_pollution_aware_leach`.
- The paper-facing short name is `TCN-PPA-LEACH`.

## Fairness Assumptions
- Shared simulation assumptions are logged for every run.
- Machine-readable fairness report: `results/logs/fairness_report.json`.
- PDR denominator is defined as delivered packets divided by raw packets generated.
- AoI and delay definitions are identical across all compared protocols.

## Multi-Seed Validation
- Main comparisons use 10 deterministic seeds.
- Aggregated statistics include mean, standard deviation, min, max, and 95% confidence intervals.

## Ablation Study Present
- Variants included: TCN-PPA-LEACH, No Prediction, No AoI, No Suppression, No Priority Scheduler.

## Reproducibility Artifacts
- Config-driven pipeline.
- Fixed seeds.
- Saved figures, tables, fairness reports, and machine-readable logs.

## Known Limitations
- Synthetic pollution data fallback may not capture all real-city effects.
- Results are simulation-based and not hardware-validated.
- Baseline set is limited to LEACH-family methods in this repository.
