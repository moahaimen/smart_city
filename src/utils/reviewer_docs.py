from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.naming import protocol_label


def write_reviewer_checklist(output_path: str | Path, fairness_report_path: str, multi_seed_count: int, ablation_variants: list[str]) -> None:
    content = f"""# Reviewer Checklist

## Model Consistency Fixed
- The repository is standardized to TCN terminology and files only.
- The proposed method identifier is `tcn_predictive_pollution_aware_leach`.
- The paper-facing short name is `TCN-PPA-LEACH`.

## Fairness Assumptions
- Shared simulation assumptions are logged for every run.
- Machine-readable fairness report: `{fairness_report_path}`.
- PDR denominator is defined as delivered packets divided by raw packets generated.
- AoI and delay definitions are identical across all compared protocols.

## Multi-Seed Validation
- Main comparisons use {multi_seed_count} deterministic seeds.
- Aggregated statistics include mean, standard deviation, min, max, and 95% confidence intervals.

## Ablation Study Present
- Variants included: {", ".join(ablation_variants)}.

## Reproducibility Artifacts
- Config-driven pipeline.
- Fixed seeds.
- Saved figures, tables, fairness reports, and machine-readable logs.

## Known Limitations
- Synthetic pollution data fallback may not capture all real-city effects.
- Results are simulation-based and not hardware-validated.
- Baseline set is limited to LEACH-family methods in this repository.
"""
    Path(output_path).write_text(content, encoding="utf-8")


def write_methods_snapshot(output_path: str | Path) -> None:
    content = """# Methods Snapshot

## TCN Role
The TCN predicts next-step PM2.5 from recent multivariate pollution windows. Its predicted severity is fed into the routing priority score for TCN-PPA-LEACH.

## Severity Mapping
PM2.5 is mapped into three classes: Normal, Warning, and Hazardous using configurable AQI-style thresholds.

## AoI Definition
Age of Information at the sink increments by one each round when no fresh packet from a node arrives and resets to zero on successful delivery.

## Priority Score
The node priority score combines current severity, predicted severity, AoI, change rate, hotspot relevance, and communication cost using configurable weights.

## Cluster-Head Election
Standard LEACH uses probabilistic election. Energy-aware LEACH uses residual-energy and distance ranking. TCN-PPA-LEACH uses residual energy, predictive priority, and sink distance.

## Suppression Rule
Only TCN-PPA-LEACH applies routine suppression. Hazardous packets are never suppressed. Warning and hazardous packets always remain eligible for transmission.

## Compared Baselines
- LEACH
- EA-LEACH
- TCN-PPA-LEACH
"""
    Path(output_path).write_text(content, encoding="utf-8")


def write_limitations(output_path: str | Path) -> None:
    content = """# Limitations

- The default pipeline uses synthetic smart-city pollution data when no real CSV is available.
- The study is simulation-based and does not include field deployment or embedded-device validation.
- Baseline coverage is limited to LEACH and an energy-aware LEACH variant in the current repository.
- The TCN is intentionally lightweight; stronger architectures may improve prediction accuracy at the cost of complexity.
- Hazardous-event success depends on the configured severity thresholds and scenario generator assumptions.
"""
    Path(output_path).write_text(content, encoding="utf-8")


def build_paper_summary_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in summary_df.itertuples(index=False):
        rows.append(
            {
                "Scenario": row.scenario_label,
                "Protocol": row.protocol_label,
                "FND mean±std": f"{row.fnd_mean:.2f}±{row.fnd_std:.2f}",
                "LND mean±std": f"{row.lnd_mean:.2f}±{row.lnd_std:.2f}",
                "PDR mean±std": f"{row.packet_delivery_ratio_mean:.3f}±{row.packet_delivery_ratio_std:.3f}",
                "Delay mean±std": f"{row.end_to_end_delay_mean:.3f}±{row.end_to_end_delay_std:.3f}",
                "AoI mean±std": f"{row.average_aoi_mean:.3f}±{row.average_aoi_std:.3f}",
                "Hazardous success mean±std": f"{row.hazardous_event_delivery_success_rate_mean:.3f}±{row.hazardous_event_delivery_success_rate_std:.3f}",
            }
        )
    return pd.DataFrame.from_records(rows)
