import unittest

import pandas as pd

from src.metrics.statistics import aggregate_metric_frame, confidence_interval


class StatisticsTest(unittest.TestCase):
    def test_confidence_interval_spans_mean(self) -> None:
        low, high, half_width = confidence_interval([1.0, 2.0, 3.0, 4.0], confidence_level=0.95)
        self.assertLess(low, 2.5)
        self.assertGreater(high, 2.5)
        self.assertGreater(half_width, 0.0)

    def test_aggregate_metric_frame_produces_expected_columns(self) -> None:
        frame = pd.DataFrame(
            [
                {"study_name": "main", "scenario": "normal", "scenario_label": "Normal", "scenario_group": "evaluation", "protocol": "standard_leach", "protocol_label": "LEACH", "fnd": 100, "pdr": 0.8},
                {"study_name": "main", "scenario": "normal", "scenario_label": "Normal", "scenario_group": "evaluation", "protocol": "standard_leach", "protocol_label": "LEACH", "fnd": 110, "pdr": 0.7},
            ]
        )
        aggregated_long, aggregated_wide = aggregate_metric_frame(
            frame=frame,
            group_cols=["study_name", "scenario", "scenario_label", "scenario_group", "protocol", "protocol_label"],
            metric_cols=["fnd", "pdr"],
            confidence_level=0.95,
        )

        self.assertEqual(len(aggregated_long), 2)
        self.assertEqual(len(aggregated_wide), 1)
        self.assertIn("fnd_mean", aggregated_wide.columns)
        self.assertIn("pdr_ci95_half_width", aggregated_wide.columns)
        self.assertAlmostEqual(float(aggregated_wide.iloc[0]["fnd_mean"]), 105.0)


if __name__ == "__main__":
    unittest.main()
