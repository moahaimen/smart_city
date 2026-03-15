import unittest

from src.baselines.protocols import build_protocol
from src.simulation.aoi import AoITracker
from src.simulation.priority import PriorityInputs, cluster_head_score, compute_priority_score
from src.simulation.severity import map_pm25_to_severity
from src.simulation.types import NodeRoundContext, NodeState


class PollutionLogicTest(unittest.TestCase):
    def test_severity_mapping(self) -> None:
        thresholds = {"normal_max": 50.0, "warning_max": 100.0}
        self.assertEqual(map_pm25_to_severity(30.0, thresholds), 1)
        self.assertEqual(map_pm25_to_severity(75.0, thresholds), 2)
        self.assertEqual(map_pm25_to_severity(140.0, thresholds), 3)

    def test_aoi_update(self) -> None:
        tracker = AoITracker([1, 2])
        tracker.update({1})
        self.assertEqual(tracker.get(1), 0)
        self.assertEqual(tracker.get(2), 1)
        tracker.update(set())
        self.assertEqual(tracker.get(1), 1)
        self.assertEqual(tracker.get(2), 2)

    def test_priority_score(self) -> None:
        score = compute_priority_score(
            PriorityInputs(
                current_severity=3,
                predicted_severity=3,
                aoi=6,
                change_rate=18.0,
                hotspot_relevance=0.9,
                communication_cost=0.2,
            ),
            weight_config={
                "current_severity": 0.28,
                "predicted_severity": 0.24,
                "aoi": 0.18,
                "change_rate": 0.16,
                "hotspot_relevance": 0.10,
                "communication_cost": 0.04,
            },
            normalization_config={"max_aoi": 12, "max_change_rate": 40.0},
        )
        self.assertGreater(score, 0.6)

    def test_cluster_head_score(self) -> None:
        high = cluster_head_score(priority_score=0.8, residual_energy_ratio=0.9, distance_to_sink_norm=0.2, weights={"energy": 0.45, "priority": 0.4, "distance": 0.15})
        low = cluster_head_score(priority_score=0.2, residual_energy_ratio=0.4, distance_to_sink_norm=0.8, weights={"energy": 0.45, "priority": 0.4, "distance": 0.15})
        self.assertGreater(high, low)

    def test_predictive_protocol_preserves_hazardous_packets(self) -> None:
        protocol = build_protocol("predictive_pollution_aware_leach")
        context = NodeRoundContext(
            node_id=1,
            current_pm25=160.0,
            predicted_pm25=170.0,
            current_severity=3,
            predicted_severity=3,
            residual_energy_ratio=0.8,
            aoi=1.0,
            change_rate=1.0,
            hotspot_relevance=0.9,
            communication_cost_norm=0.4,
            distance_to_sink_norm=0.4,
            priority_score=0.9,
        )
        self.assertTrue(
            protocol.should_transmit(
                context,
                {
                    "suppression": {
                        "priority_threshold": 0.9,
                        "aoi_threshold": 10,
                        "change_threshold": 100.0,
                    }
                },
            )
        )


if __name__ == "__main__":
    unittest.main()
