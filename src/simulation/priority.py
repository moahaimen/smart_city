from __future__ import annotations

from dataclasses import dataclass

from src.simulation.severity import severity_to_norm


@dataclass
class PriorityInputs:
    current_severity: int
    predicted_severity: int
    aoi: float
    change_rate: float
    hotspot_relevance: float
    communication_cost: float


def compute_priority_score(
    inputs: PriorityInputs,
    weight_config: dict[str, float],
    normalization_config: dict[str, float],
) -> float:
    aoi_norm = min(inputs.aoi / normalization_config["max_aoi"], 1.0)
    change_norm = min(abs(inputs.change_rate) / normalization_config["max_change_rate"], 1.0)
    cost_norm = min(max(inputs.communication_cost, 0.0), 1.0)
    score = (
        weight_config["current_severity"] * float(severity_to_norm(inputs.current_severity))
        + weight_config["predicted_severity"] * float(severity_to_norm(inputs.predicted_severity))
        + weight_config["aoi"] * aoi_norm
        + weight_config["change_rate"] * change_norm
        + weight_config["hotspot_relevance"] * inputs.hotspot_relevance
        - weight_config["communication_cost"] * cost_norm
    )
    return max(0.0, min(score, 1.0))


def cluster_head_score(priority_score: float, residual_energy_ratio: float, distance_to_sink_norm: float, weights: dict[str, float]) -> float:
    return (
        weights["energy"] * residual_energy_ratio
        + weights["priority"] * priority_score
        + weights["distance"] * (1.0 - distance_to_sink_norm)
    )
