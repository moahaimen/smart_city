from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NodeState:
    node_id: int
    x: float
    y: float
    hotspot_relevance: float
    energy: float
    cooldown: int = 0
    cluster_head_id: int | None = None

    @property
    def alive(self) -> bool:
        return self.energy > 0.0


@dataclass
class NodeRoundContext:
    node_id: int
    current_pm25: float
    predicted_pm25: float
    current_severity: int
    predicted_severity: int
    residual_energy_ratio: float
    aoi: float
    change_rate: float
    hotspot_relevance: float
    communication_cost_norm: float
    distance_to_sink_norm: float
    priority_score: float
