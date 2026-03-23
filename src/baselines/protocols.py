from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.simulation.priority import cluster_head_score
from src.simulation.types import NodeRoundContext, NodeState
from src.utils.naming import protocol_label


@dataclass
class ProtocolBase:
    name: str
    short_name: str
    cluster_head_rule: str
    join_rule: str
    uses_prediction: bool = False
    uses_aoi_term: bool = False
    uses_suppression: bool = False
    uses_priority_scheduler: bool = False
    priority_scoring_enabled: bool = False

    def select_cluster_heads(
        self,
        node_states: dict[int, NodeState],
        contexts: dict[int, NodeRoundContext],
        round_index: int,
        network_config: dict,
        rng: np.random.Generator,
    ) -> list[int]:
        raise NotImplementedError

    def choose_cluster_head(
        self,
        node_id: int,
        candidate_cluster_heads: list[int],
        node_states: dict[int, NodeState],
        contexts: dict[int, NodeRoundContext],
        sink_position: tuple[float, float],
        area_diagonal: float,
        network_config: dict,
    ) -> int | None:
        if not candidate_cluster_heads:
            return None

        node_state = node_states[node_id]
        best_cluster = None
        best_distance = None
        for cluster_head_id in candidate_cluster_heads:
            cluster_head = node_states[cluster_head_id]
            distance = np.hypot(node_state.x - cluster_head.x, node_state.y - cluster_head.y)
            if best_distance is None or distance < best_distance:
                best_cluster = cluster_head_id
                best_distance = distance
        return best_cluster

    def should_transmit(self, context: NodeRoundContext, network_config: dict) -> bool:
        return True

    def transmission_order(self, node_ids: list[int], contexts: dict[int, NodeRoundContext]) -> list[int]:
        return sorted(node_ids)


class StandardLEACHProtocol(ProtocolBase):
    def __init__(self) -> None:
        super().__init__(
            name="standard_leach",
            short_name=protocol_label("standard_leach"),
            cluster_head_rule="probabilistic_leach",
            join_rule="nearest_cluster_head",
        )

    def select_cluster_heads(self, node_states: dict[int, NodeState], contexts: dict[int, NodeRoundContext], round_index: int, network_config: dict, rng: np.random.Generator) -> list[int]:
        eligible = [node_id for node_id, state in node_states.items() if state.alive and state.cooldown <= 0]
        if not eligible:
            return []

        p = network_config["cluster_head_ratio"]
        epoch = max(1, int(network_config["epoch_length"]))
        denominator = 1.0 - p * (round_index % epoch)
        threshold = 1.0 if denominator <= 0 else p / denominator
        selected = [node_id for node_id in eligible if rng.random() <= threshold]
        if not selected:
            selected = [max(eligible, key=lambda nid: contexts[nid].residual_energy_ratio)]
        return selected


class EnergyAwareLEACHProtocol(ProtocolBase):
    def __init__(self) -> None:
        super().__init__(
            name="energy_aware_leach",
            short_name=protocol_label("energy_aware_leach"),
            cluster_head_rule="energy_distance_ranking",
            join_rule="nearest_cluster_head",
        )

    def select_cluster_heads(self, node_states: dict[int, NodeState], contexts: dict[int, NodeRoundContext], round_index: int, network_config: dict, rng: np.random.Generator) -> list[int]:
        eligible = [node_id for node_id, state in node_states.items() if state.alive and state.cooldown <= 0]
        if not eligible:
            return []

        weights = network_config["energy_ch_weights"]
        target_count = max(1, int(round(network_config["cluster_head_ratio"] * len(eligible))))
        ranked = sorted(
            eligible,
            key=lambda nid: (
                weights["energy"] * contexts[nid].residual_energy_ratio
                + weights["distance"] * (1.0 - contexts[nid].distance_to_sink_norm)
                + 1e-6 * rng.random()
            ),
            reverse=True,
        )
        return ranked[:target_count]


class TCNPredictivePollutionAwareLEACHProtocol(ProtocolBase):
    def __init__(
        self,
        name: str = "tcn_predictive_pollution_aware_leach",
        uses_prediction: bool = True,
        uses_aoi_term: bool = True,
        uses_suppression: bool = True,
        uses_priority_scheduler: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            short_name=protocol_label(name),
            cluster_head_rule="energy_priority_distance",
            join_rule="cost_based_priority_join",
            uses_prediction=uses_prediction,
            uses_aoi_term=uses_aoi_term,
            uses_suppression=uses_suppression,
            uses_priority_scheduler=uses_priority_scheduler,
            priority_scoring_enabled=True,
        )

    def select_cluster_heads(self, node_states: dict[int, NodeState], contexts: dict[int, NodeRoundContext], round_index: int, network_config: dict, rng: np.random.Generator) -> list[int]:
        eligible = [node_id for node_id, state in node_states.items() if state.alive and state.cooldown <= 0]
        if not eligible:
            return []

        weights = network_config["predictive_ch_weights"]
        target_count = max(1, int(round(network_config["cluster_head_ratio"] * len(eligible))))
        ranked = sorted(
            eligible,
            key=lambda nid: cluster_head_score(
                priority_score=contexts[nid].priority_score,
                residual_energy_ratio=contexts[nid].residual_energy_ratio,
                distance_to_sink_norm=contexts[nid].distance_to_sink_norm,
                weights=weights,
            )
            + 1e-6 * rng.random(),
            reverse=True,
        )
        return ranked[:target_count]

    def choose_cluster_head(
        self,
        node_id: int,
        candidate_cluster_heads: list[int],
        node_states: dict[int, NodeState],
        contexts: dict[int, NodeRoundContext],
        sink_position: tuple[float, float],
        area_diagonal: float,
        network_config: dict,
    ) -> int | None:
        if not candidate_cluster_heads:
            return None

        node_state = node_states[node_id]
        join_weights = network_config["join_weights"]
        direct_distance = np.hypot(node_state.x - sink_position[0], node_state.y - sink_position[1]) / area_diagonal
        direct_cost = direct_distance

        best_cluster = None
        best_cost = direct_cost
        for cluster_head_id in candidate_cluster_heads:
            cluster_head = node_states[cluster_head_id]
            distance = np.hypot(node_state.x - cluster_head.x, node_state.y - cluster_head.y) / area_diagonal
            cost = (
                join_weights["distance"] * distance
                + join_weights["energy"] * (1.0 - contexts[cluster_head_id].residual_energy_ratio)
                + join_weights["priority"] * (1.0 - contexts[cluster_head_id].priority_score)
            )
            if cost < best_cost:
                best_cluster = cluster_head_id
                best_cost = cost
        return best_cluster

    def should_transmit(self, context: NodeRoundContext, network_config: dict) -> bool:
        if not self.uses_suppression:
            return True
        suppression = network_config["suppression"]
        if context.current_severity == 3 or context.predicted_severity == 3:
            return True
        if context.current_severity == 2 or context.predicted_severity == 2:
            return True
        low_change = abs(context.change_rate) < suppression["change_threshold"]
        low_priority = context.priority_score < suppression["priority_threshold"]
        low_aoi = context.aoi < suppression["aoi_threshold"]
        return not (low_change and low_priority and low_aoi)

    def transmission_order(self, node_ids: list[int], contexts: dict[int, NodeRoundContext]) -> list[int]:
        if not self.uses_priority_scheduler:
            return super().transmission_order(node_ids, contexts)
        return sorted(
            node_ids,
            key=lambda node_id: (
                contexts[node_id].current_severity,
                contexts[node_id].predicted_severity,
                contexts[node_id].priority_score,
            ),
            reverse=True,
        )


def build_protocol(name: str) -> ProtocolBase:
    if name == "standard_leach":
        return StandardLEACHProtocol()
    if name == "energy_aware_leach":
        return EnergyAwareLEACHProtocol()
    if name == "tcn_predictive_pollution_aware_leach":
        return TCNPredictivePollutionAwareLEACHProtocol(name=name)
    if name == "full_tcn_ppa_leach":
        return TCNPredictivePollutionAwareLEACHProtocol(name=name)
    if name == "no_tcn_prediction":
        return TCNPredictivePollutionAwareLEACHProtocol(name=name, uses_prediction=False)
    if name == "no_aoi_term":
        return TCNPredictivePollutionAwareLEACHProtocol(name=name, uses_aoi_term=False)
    if name == "no_suppression":
        return TCNPredictivePollutionAwareLEACHProtocol(name=name, uses_suppression=False)
    if name == "no_priority_scheduler":
        return TCNPredictivePollutionAwareLEACHProtocol(name=name, uses_priority_scheduler=False)
    raise ValueError(f"Unsupported protocol: {name}")
