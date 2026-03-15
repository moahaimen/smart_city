from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines.protocols import build_protocol
from src.data.pollution_data import ScenarioData
from src.models.tcn_predictor import PredictorBundle
from src.simulation.aoi import AoITracker
from src.simulation.priority import PriorityInputs, compute_priority_score
from src.simulation.severity import map_pm25_to_severity
from src.simulation.types import NodeRoundContext, NodeState


def _distance_threshold(network_config: dict) -> float:
    return math.sqrt(network_config["fs_energy"] / network_config["mp_energy"])


def _tx_energy(bits: int, distance: float, network_config: dict) -> float:
    if distance > _distance_threshold(network_config):
        return network_config["tx_energy"] * bits + network_config["mp_energy"] * bits * (distance ** 4)
    return network_config["tx_energy"] * bits + network_config["fs_energy"] * bits * (distance ** 2)


def _rx_energy(bits: int, network_config: dict, aggregate: bool) -> float:
    energy = network_config["rx_energy"] * bits
    if aggregate:
        energy += network_config["aggregation_energy"] * bits
    return energy


def _consume_energy(node_state: NodeState, amount: float) -> bool:
    if node_state.energy <= 0.0:
        return False
    if node_state.energy < amount:
        node_state.energy = 0.0
        return False
    node_state.energy -= amount
    return True


def _send_control(sender: NodeState, receiver: NodeState, network_config: dict) -> bool:
    bits = int(network_config["control_packet_bits"])
    distance = float(np.hypot(sender.x - receiver.x, sender.y - receiver.y))
    if not _consume_energy(sender, _tx_energy(bits, distance, network_config)):
        return False
    return _consume_energy(receiver, _rx_energy(bits, network_config, aggregate=False))


def _broadcast_cluster_head(sender: NodeState, receivers: list[NodeState], network_config: dict) -> None:
    if not receivers:
        return
    bits = int(network_config["control_packet_bits"])
    max_distance = max(float(np.hypot(sender.x - receiver.x, sender.y - receiver.y)) for receiver in receivers)
    if not _consume_energy(sender, _tx_energy(bits, max_distance, network_config)):
        return
    for receiver in receivers:
        if receiver.alive:
            _consume_energy(receiver, _rx_energy(bits, network_config, aggregate=False))


def _send_data(sender: NodeState, receiver: NodeState | None, sink_position: tuple[float, float], network_config: dict, aggregate_at_receiver: bool) -> bool:
    bits = int(network_config["packet_size_bits"])
    if receiver is None:
        distance = float(np.hypot(sender.x - sink_position[0], sender.y - sink_position[1]))
    else:
        distance = float(np.hypot(sender.x - receiver.x, sender.y - receiver.y))
    if not _consume_energy(sender, _tx_energy(bits, distance, network_config)):
        return False
    if receiver is None:
        return True
    return _consume_energy(receiver, _rx_energy(bits, network_config, aggregate=aggregate_at_receiver))


def _scenario_series(scenario: ScenarioData) -> dict[int, dict[str, np.ndarray]]:
    series = {}
    grouped = scenario.frame.sort_values(["node_id", "step"]).groupby("node_id", sort=False)
    for node_id, group in grouped:
        series[int(node_id)] = {
            "features": group[scenario.feature_columns].to_numpy(dtype=np.float32),
            "pm25": group[scenario.target_column].to_numpy(dtype=np.float32),
        }
    return series


def _build_contexts(
    node_states: dict[int, NodeState],
    series: dict[int, dict[str, np.ndarray]],
    predictor: PredictorBundle,
    scenario: ScenarioData,
    round_index: int,
    priority_config: dict,
    severity_thresholds: dict,
    aoi_tracker: AoITracker,
) -> dict[int, NodeRoundContext]:
    active_node_ids = [node_id for node_id, state in node_states.items() if state.alive]
    windows = []
    for node_id in active_node_ids:
        features = series[node_id]["features"][round_index : round_index + predictor.window_size]
        windows.append(features)

    predicted_pm25 = predictor.predict(np.asarray(windows, dtype=np.float32))
    sink_position = (scenario.area_size / 2.0, scenario.area_size / 2.0)
    area_diagonal = math.sqrt(2.0) * scenario.area_size

    contexts: dict[int, NodeRoundContext] = {}
    for idx, node_id in enumerate(active_node_ids):
        state = node_states[node_id]
        current_index = round_index + predictor.window_size - 1
        current_pm25 = float(series[node_id]["pm25"][current_index])
        previous_pm25 = float(series[node_id]["pm25"][max(current_index - 1, 0)])
        change_rate = current_pm25 - previous_pm25
        distance_norm = float(np.hypot(state.x - sink_position[0], state.y - sink_position[1]) / area_diagonal)
        current_severity = int(map_pm25_to_severity(current_pm25, severity_thresholds))
        predicted_severity = int(map_pm25_to_severity(float(predicted_pm25[idx]), severity_thresholds))
        priority_score = compute_priority_score(
            PriorityInputs(
                current_severity=current_severity,
                predicted_severity=predicted_severity,
                aoi=float(aoi_tracker.get(node_id)),
                change_rate=float(change_rate),
                hotspot_relevance=float(state.hotspot_relevance),
                communication_cost=distance_norm,
            ),
            weight_config=priority_config["weights"],
            normalization_config=priority_config["normalization"],
        )
        contexts[node_id] = NodeRoundContext(
            node_id=node_id,
            current_pm25=current_pm25,
            predicted_pm25=float(predicted_pm25[idx]),
            current_severity=current_severity,
            predicted_severity=predicted_severity,
            residual_energy_ratio=float(max(state.energy, 0.0)),
            aoi=float(aoi_tracker.get(node_id)),
            change_rate=float(change_rate),
            hotspot_relevance=float(state.hotspot_relevance),
            communication_cost_norm=distance_norm,
            distance_to_sink_norm=distance_norm,
            priority_score=priority_score,
        )
    return contexts


def _initialize_nodes(scenario: ScenarioData, initial_energy: float) -> dict[int, NodeState]:
    nodes = {}
    for row in scenario.node_metadata.itertuples(index=False):
        nodes[int(row.node_id)] = NodeState(
            node_id=int(row.node_id),
            x=float(row.x),
            y=float(row.y),
            hotspot_relevance=float(row.hotspot_relevance),
            energy=float(initial_energy),
        )
    return nodes


def run_protocol_simulation(
    scenario: ScenarioData,
    predictor: PredictorBundle,
    config: dict,
    protocol_name: str,
    seed: int,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    protocol = build_protocol(protocol_name)
    network_config = copy.deepcopy(config["network"])
    network_config.update(scenario.network_overrides)
    initial_energy = float(network_config["initial_energy"])
    area_diagonal = math.sqrt(2.0) * scenario.area_size
    radio_range = float(network_config["radio_range_factor"]) * area_diagonal
    severity_thresholds = config["severity"]["pm25_thresholds"]
    sink_position = (scenario.area_size / 2.0, scenario.area_size / 2.0)

    node_states = _initialize_nodes(scenario, initial_energy)
    node_series = _scenario_series(scenario)
    aoi_tracker = AoITracker(list(node_states))
    rng = np.random.default_rng(seed)

    total_generated = 0
    total_delivered = 0
    total_delay = 0.0
    total_delay_count = 0
    total_hazardous = 0
    total_hazardous_delivered = 0
    total_bits_received = 0
    round_rows: list[dict[str, float | int | str]] = []

    for round_index in range(scenario.rounds):
        for node_state in node_states.values():
            node_state.cluster_head_id = None

        contexts = _build_contexts(
            node_states=node_states,
            series=node_series,
            predictor=predictor,
            scenario=scenario,
            round_index=round_index,
            priority_config=config["priority"],
            severity_thresholds=severity_thresholds,
            aoi_tracker=aoi_tracker,
        )
        if not contexts:
            break

        cluster_heads = protocol.select_cluster_heads(node_states, contexts, round_index, network_config, rng)
        for cluster_head_id in cluster_heads:
            if cluster_head_id in node_states and node_states[cluster_head_id].alive:
                node_states[cluster_head_id].cluster_head_id = cluster_head_id
                node_states[cluster_head_id].cooldown = int(network_config["epoch_length"])

        alive_receivers = [state for node_id, state in node_states.items() if state.alive]
        for cluster_head_id in cluster_heads:
            sender = node_states[cluster_head_id]
            receivers = [
                receiver
                for receiver in alive_receivers
                if receiver.node_id != cluster_head_id
                and float(np.hypot(sender.x - receiver.x, sender.y - receiver.y)) <= radio_range
            ]
            _broadcast_cluster_head(sender, receivers, network_config)

        for node_id, node_state in node_states.items():
            if not node_state.alive or node_id in cluster_heads:
                continue
            chosen_cluster = protocol.choose_cluster_head(
                node_id=node_id,
                candidate_cluster_heads=[cluster_head_id for cluster_head_id in cluster_heads if node_states[cluster_head_id].alive],
                node_states=node_states,
                contexts=contexts,
                sink_position=sink_position,
                area_diagonal=area_diagonal,
                network_config=network_config,
            )
            if chosen_cluster is None:
                continue
            distance = float(np.hypot(node_state.x - node_states[chosen_cluster].x, node_state.y - node_states[chosen_cluster].y))
            if distance > radio_range:
                continue
            if _send_control(node_state, node_states[chosen_cluster], network_config):
                node_state.cluster_head_id = chosen_cluster

        delivered_node_ids: set[int] = set()
        pending_forward: dict[int, list[tuple[int, int]]] = {cluster_head_id: [] for cluster_head_id in cluster_heads}
        generated_this_round = 0
        delivered_this_round = 0
        delay_this_round: list[int] = []
        hazardous_total_round = 0
        hazardous_delivered_round = 0

        ordered_nodes = protocol.transmission_order(
            [node_id for node_id, state in node_states.items() if state.alive],
            contexts,
        )
        for node_id in ordered_nodes:
            context = contexts[node_id]
            sender = node_states[node_id]
            if not sender.alive:
                continue
            if not protocol.should_transmit(context, network_config):
                continue

            generated_this_round += 1
            total_generated += 1
            hazardous_packet = context.current_severity == 3
            if hazardous_packet:
                hazardous_total_round += 1
                total_hazardous += 1

            cluster_head_id = sender.cluster_head_id
            if cluster_head_id is None:
                if _send_data(sender, None, sink_position, network_config, aggregate_at_receiver=False):
                    delivered_this_round += 1
                    total_delivered += 1
                    total_bits_received += int(network_config["packet_size_bits"])
                    delay_this_round.append(1)
                    total_delay += 1.0
                    total_delay_count += 1
                    delivered_node_ids.add(node_id)
                    if hazardous_packet:
                        hazardous_delivered_round += 1
                        total_hazardous_delivered += 1
                continue

            if cluster_head_id == node_id:
                pending_forward.setdefault(cluster_head_id, []).append((node_id, 1))
                continue

            receiver = node_states[cluster_head_id]
            if not receiver.alive:
                continue
            if _send_data(sender, receiver, sink_position, network_config, aggregate_at_receiver=True):
                pending_forward.setdefault(cluster_head_id, []).append((node_id, 2))

        for cluster_head_id, queue in pending_forward.items():
            cluster_head = node_states[cluster_head_id]
            if not cluster_head.alive:
                continue
            forward_order = protocol.transmission_order([source_id for source_id, _ in queue], contexts)
            ordered_queue = sorted(queue, key=lambda item: forward_order.index(item[0]))
            for source_id, hop_count in ordered_queue:
                if not cluster_head.alive:
                    break
                if _send_data(cluster_head, None, sink_position, network_config, aggregate_at_receiver=False):
                    delivered_this_round += 1
                    total_delivered += 1
                    total_bits_received += int(network_config["packet_size_bits"])
                    delay_this_round.append(hop_count)
                    total_delay += float(hop_count)
                    total_delay_count += 1
                    delivered_node_ids.add(source_id)
                    if contexts[source_id].current_severity == 3:
                        hazardous_delivered_round += 1
                        total_hazardous_delivered += 1

        aoi_tracker.update(delivered_node_ids)
        for node_state in node_states.values():
            if node_state.cooldown > 0:
                node_state.cooldown -= 1

        dead_nodes = sum(1 for node_state in node_states.values() if not node_state.alive)
        alive_nodes = scenario.node_count - dead_nodes
        avg_residual_energy = float(sum(max(node_state.energy, 0.0) for node_state in node_states.values()) / scenario.node_count)
        round_rows.append(
            {
                "scenario": scenario.name,
                "scenario_group": scenario.scenario_group,
                "protocol": protocol.name,
                "round": round_index + 1,
                "cluster_heads": len(cluster_heads),
                "alive_nodes": alive_nodes,
                "dead_nodes": dead_nodes,
                "avg_residual_energy": avg_residual_energy,
                "generated_packets": generated_this_round,
                "delivered_packets": delivered_this_round,
                "pdr_round": (delivered_this_round / generated_this_round) if generated_this_round else 0.0,
                "avg_delay_round": float(np.mean(delay_this_round)) if delay_this_round else 0.0,
                "throughput_bits_round": delivered_this_round * int(network_config["packet_size_bits"]),
                "avg_aoi": aoi_tracker.average(),
                "hazardous_generated": hazardous_total_round,
                "hazardous_delivered": hazardous_delivered_round,
                "hazardous_success_round": (hazardous_delivered_round / hazardous_total_round) if hazardous_total_round else 1.0,
            }
        )
        if dead_nodes == scenario.node_count:
            break

    rounds_df = pd.DataFrame.from_records(round_rows)
    if rounds_df.empty:
        raise RuntimeError(f"No simulation rounds completed for {scenario.name} with protocol {protocol.name}.")

    fnd = int(rounds_df.loc[rounds_df["dead_nodes"] > 0, "round"].iloc[0]) if (rounds_df["dead_nodes"] > 0).any() else int(rounds_df["round"].iloc[-1])
    hnd = int(rounds_df.loc[rounds_df["dead_nodes"] >= scenario.node_count / 2.0, "round"].iloc[0]) if (rounds_df["dead_nodes"] >= scenario.node_count / 2.0).any() else int(rounds_df["round"].iloc[-1])
    lnd = int(rounds_df.loc[rounds_df["dead_nodes"] >= scenario.node_count, "round"].iloc[0]) if (rounds_df["dead_nodes"] >= scenario.node_count).any() else int(rounds_df["round"].iloc[-1])

    summary = {
        "scenario": scenario.name,
        "scenario_group": scenario.scenario_group,
        "protocol": protocol.name,
        "node_count": scenario.node_count,
        "area_size": scenario.area_size,
        "initial_energy": float(network_config["initial_energy"]),
        "rounds_completed": int(rounds_df["round"].iloc[-1]),
        "fnd": fnd,
        "hnd": hnd,
        "lnd": lnd,
        "average_residual_energy": float(rounds_df["avg_residual_energy"].mean()),
        "packet_delivery_ratio": float(total_delivered / total_generated) if total_generated else 0.0,
        "end_to_end_delay": float(total_delay / total_delay_count) if total_delay_count else 0.0,
        "throughput_bits_per_round": float(total_bits_received / len(rounds_df)),
        "average_aoi": float(rounds_df["avg_aoi"].mean()),
        "hazardous_event_delivery_success_rate": float(total_hazardous_delivered / total_hazardous) if total_hazardous else 1.0,
        "total_generated_packets": int(total_generated),
        "total_delivered_packets": int(total_delivered),
    }
    return summary, rounds_df
