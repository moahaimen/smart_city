from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import loadmat, savemat


@dataclass
class Area:
    x: float
    y: float


@dataclass
class Model:
    n: int
    sink_x: float
    sink_y: float
    p: float
    initial_energy: float
    tx_energy: float
    rx_energy: float
    fs_energy: float
    mp_energy: float
    aggregation_energy: float
    rmax: int
    data_packet_len: int
    hello_packet_len: int
    num_packets: int
    radio_range: float
    min_cluster_head_energy_fraction: float
    sink_energy: float = math.inf

    @property
    def distance_threshold(self) -> float:
        return math.sqrt(self.fs_energy / self.mp_energy)


@dataclass
class Sensor:
    sensor_id: int
    x: float
    y: float
    energy: float
    sensor_type: str = "N"
    rounds_until_eligible: int = 0
    distance_to_sink: float = math.inf
    distance_to_cluster: float = math.inf
    member_cluster: int = -1

    @property
    def alive(self) -> bool:
        return math.isinf(self.energy) or self.energy > 0.0


@dataclass
class PacketCounters:
    routing_sent: int = 0
    routing_received: int = 0
    data_sent: int = 0
    data_received: int = 0


@dataclass
class SimulationReport:
    rounds_completed: int
    first_dead_round: int
    last_round: int
    dead_nodes: np.ndarray
    cluster_heads: np.ndarray
    routing_sent: np.ndarray
    routing_received: np.ndarray
    data_sent: np.ndarray
    data_received: np.ndarray
    alive_sensors: np.ndarray
    total_sensor_energy: np.ndarray
    average_sensor_energy: np.ndarray
    average_energy_consumed: np.ndarray
    energy_variance: np.ndarray


def sensor_index(sensor_id: int) -> int:
    return sensor_id - 1


def sensor_by_id(sensors: list[Sensor], sensor_id: int) -> Sensor:
    return sensors[sensor_index(sensor_id)]


def standard_model(node_count: int, rounds: int, field_size: float) -> tuple[Area, Model]:
    area = Area(x=field_size, y=field_size)
    sink_x = 0.5 * area.x
    sink_y = 0.5 * area.y
    model = Model(
        n=node_count,
        sink_x=sink_x,
        sink_y=sink_y,
        p=0.1,
        initial_energy=0.5,
        tx_energy=50e-9,
        rx_energy=50e-9,
        fs_energy=10e-12,
        mp_energy=0.0013e-12,
        aggregation_energy=5e-9,
        rmax=rounds,
        data_packet_len=4000,
        hello_packet_len=100,
        num_packets=10,
        radio_range=0.5 * area.x * math.sqrt(2.0),
        min_cluster_head_energy_fraction=0.5,
    )
    return area, model


def apply_legacy_energy_model(model: Model) -> None:
    model.initial_energy = 10.0
    model.tx_energy = 5e-9
    model.rx_energy = 50e-9
    model.fs_energy = 5e-6
    model.mp_energy = 5e-6
    model.aggregation_energy = 5e-9


def load_locations(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix.lower() == ".mat":
        data = loadmat(path)
        if "X" not in data or "Y" not in data:
            raise ValueError(f"{path} does not contain X and Y arrays.")
        x = np.asarray(data["X"]).reshape(-1)
        y = np.asarray(data["Y"]).reshape(-1)
        return x.astype(float), y.astype(float)

    data = np.load(path)
    if "X" not in data or "Y" not in data:
        raise ValueError(f"{path} does not contain X and Y arrays.")
    return np.asarray(data["X"], dtype=float).reshape(-1), np.asarray(data["Y"], dtype=float).reshape(-1)


def save_locations(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".mat":
        savemat(path, {"X": x.reshape(1, -1), "Y": y.reshape(1, -1)})
        return

    np.savez(path, X=x, Y=y)


def create_random_locations(model: Model, area: Area, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    x = rng.uniform(0.0, area.x, size=model.n)
    y = rng.uniform(0.0, area.y, size=model.n)
    return x, y


def configure_sensors(model: Model, x: np.ndarray, y: np.ndarray) -> list[Sensor]:
    sensors: list[Sensor] = []
    sink_id = model.n + 1

    for node_id in range(1, model.n + 1):
        sensors.append(
            Sensor(
                sensor_id=node_id,
                x=float(x[node_id - 1]),
                y=float(y[node_id - 1]),
                energy=model.initial_energy,
                member_cluster=sink_id,
            )
        )

    sensors.append(
        Sensor(
            sensor_id=sink_id,
            x=model.sink_x,
            y=model.sink_y,
            energy=model.sink_energy,
            sensor_type="S",
            member_cluster=sink_id,
        )
    )
    refresh_sink_distances(sensors, model)
    return sensors


def refresh_sink_distances(sensors: list[Sensor], model: Model) -> None:
    sink = sensor_by_id(sensors, model.n + 1)
    for node_id in range(1, model.n + 1):
        sensor = sensor_by_id(sensors, node_id)
        sensor.distance_to_sink = math.hypot(sensor.x - sink.x, sensor.y - sink.y)
        sensor.distance_to_cluster = sensor.distance_to_sink
        sensor.member_cluster = model.n + 1


def reset_round_state(sensors: list[Sensor], model: Model) -> None:
    for node_id in range(1, model.n + 1):
        sensor = sensor_by_id(sensors, node_id)
        sensor.sensor_type = "N"
        sensor.distance_to_cluster = sensor.distance_to_sink
        sensor.member_cluster = model.n + 1


def decay_cluster_head_cooldown(sensors: list[Sensor], model: Model) -> None:
    for node_id in range(1, model.n + 1):
        sensor = sensor_by_id(sensors, node_id)
        if sensor.rounds_until_eligible > 0:
            sensor.rounds_until_eligible -= 1


def transmit_energy(model: Model, bits: int, distance: float) -> float:
    if distance > model.distance_threshold:
        return model.tx_energy * bits + model.mp_energy * bits * (distance ** 4)
    return model.tx_energy * bits + model.fs_energy * bits * (distance ** 2)


def receive_energy(model: Model, bits: int, aggregate: bool) -> float:
    energy = model.rx_energy * bits
    if aggregate:
        energy += model.aggregation_energy * bits
    return energy


def consume_energy(sensor: Sensor, amount: float) -> bool:
    if math.isinf(sensor.energy):
        return True
    if sensor.energy <= 0.0:
        return False
    if sensor.energy < amount:
        sensor.energy = 0.0
        return False
    sensor.energy -= amount
    return True


def deliver_unicast(
    sensors: list[Sensor],
    model: Model,
    sender_id: int,
    receiver_id: int,
    packet_type: str,
    counters: PacketCounters,
    aggregate_at_receiver: bool = False,
) -> bool:
    sender = sensor_by_id(sensors, sender_id)
    receiver = sensor_by_id(sensors, receiver_id)
    if not sender.alive or (not receiver.alive and receiver_id != model.n + 1):
        return False

    bits = model.hello_packet_len if packet_type == "Hello" else model.data_packet_len
    distance = math.hypot(sender.x - receiver.x, sender.y - receiver.y)
    if not consume_energy(sender, transmit_energy(model, bits, distance)):
        return False

    if packet_type == "Hello":
        counters.routing_sent += 1
    else:
        counters.data_sent += 1

    if receiver_id == model.n + 1:
        if packet_type == "Hello":
            counters.routing_received += 1
        else:
            counters.data_received += 1
        return True

    if not consume_energy(receiver, receive_energy(model, bits, aggregate_at_receiver)):
        return False

    if packet_type == "Hello":
        counters.routing_received += 1
    else:
        counters.data_received += 1
    return True


def deliver_broadcast(
    sensors: list[Sensor],
    model: Model,
    sender_id: int,
    receiver_ids: Iterable[int],
    packet_type: str,
    counters: PacketCounters,
) -> int:
    sender = sensor_by_id(sensors, sender_id)
    alive_receivers = [rid for rid in receiver_ids if sensor_by_id(sensors, rid).alive]
    if not sender.alive or not alive_receivers:
        return 0

    bits = model.hello_packet_len if packet_type == "Hello" else model.data_packet_len
    distances = [
        math.hypot(sender.x - sensor_by_id(sensors, rid).x, sender.y - sensor_by_id(sensors, rid).y)
        for rid in alive_receivers
    ]
    if not consume_energy(sender, transmit_energy(model, bits, max(distances))):
        return 0

    if packet_type == "Hello":
        counters.routing_sent += 1
    else:
        counters.data_sent += 1

    delivered = 0
    for rid in alive_receivers:
        receiver = sensor_by_id(sensors, rid)
        if consume_energy(receiver, receive_energy(model, bits, aggregate=False)):
            delivered += 1

    if packet_type == "Hello":
        counters.routing_received += delivered
    else:
        counters.data_received += delivered
    return delivered


def select_cluster_heads(
    sensors: list[Sensor],
    model: Model,
    round_index: int,
    rng: np.random.Generator,
) -> list[int]:
    epoch = max(1, round(1.0 / model.p))
    denominator = 1.0 - model.p * (round_index % epoch)
    threshold = 1.0 if denominator <= 0 else model.p / denominator
    min_energy = model.initial_energy * model.min_cluster_head_energy_fraction
    cluster_heads: list[int] = []

    for node_id in range(1, model.n + 1):
        sensor = sensor_by_id(sensors, node_id)
        if not sensor.alive or sensor.energy < min_energy:
            continue
        if sensor.rounds_until_eligible > 0:
            continue
        if rng.random() <= threshold:
            sensor.sensor_type = "C"
            sensor.rounds_until_eligible = epoch
            sensor.member_cluster = sensor.sensor_id
            sensor.distance_to_cluster = 0.0
            cluster_heads.append(sensor.sensor_id)

    return cluster_heads


def find_receivers(sensors: list[Sensor], model: Model, sender_id: int, radius: float) -> list[int]:
    sender = sensor_by_id(sensors, sender_id)
    receivers: list[int] = []

    for node_id in range(1, model.n + 1):
        if node_id == sender_id:
            continue
        receiver = sensor_by_id(sensors, node_id)
        if not receiver.alive:
            continue
        distance = math.hypot(receiver.x - sender.x, receiver.y - sender.y)
        if distance <= radius:
            receivers.append(node_id)

    return receivers


def join_to_nearest_cluster_head(sensors: list[Sensor], model: Model, cluster_heads: list[int]) -> None:
    if not cluster_heads:
        return

    for node_id in range(1, model.n + 1):
        sensor = sensor_by_id(sensors, node_id)
        if not sensor.alive:
            continue
        if sensor.sensor_type == "C":
            sensor.member_cluster = sensor.sensor_id
            sensor.distance_to_cluster = 0.0
            continue

        best_cluster = model.n + 1
        best_distance = sensor.distance_to_sink
        for cluster_head_id in cluster_heads:
            cluster_head = sensor_by_id(sensors, cluster_head_id)
            if not cluster_head.alive:
                continue
            distance = math.hypot(sensor.x - cluster_head.x, sensor.y - cluster_head.y)
            if distance <= model.radio_range and distance < best_distance:
                best_cluster = cluster_head_id
                best_distance = distance

        sensor.member_cluster = best_cluster
        sensor.distance_to_cluster = best_distance


def find_senders(sensors: list[Sensor], model: Model, cluster_head_id: int) -> list[int]:
    senders: list[int] = []
    for node_id in range(1, model.n + 1):
        sensor = sensor_by_id(sensors, node_id)
        if not sensor.alive:
            continue
        if sensor.member_cluster == cluster_head_id and sensor.sensor_id != cluster_head_id:
            senders.append(sensor.sensor_id)
    return senders


def count_dead_nodes(sensors: list[Sensor], model: Model) -> int:
    return sum(1 for node_id in range(1, model.n + 1) if not sensor_by_id(sensors, node_id).alive)


def energy_statistics(sensors: list[Sensor], model: Model, initial_total_energy: float) -> tuple[int, float, float, float, float]:
    alive_energies = [sensor_by_id(sensors, node_id).energy for node_id in range(1, model.n + 1) if sensor_by_id(sensors, node_id).alive]
    alive = len(alive_energies)
    total_energy = float(sum(alive_energies))
    average_energy = total_energy / alive if alive else 0.0
    average_consumed = (initial_total_energy - total_energy) / model.n
    variance = float(np.var(alive_energies)) if alive else 0.0
    return alive, total_energy, average_energy, average_consumed, variance


def plot_network(
    sensors: list[Sensor],
    model: Model,
    round_index: int,
    dead_nodes: int,
    output_path: Path | None = None,
    show_plot: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 7))
    sink = sensor_by_id(sensors, model.n + 1)

    for node_id in range(1, model.n + 1):
        sensor = sensor_by_id(sensors, node_id)
        if not sensor.alive:
            plt.plot(sensor.x, sensor.y, "r.", markersize=6)
            continue

        if sensor.sensor_type == "C":
            plt.plot(sensor.x, sensor.y, "kx", markersize=8)
        else:
            plt.plot(sensor.x, sensor.y, "bo", markersize=4)

        if sensor.member_cluster not in (sensor.sensor_id, model.n + 1):
            cluster_head = sensor_by_id(sensors, sensor.member_cluster)
            plt.plot([sensor.x, cluster_head.x], [sensor.y, cluster_head.y], "k-", linewidth=0.4, alpha=0.3)

    plt.plot(sink.x, sink.y, "g*", markersize=14)
    plt.title(f"Round={round_index}, Dead nodes={dead_nodes}")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.axis("square")
    plt.xlim(0, max(model.sink_x * 2.0, max(sensor.x for sensor in sensors[:-1]) + 5.0))
    plt.ylim(0, max(model.sink_y * 2.0, max(sensor.y for sensor in sensors[:-1]) + 5.0))
    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200)
    if show_plot:
        plt.show()
    plt.close()


def run_simulation(
    area: Area,
    model: Model,
    sensors: list[Sensor],
    final_figure: Path | None,
    show_plot: bool,
    rng: np.random.Generator,
) -> SimulationReport:
    initial_total_energy = model.n * model.initial_energy
    dead_nodes = np.zeros(model.rmax + 1, dtype=int)
    cluster_heads = np.zeros(model.rmax + 1, dtype=int)
    routing_sent = np.zeros(model.rmax + 1, dtype=int)
    routing_received = np.zeros(model.rmax + 1, dtype=int)
    data_sent = np.zeros(model.rmax + 1, dtype=int)
    data_received = np.zeros(model.rmax + 1, dtype=int)
    alive_sensors = np.zeros(model.rmax + 1, dtype=int)
    total_sensor_energy = np.zeros(model.rmax + 1, dtype=float)
    average_sensor_energy = np.zeros(model.rmax + 1, dtype=float)
    average_energy_consumed = np.zeros(model.rmax + 1, dtype=float)
    energy_variance = np.zeros(model.rmax + 1, dtype=float)

    initial_counters = PacketCounters()
    deliver_broadcast(
        sensors=sensors,
        model=model,
        sender_id=model.n + 1,
        receiver_ids=range(1, model.n + 1),
        packet_type="Hello",
        counters=initial_counters,
    )

    alive, total_energy, average_energy, average_consumed, variance = energy_statistics(
        sensors, model, initial_total_energy
    )
    routing_sent[0] = initial_counters.routing_sent
    routing_received[0] = initial_counters.routing_received
    data_sent[0] = initial_counters.data_sent
    data_received[0] = initial_counters.data_received
    dead_nodes[0] = count_dead_nodes(sensors, model)
    alive_sensors[0] = alive
    total_sensor_energy[0] = total_energy
    average_sensor_energy[0] = average_energy
    average_energy_consumed[0] = average_consumed
    energy_variance[0] = variance

    first_dead_round = 0
    last_round = 0

    for round_index in range(1, model.rmax + 1):
        reset_round_state(sensors, model)
        decay_cluster_head_cooldown(sensors, model)
        counters = PacketCounters()

        elected = select_cluster_heads(sensors, model, round_index, rng)
        cluster_heads[round_index] = len(elected)

        for cluster_head_id in elected:
            receivers = find_receivers(sensors, model, cluster_head_id, model.radio_range)
            deliver_broadcast(sensors, model, cluster_head_id, receivers, "Hello", counters)

        join_to_nearest_cluster_head(sensors, model, elected)

        for _ in range(model.num_packets):
            for cluster_head_id in elected:
                senders = find_senders(sensors, model, cluster_head_id)
                for sender_id in senders:
                    deliver_unicast(
                        sensors,
                        model,
                        sender_id,
                        cluster_head_id,
                        "Data",
                        counters,
                        aggregate_at_receiver=True,
                    )

        for cluster_head_id in elected:
            deliver_unicast(sensors, model, cluster_head_id, model.n + 1, "Data", counters)

        for node_id in range(1, model.n + 1):
            sensor = sensor_by_id(sensors, node_id)
            if sensor.alive and sensor.member_cluster == model.n + 1:
                deliver_unicast(sensors, model, node_id, model.n + 1, "Data", counters)

        round_dead_nodes = count_dead_nodes(sensors, model)
        if round_dead_nodes > 0 and first_dead_round == 0:
            first_dead_round = round_index

        alive, total_energy, average_energy, average_consumed, variance = energy_statistics(
            sensors, model, initial_total_energy
        )
        dead_nodes[round_index] = round_dead_nodes
        routing_sent[round_index] = counters.routing_sent
        routing_received[round_index] = counters.routing_received
        data_sent[round_index] = counters.data_sent
        data_received[round_index] = counters.data_received
        alive_sensors[round_index] = alive
        total_sensor_energy[round_index] = total_energy
        average_sensor_energy[round_index] = average_energy
        average_energy_consumed[round_index] = average_consumed
        energy_variance[round_index] = variance
        last_round = round_index

        if round_dead_nodes == model.n:
            break

    if final_figure is not None or show_plot:
        plot_network(
            sensors=sensors,
            model=model,
            round_index=last_round,
            dead_nodes=dead_nodes[last_round],
            output_path=final_figure,
            show_plot=show_plot,
        )

    end_index = last_round + 1
    return SimulationReport(
        rounds_completed=last_round,
        first_dead_round=first_dead_round,
        last_round=last_round,
        dead_nodes=dead_nodes[:end_index],
        cluster_heads=cluster_heads[:end_index],
        routing_sent=routing_sent[:end_index],
        routing_received=routing_received[:end_index],
        data_sent=data_sent[:end_index],
        data_received=data_received[:end_index],
        alive_sensors=alive_sensors[:end_index],
        total_sensor_energy=total_sensor_energy[:end_index],
        average_sensor_energy=average_sensor_energy[:end_index],
        average_energy_consumed=average_energy_consumed[:end_index],
        energy_variance=energy_variance[:end_index],
    )


def save_report(
    output_path: Path,
    report: SimulationReport,
    area: Area,
    model: Model,
    sensors: list[Sensor],
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> None:
    node_x = np.array([sensor.x for sensor in sensors[:-1]], dtype=float)
    node_y = np.array([sensor.y for sensor in sensors[:-1]], dtype=float)
    node_energy = np.array([sensor.energy for sensor in sensors[:-1]], dtype=float)
    node_types = np.array([sensor.sensor_type for sensor in sensors[:-1]], dtype=object)
    node_cluster = np.array([sensor.member_cluster for sensor in sensors[:-1]], dtype=int)

    payload = {
        "Area": {"x": area.x, "y": area.y},
        "Model": {
            "n": model.n,
            "Sinkx": model.sink_x,
            "Sinky": model.sink_y,
            "p": model.p,
            "Eo": model.initial_energy,
            "ETX": model.tx_energy,
            "ERX": model.rx_energy,
            "Efs": model.fs_energy,
            "Emp": model.mp_energy,
            "EDA": model.aggregation_energy,
            "do": model.distance_threshold,
            "rmax": model.rmax,
            "DpacketLen": model.data_packet_len,
            "HpacketLen": model.hello_packet_len,
            "NumPacket": model.num_packets,
            "RR": model.radio_range,
        },
        "X": x.reshape(1, -1),
        "Y": y.reshape(1, -1),
        "seed": np.array([[seed]], dtype=int),
        "first_dead_round": np.array([[report.first_dead_round]], dtype=int),
        "last_round": np.array([[report.last_round]], dtype=int),
        "dead_nodes": report.dead_nodes.reshape(1, -1),
        "cluster_heads": report.cluster_heads.reshape(1, -1),
        "routing_sent": report.routing_sent.reshape(1, -1),
        "routing_received": report.routing_received.reshape(1, -1),
        "data_sent": report.data_sent.reshape(1, -1),
        "data_received": report.data_received.reshape(1, -1),
        "alive_sensors": report.alive_sensors.reshape(1, -1),
        "total_sensor_energy": report.total_sensor_energy.reshape(1, -1),
        "average_sensor_energy": report.average_sensor_energy.reshape(1, -1),
        "average_energy_consumed": report.average_energy_consumed.reshape(1, -1),
        "energy_variance": report.energy_variance.reshape(1, -1),
        "sensor_x": node_x.reshape(1, -1),
        "sensor_y": node_y.reshape(1, -1),
        "sensor_energy": node_energy.reshape(1, -1),
        "sensor_type": node_types.reshape(1, -1),
        "sensor_cluster": node_cluster.reshape(1, -1),
        "sink_energy": np.array([[sensor_by_id(sensors, model.n + 1).energy]], dtype=float),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".mat":
        savemat(output_path, payload)
        return

    np.savez(output_path, **payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LEACH wireless sensor network simulator in Python.")
    parser.add_argument("--nodes", type=int, default=150, help="Number of sensor nodes.")
    parser.add_argument("--rounds", type=int, default=5000, help="Maximum number of rounds.")
    parser.add_argument(
        "--field-size",
        type=float,
        default=None,
        help="Square field side length in meters. Defaults to the number of nodes, matching the MATLAB code.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducible layouts and CH election.")
    parser.add_argument("--locations", type=Path, default=None, help="Optional .mat or .npz file containing X and Y.")
    parser.add_argument("--save-locations", type=Path, default=None, help="Optional path to save generated X and Y.")
    parser.add_argument("--report", type=Path, default=Path("leach_python.mat"), help="Output report path (.mat or .npz).")
    parser.add_argument("--figure", type=Path, default=None, help="Optional path to save the final network plot.")
    parser.add_argument("--plot", action="store_true", help="Display the final network plot.")
    parser.add_argument(
        "--legacy-energy-model",
        action="store_true",
        help="Use the original MATLAB energy constants, even though they collapse the network very quickly.",
    )
    parser.add_argument(
        "--min-ch-energy-fraction",
        type=float,
        default=0.5,
        help="Minimum remaining energy, as a fraction of initial energy, for CH eligibility.",
    )
    parser.add_argument("--num-packets", type=int, default=10, help="Packets per round during steady state.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    field_size = float(args.field_size) if args.field_size is not None else float(args.nodes)
    area, model = standard_model(args.nodes, args.rounds, field_size)
    model.num_packets = args.num_packets
    model.min_cluster_head_energy_fraction = args.min_ch_energy_fraction
    if args.legacy_energy_model:
        apply_legacy_energy_model(model)

    rng = np.random.default_rng(args.seed)
    if args.locations is not None:
        x, y = load_locations(args.locations)
        if len(x) != model.n or len(y) != model.n:
            raise ValueError(
                f"Location count mismatch: expected {model.n} nodes, got {len(x)} X values and {len(y)} Y values."
            )
    else:
        x, y = create_random_locations(model, area, rng)
        if args.save_locations is not None:
            save_locations(args.save_locations, x, y)

    sensors = configure_sensors(model, x, y)
    report = run_simulation(area, model, sensors, args.figure, args.plot, rng)
    save_report(args.report, report, area, model, sensors, x, y, args.seed)

    final_alive = int(report.alive_sensors[-1])
    final_energy = float(report.total_sensor_energy[-1])
    print(f"Saved report to {args.report}")
    print(f"Rounds completed: {report.rounds_completed}")
    print(f"First dead round: {report.first_dead_round}")
    print(f"Alive sensors at end: {final_alive}/{model.n}")
    print(f"Total remaining node energy: {final_energy:.6f} J")


if __name__ == "__main__":
    main()
