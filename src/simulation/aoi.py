from __future__ import annotations


class AoITracker:
    def __init__(self, node_ids: list[int]) -> None:
        self.ages = {node_id: 0 for node_id in node_ids}

    def update(self, delivered_node_ids: set[int]) -> None:
        for node_id in self.ages:
            self.ages[node_id] += 1
        for node_id in delivered_node_ids:
            if node_id in self.ages:
                self.ages[node_id] = 0

    def get(self, node_id: int) -> int:
        return self.ages[node_id]

    def average(self) -> float:
        if not self.ages:
            return 0.0
        return sum(self.ages.values()) / len(self.ages)
