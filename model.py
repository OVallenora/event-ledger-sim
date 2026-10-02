import math
import random
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class TransactionEvent:
    sender_id: str
    recipient_id: str
    amount: float
    latitude: float
    longitude: float


class SpatialRiskDetector:
    """
    K-means-based spatial partitioner to detect anomalous transaction dispatch coordinates.
    """
    def __init__(self, k: int = 2, max_radius_deg: float = 0.5):
        self.k = k
        self.max_radius_deg = max_radius_deg
        self.centroids: List[Tuple[float, float]] = []

    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def fit(self, historical_points: List[Tuple[float, float]]) -> None:
        if not historical_points:
            raise ValueError("Training coordinates cannot be empty.")
            
        # Initialize centroids deterministically
        self.centroids = historical_points[: self.k]

        # Iterative mean updates
        for _ in range(10):
            clusters: List[List[Tuple[float, float]]] = [[] for _ in self.centroids]
            for point in historical_points:
                dists = [self._distance(point, c) for c in self.centroids]
                nearest_idx = dists.index(min(dists))
                clusters[nearest_idx].append(point)

            for i, cluster in enumerate(clusters):
                if cluster:
                    mean_lat = sum(p[0] for p in cluster) / len(cluster)
                    mean_lon = sum(p[1] for p in cluster) / len(cluster)
                    self.centroids[i] = (mean_lat, mean_lon)

    def is_anomalous(self, lat: float, lon: float) -> bool:
        if not self.centroids:
            return False
        min_dist = min(self._distance((lat, lon), c) for c in self.centroids)
        return min_dist > self.max_radius_deg


class MonteCarloRiskEngine:
    """
    Evaluates gross portfolio loss using Monte Carlo simulation loops.
    """
    def __init__(self, simulation_loops: int = 1000):
        self.simulation_loops = simulation_loops

    def compute_gross_weighted_loss(
        self,
        flagged_events: List[TransactionEvent],
        default_fraud_prob: float = 0.35
    ) -> float:
        """
        Runs Monte Carlo iterations over anomalous transactions to estimate expected portfolio loss.
        """
        if not flagged_events:
            return 0.0

        total_simulated_loss = 0.0

        for _ in range(self.simulation_loops):
            loop_loss = 0.0
            for event in flagged_events:
                # Stochastic default event based on spatial risk weight
                if random.random() < default_fraud_prob:
                    loop_loss += event.amount
            total_simulated_loss += loop_loss

        return round(total_simulated_loss / self.simulation_loops, 2)
