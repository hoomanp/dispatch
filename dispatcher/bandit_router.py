"""
DISPATCH Adaptive Contextual Bandits for Dynamic Routing (Phase 4).

Implements an online Multi-Armed Bandit (Thompson Sampling + Epsilon-Greedy)
optimizer that dynamically shifts routing traffic across eligible model tier
candidates to maximize reliability and latency while minimizing inference costs.

Reward Signal Model:
  R = success * [ w_lat * (1 - norm_lat) + w_cost * (1 - norm_cost) + w_user * rating ] - (1 - success) * penalty
"""

import logging
import math
import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("dispatch.bandit_router")

# Normalization constants
MAX_EXPECTED_LATENCY_MS = 5000.0  # 5 seconds
MAX_EXPECTED_COST_USD = 0.50


@dataclass
class ArmStats:
    model_group: str
    pulls: int = 0
    successes: int = 0
    failures: int = 0
    alpha: float = 1.0  # Thompson Sampling Beta prior
    beta: float = 1.0
    total_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    total_reward: float = 0.0
    last_pulled_at: float = field(default_factory=time.time)

    @property
    def avg_latency_ms(self) -> float:
        return round(self.total_latency_ms / max(1, self.pulls), 1)

    @property
    def avg_cost_usd(self) -> float:
        return round(self.total_cost_usd / max(1, self.pulls), 5)

    @property
    def success_rate(self) -> float:
        return round(self.successes / max(1, self.pulls), 3)

    @property
    def expected_value(self) -> float:
        return round(self.alpha / (self.alpha + self.beta), 4)


class BanditRouter:
    """Adaptive Contextual Multi-Armed Bandit Router."""

    def __init__(self, epsilon: float = 0.10, decay_rate: float = 0.995):
        self.epsilon = epsilon  # Exploration probability
        self.decay_rate = decay_rate
        self._lock = threading.Lock()
        self._arms: Dict[str, ArmStats] = {}

    def _get_arm(self, model_group: str) -> ArmStats:
        if model_group not in self._arms:
            self._arms[model_group] = ArmStats(model_group=model_group)
        return self._arms[model_group]

    def select_arm(self, candidates: List[str], strategy: str = "bandit") -> Tuple[str, Dict[str, float]]:
        """Select the optimal model group arm using Thompson Sampling or deterministic priority."""
        if not candidates:
            return "local-always-on", {}
        if len(candidates) == 1:
            return candidates[0], {candidates[0]: 1.0}

        with self._lock:
            scores: Dict[str, float] = {}

            # Epsilon exploration: randomly pick among candidates to discover shifts
            if strategy == "bandit" and random.random() < self.epsilon:
                selected = random.choice(candidates)
                logger.debug(f"Bandit exploring candidate: {selected}")
                for c in candidates:
                    arm = self._get_arm(c)
                    scores[c] = arm.expected_value
                return selected, scores

            # Thompson Sampling: sample from Beta distribution for each candidate
            sampled_values: Dict[str, float] = {}
            for c in candidates:
                arm = self._get_arm(c)
                # Sample from Beta(alpha, beta) using random.betavariate
                sample = random.betavariate(max(0.1, arm.alpha), max(0.1, arm.beta))
                sampled_values[c] = sample
                scores[c] = round(sample, 4)

            # Pick highest sampled value
            best_arm = max(sampled_values, key=sampled_values.get)
            return best_arm, scores

    def record_feedback(self, model_group: str, success: bool,
                        latency_ms: float = 0.0, cost_usd: float = 0.0,
                        user_rating: Optional[float] = None) -> float:
        """Update arm distributions based on execution feedback.
        
        Args:
            model_group: The model group chosen.
            success: Whether the request succeeded (HTTP 200).
            latency_ms: Observed latency in milliseconds.
            cost_usd: Estimated or actual cost in USD.
            user_rating: Optional normalized user satisfaction (0.0 to 1.0).
            
        Returns:
            Calculated reward score (-1.0 to 1.0).
        """
        # Calculate composite normalized reward in [0.0, 1.0]
        norm_lat = min(1.0, max(0.0, latency_ms / MAX_EXPECTED_LATENCY_MS))
        norm_cost = min(1.0, max(0.0, cost_usd / MAX_EXPECTED_COST_USD))

        if success:
            # Latency weight (40%), Cost efficiency weight (40%), User satisfaction (20%)
            lat_score = 1.0 - norm_lat
            cost_score = 1.0 - norm_cost
            rating_score = user_rating if user_rating is not None else 0.8
            reward = (0.40 * lat_score) + (0.40 * cost_score) + (0.20 * rating_score)
            reward = max(0.05, min(1.0, reward))
        else:
            reward = 0.0  # Failure penalty

        with self._lock:
            arm = self._get_arm(model_group)
            arm.pulls += 1
            arm.last_pulled_at = time.time()
            arm.total_latency_ms += latency_ms
            arm.total_cost_usd += cost_usd
            arm.total_reward += reward

            if success:
                arm.successes += 1
                arm.alpha += reward * 1.5
                arm.beta += (1.0 - reward) * 0.5
            else:
                arm.failures += 1
                arm.beta += 3.0  # Penalize failures heavily

        return round(reward, 4)

    def stats(self) -> Dict[str, dict]:
        """Return comprehensive telemetry and performance stats for all arms."""
        with self._lock:
            return {
                name: {
                    "pulls": arm.pulls,
                    "successes": arm.successes,
                    "failures": arm.failures,
                    "success_rate": arm.success_rate,
                    "avg_latency_ms": arm.avg_latency_ms,
                    "avg_cost_usd": arm.avg_cost_usd,
                    "expected_value": arm.expected_value,
                    "alpha": round(arm.alpha, 2),
                    "beta": round(arm.beta, 2),
                }
                for name, arm in self._arms.items()
            }


bandit_router = BanditRouter()
