"""
DISPATCH Adaptive Contextual Bandits for Dynamic Routing (Phase 4).

Implements an online Contextual Multi-Armed Bandit (Discounted Thompson Sampling
+ Epsilon-Greedy) optimizer that dynamically shifts routing traffic across
eligible model tier candidates conditioned on task context to maximize
reliability and latency while minimizing inference costs.

Algorithmic Design:
  1. Tier-Aware Informative Priors: Initializes Beta(alpha_0, beta_0) with a
     local-first / cost-aware prior mean so cold-start sampling respects the
     4-tier cost hierarchy unless empirical feedback proves otherwise.
  2. Discounted Thompson Sampling (dTS): Applies exponential discounting (gamma)
     to posterior parameters on each update to track non-stationary provider
     brownouts, rate-limit waves, and latency drift.
  3. Contextual Conditioning: Tracks per-(task, model_group) posteriors alongside
     global marginal arm posteriors for empirical Bayes warm-starting.
"""

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("dispatch.bandit_router")

# Normalization constants
MAX_EXPECTED_LATENCY_MS = 5000.0  # 5 seconds
MAX_EXPECTED_COST_USD = 0.50

# Tier-aware informative Beta priors (alpha_0, beta_0)
TIER_PRIORS: Dict[str, Tuple[float, float]] = {
    "local":   (2.4, 1.0),  # Prior E[r] = 0.706
    "free":    (2.1, 1.0),  # Prior E[r] = 0.677
    "budget":  (1.8, 1.2),  # Prior E[r] = 0.600
    "premium": (1.5, 1.5),  # Prior E[r] = 0.500
}


def _prior_for_group(model_group: str) -> Tuple[float, float]:
    prefix = model_group.split("-")[0] if "-" in model_group else model_group
    return TIER_PRIORS.get(prefix, (1.5, 1.2))


@dataclass
class ArmStats:
    model_group: str
    pulls: int = 0
    successes: int = 0
    failures: int = 0
    alpha: float = 1.5
    beta: float = 1.2
    alpha0: float = 1.5
    beta0: float = 1.2
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
    def mean_reward(self) -> float:
        return round(self.total_reward / max(1, self.pulls), 4)

    @property
    def expected_value(self) -> float:
        return round(self.alpha / (self.alpha + self.beta), 4)


class BanditRouter:
    """Adaptive Contextual Multi-Armed Bandit Router with Discounted Thompson Sampling."""

    def __init__(self, epsilon: float = 0.10, decay_rate: float = 0.99):
        self.epsilon = epsilon  # Exploration probability
        self.decay_rate = decay_rate  # Non-stationary posterior discount factor (gamma)
        self._lock = threading.Lock()
        self._arms: Dict[str, ArmStats] = {}
        self._contextual_arms: Dict[Tuple[str, str], ArmStats] = {}

    def _get_arm(self, model_group: str) -> ArmStats:
        if model_group not in self._arms:
            a0, b0 = _prior_for_group(model_group)
            self._arms[model_group] = ArmStats(
                model_group=model_group, alpha=a0, beta=b0, alpha0=a0, beta0=b0
            )
        return self._arms[model_group]

    def _get_contextual_arm(self, task: str, model_group: str) -> ArmStats:
        key = (task, model_group)
        if key not in self._contextual_arms:
            global_arm = self._get_arm(model_group)
            # Empirical Bayes warm start from global posterior
            self._contextual_arms[key] = ArmStats(
                model_group=model_group,
                alpha=global_arm.alpha,
                beta=global_arm.beta,
                alpha0=global_arm.alpha0,
                beta0=global_arm.beta0,
            )
        return self._contextual_arms[key]

    def select_arm(
        self,
        candidates: List[str],
        strategy: str = "bandit",
        task: Optional[str] = None,
    ) -> Tuple[str, Dict[str, float]]:
        """Select the optimal model group arm using Contextual Discounted Thompson Sampling."""
        if not candidates:
            return "local-always-on", {}
        if len(candidates) == 1:
            return candidates[0], {candidates[0]: 1.0}

        with self._lock:
            scores: Dict[str, float] = {}

            # Epsilon exploration: randomly pick among candidates to discover shifts
            if strategy in ("bandit", "epsilon_greedy", "thompson_sampling") and random.random() < self.epsilon:
                selected = random.choice(candidates)
                for c in candidates:
                    arm = self._get_contextual_arm(task, c) if task else self._get_arm(c)
                    scores[c] = arm.expected_value
                return selected, scores

            if strategy == "epsilon_greedy":
                for c in candidates:
                    arm = self._get_contextual_arm(task, c) if task else self._get_arm(c)
                    scores[c] = arm.expected_value
                best_arm = max(scores, key=scores.get)
                return best_arm, scores

            # Thompson Sampling: sample from Beta distribution for each candidate
            sampled_values: Dict[str, float] = {}
            for c in candidates:
                arm = self._get_contextual_arm(task, c) if task else self._get_arm(c)
                sample = random.betavariate(max(0.1, arm.alpha), max(0.1, arm.beta))
                sampled_values[c] = sample
                scores[c] = round(sample, 4)

            best_arm = max(sampled_values, key=sampled_values.get)
            return best_arm, scores

    @staticmethod
    def _update_arm(arm: ArmStats, success: bool, reward: float,
                    latency_ms: float, cost_usd: float, decay_rate: float) -> None:
        arm.pulls += 1
        arm.last_pulled_at = time.time()
        arm.total_latency_ms += latency_ms
        arm.total_cost_usd += cost_usd
        arm.total_reward += reward

        # Discounted Thompson Sampling (dTS): decay toward informative prior
        arm.alpha = arm.alpha0 + (arm.alpha - arm.alpha0) * decay_rate
        arm.beta = arm.beta0 + (arm.beta - arm.beta0) * decay_rate

        if success:
            arm.successes += 1
            arm.alpha += reward * 1.5
            arm.beta += (1.0 - reward) * 0.5
        else:
            arm.failures += 1
            arm.beta += 3.0  # Asymmetric failure penalty

    def record_feedback(
        self,
        model_group: str,
        success: bool,
        latency_ms: float = 0.0,
        cost_usd: float = 0.0,
        user_rating: Optional[float] = None,
        user_satisfaction: Optional[float] = None,
        task: Optional[str] = None,
    ) -> float:
        """Update global and contextual arm distributions based on execution feedback.

        Args:
            model_group: The model group chosen.
            success: Whether the provider execution succeeded (HTTP 200).
            latency_ms: Observed latency in milliseconds.
            cost_usd: Estimated or actual cost in USD.
            user_rating: Optional normalized user satisfaction (0.0 to 1.0).
            user_satisfaction: Alias for user_rating (accepts -1.0..1.0 or 0.0..1.0).
            task: Optional task context for contextual posterior update.

        Returns:
            Calculated composite reward score in [0.0, 1.0].
        """
        norm_lat = min(1.0, max(0.0, latency_ms / MAX_EXPECTED_LATENCY_MS))
        norm_cost = min(1.0, max(0.0, cost_usd / MAX_EXPECTED_COST_USD))

        raw_rating = user_rating if user_rating is not None else user_satisfaction
        if raw_rating is not None:
            # Map [-1.0, 1.0] to [0.0, 1.0] if negative values are supplied
            rating_score = (raw_rating + 1.0) / 2.0 if raw_rating < 0.0 else min(1.0, max(0.0, raw_rating))
        else:
            rating_score = 0.8

        if success:
            lat_score = 1.0 - norm_lat
            cost_score = 1.0 - norm_cost
            reward = (0.40 * lat_score) + (0.40 * cost_score) + (0.20 * rating_score)
            reward = max(0.05, min(1.0, reward))
        else:
            reward = 0.0

        with self._lock:
            global_arm = self._get_arm(model_group)
            self._update_arm(global_arm, success, reward, latency_ms, cost_usd, self.decay_rate)
            if task:
                ctx_arm = self._get_contextual_arm(task, model_group)
                self._update_arm(ctx_arm, success, reward, latency_ms, cost_usd, self.decay_rate)

        return round(reward, 4)

    def stats(self, task: Optional[str] = None) -> Dict[str, dict]:
        """Return comprehensive telemetry and performance stats for arms."""
        with self._lock:
            if task:
                source = {
                    mg: arm for (t, mg), arm in self._contextual_arms.items() if t == task
                }
            else:
                source = self._arms

            return {
                name: {
                    "pulls": arm.pulls,
                    "successes": arm.successes,
                    "failures": arm.failures,
                    "success_rate": arm.success_rate,
                    "avg_latency_ms": arm.avg_latency_ms,
                    "avg_cost_usd": arm.avg_cost_usd,
                    "mean_reward": arm.mean_reward,
                    "expected_value": arm.expected_value,
                    "alpha": round(arm.alpha, 2),
                    "beta": round(arm.beta, 2),
                }
                for name, arm in source.items()
            }


bandit_router = BanditRouter()
