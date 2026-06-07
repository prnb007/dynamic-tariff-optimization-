"""
Demand simulator for the Monitoring & Learning Agent.

The simulator is the 'environment' that returns realized outcomes given a tariff.
It is parameterised by the estimated price elasticity and random noise,
making it a stochastic proxy for real-world demand response.
"""
import numpy as np

BASELINE_TARIFF = 15.0
SURGE_THRESHOLD = 0.80
DISCOUNT_THRESHOLD = 0.30


class ChargingEnvironment:
    """
    Episodic simulation environment for one station/grid slot.

    State: (predicted_utilization, predicted_baseline_demand, hour, is_peak)
    Action: tariff ∈ [9, 24] ₹/kWh
    Reward: revenue - congestion_penalty - unmet_demand_penalty
    """

    def __init__(self, elasticity: float = -0.3, noise_std: float = 0.05, seed: int = 0):
        self.elasticity = elasticity
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

    def demand_response(self, baseline_demand: float, tariff: float) -> float:
        """Stochastic demand given tariff via elasticity + noise."""
        ratio = tariff / BASELINE_TARIFF
        det = baseline_demand * (ratio ** self.elasticity)
        noise = self.rng.normal(0, self.noise_std * det)
        return float(max(0.0, det + noise))

    def step(self, state: dict, tariff: float) -> dict:
        """
        Simulate one time slot.
        Returns realized metrics and scalar reward.
        """
        pred_util = state["pred_utilization"]
        baseline_demand = state["baseline_demand"]
        capacity = state.get("capacity", max(baseline_demand, 1e-3))

        realized_demand = self.demand_response(baseline_demand, tariff)
        realized_util = min(realized_demand / capacity, 1.0)

        revenue = tariff * realized_demand

        # Congestion penalty (₹ equivalent cost of queue)
        congestion_penalty = 500 * max(0.0, realized_util - SURGE_THRESHOLD)

        # Unmet demand penalty when utilization so high chargers are full
        unmet = max(0.0, realized_demand - capacity)
        unmet_penalty = 200 * unmet

        # Wait time proxy: proportional to overflow above capacity
        wait_time_proxy = max(0.0, realized_util - SURGE_THRESHOLD) * 30  # minutes

        reward = revenue - congestion_penalty - unmet_penalty

        return {
            "tariff": tariff,
            "realized_demand": realized_demand,
            "realized_util": realized_util,
            "revenue": revenue,
            "congestion_penalty": congestion_penalty,
            "wait_time_proxy": wait_time_proxy,
            "reward": reward,
        }
