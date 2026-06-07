"""Evaluation metrics for all three agents (PS-specified)."""
import numpy as np
import pandas as pd

BASELINE_TARIFF = 15.0
DISCOUNT_THRESHOLD = 0.30


def revenue_gain_pct(new_revenue: float, baseline_revenue: float) -> float:
    """((New − Old) / Old) × 100 compared against ₹15/kWh baseline."""
    if baseline_revenue == 0:
        return 0.0
    return (new_revenue - baseline_revenue) / baseline_revenue * 100


def charger_utilization_rate(charging_time: np.ndarray, total_available_time: np.ndarray) -> np.ndarray:
    """Charging Time / Total Available Time per slot."""
    return np.where(total_available_time > 0, charging_time / total_available_time, 0.0)


def off_peak_uplift(sessions_before: pd.Series, sessions_after: pd.Series,
                    util_before: pd.Series) -> float:
    """
    Increase in sessions during low-demand periods (util < 30%) after discounts.
    Returns absolute increase in mean session count over off-peak slots.
    """
    mask = util_before < DISCOUNT_THRESHOLD
    if mask.sum() == 0:
        return 0.0
    return float((sessions_after[mask] - sessions_before[mask]).mean())


def avg_wait_time_reduction(wait_before: np.ndarray, wait_after: np.ndarray) -> float:
    """Average reduction in wait-time proxy across peak periods."""
    return float(np.mean(wait_before - wait_after))


def customer_response_rate(sessions_before: np.ndarray, sessions_after: np.ndarray,
                           tariff_before: np.ndarray, tariff_after: np.ndarray) -> float:
    """
    Session volume shift per unit tariff change (demand elasticity proxy).
    Returns Δsessions / Δtariff averaged across non-zero tariff changes.
    """
    delta_sessions = sessions_after - sessions_before
    delta_tariff = tariff_after - tariff_before
    mask = np.abs(delta_tariff) > 0.01
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(delta_sessions[mask] / delta_tariff[mask]))


def pricing_efficiency_score(revenue: np.ndarray, kwh_delivered: np.ndarray) -> np.ndarray:
    """Revenue per kWh delivered (₹/kWh) over time. Higher is better."""
    return np.where(kwh_delivered > 0, revenue / kwh_delivered, 0.0)


def compile_agent_metrics(episodes: list[dict]) -> pd.DataFrame:
    """Compile per-episode monitoring metrics from RL loop results."""
    return pd.DataFrame(episodes)
