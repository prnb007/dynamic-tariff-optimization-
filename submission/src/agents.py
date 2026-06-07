"""
Demand Prediction Agent and Tariff Pricing Agent.

Design notes
- HistGradientBoostingRegressor / Classifier: handles NaN natively (no imputation needed),
  no native wheels required on Python 3.12.
- Time-based train/test split: strictly no leakage.
- Tariff Pricing Agent: rule-based policy + scipy-optimised variant.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASELINE_TARIFF = 15.0
SURGE_THRESHOLD = 0.80
DISCOUNT_THRESHOLD = 0.30
MIN_TARIFF = 9.0
MAX_TARIFF = 24.0


# ---------------------------------------------------------------------------
# Demand Prediction Agent
# ---------------------------------------------------------------------------

class DemandPredictionAgent:
    """
    Predicts next-slot utilization rate, charging load, and congestion probability.
    Uses HistGradientBoosting as the primary model, compared to a naïve seasonal baseline.
    """

    def __init__(self, target_util="utilization_rate", target_load="volume", target_cong="congested"):
        self.target_util = target_util
        self.target_load = target_load
        self.target_cong = target_cong

        self.model_util = HistGradientBoostingRegressor(max_iter=300, random_state=42)
        self.model_load = HistGradientBoostingRegressor(max_iter=300, random_state=42)
        self.model_cong = HistGradientBoostingClassifier(max_iter=300, random_state=42)

    def fit(self, X_train: pd.DataFrame, y_util, y_load, y_cong):
        self.model_util.fit(X_train, y_util)
        self.model_load.fit(X_train, y_load)
        self.model_cong.fit(X_train, y_cong)
        self.feature_names = list(X_train.columns)
        return self

    def predict(self, X_test: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "pred_utilization": self.model_util.predict(X_test),
            "pred_load": self.model_load.predict(X_test),
            "pred_congestion_prob": self.model_cong.predict_proba(X_test)[:, 1],
        }, index=X_test.index)

    @staticmethod
    def naive_baseline(df: pd.DataFrame, target: str, group_cols=("hour", "dow")) -> pd.Series:
        """Seasonal-naïve baseline: average of same hour+dow in training set."""
        means = df.groupby(list(group_cols))[target].mean()
        return df.set_index(list(group_cols)).index.map(means).values

    @staticmethod
    def evaluate(y_true, y_pred, label="model"):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {"label": label, "RMSE": rmse, "MAE": mae, "R2": r2}


def select_features_urbanev(df: pd.DataFrame) -> list[str]:
    """Return the feature columns used for UrbanEV modelling."""
    base = [
        "hour", "dow", "weekend", "is_peak", "is_off_peak",
        "count", "fast_count", "slow_count", "area", "CBD",
        "lag_1_occupancy", "lag_12_occupancy", "lag_288_occupancy",
        "lag_1_volume", "lag_12_volume",
        "lag_1_utilization_rate", "lag_12_utilization_rate", "lag_288_utilization_rate",
        "roll_2h_occupancy", "roll_2h_volume", "roll_2h_utilization_rate",
        "lag_1_price", "roll_2h_price",
    ]
    return [c for c in base if c in df.columns]


def select_features_acn(df: pd.DataFrame) -> list[str]:
    """Return the feature columns used for ACN modelling."""
    base = [
        "hour", "dow", "weekend", "is_peak", "is_off_peak",
        "lag_1h_sessions", "lag_24h_sessions", "rolling_7d_sessions",
    ]
    return [c for c in base if c in df.columns]


# ---------------------------------------------------------------------------
# Tariff Pricing Agent
# ---------------------------------------------------------------------------

class TariffPricingAgent:
    """
    Translates demand forecasts into optimal per-kWh tariffs.

    Elasticity note (documented assumption):
    - Price elasticity ε is estimated from UrbanEV (¥ prices → demand).
    - We treat |ε| as an order-of-magnitude proxy for the INR market; actual
      calibration would require market-specific data.
    - Revenue function: R(p) = p × D0 × (p/p0)^ε  where ε < 0.
    """

    def __init__(self, elasticity: float = -0.3):
        self.elasticity = elasticity  # from UrbanEV estimation

    def rule_based_tariff(self, utilization: float) -> float:
        """Simple rule-based policy matching PS thresholds."""
        if utilization > SURGE_THRESHOLD:
            surge_factor = 1.0 + 0.6 * ((utilization - SURGE_THRESHOLD) / (1 - SURGE_THRESHOLD))
            return float(np.clip(BASELINE_TARIFF * surge_factor, BASELINE_TARIFF, MAX_TARIFF))
        elif utilization < DISCOUNT_THRESHOLD:
            discount_factor = 1.0 - 0.4 * ((DISCOUNT_THRESHOLD - utilization) / DISCOUNT_THRESHOLD)
            return float(np.clip(BASELINE_TARIFF * discount_factor, MIN_TARIFF, BASELINE_TARIFF))
        else:
            return BASELINE_TARIFF

    def optimal_tariff(self, pred_util: float, baseline_demand: float) -> float:
        """
        Maximise R(p) = p × D(p) − congestion_penalty subject to bounds.
        D(p) = baseline_demand × (p/BASELINE_TARIFF)^ε
        Congestion penalty = 1000 × max(0, util_implied − 0.8)
        """
        eps = self.elasticity

        def neg_revenue(p):
            demand = baseline_demand * (p / BASELINE_TARIFF) ** eps
            util_implied = pred_util * (demand / max(baseline_demand, 1e-6))
            cong_penalty = 1000 * max(0.0, util_implied - SURGE_THRESHOLD)
            return -(p * demand - cong_penalty)

        result = minimize_scalar(neg_revenue, bounds=(MIN_TARIFF, MAX_TARIFF), method="bounded")
        return float(result.x)

    def recommend(self, df: pd.DataFrame, demand_col: str = "pred_load") -> pd.DataFrame:
        """Apply both policies to a predictions dataframe."""
        df = df.copy()
        df["tariff_rule"] = df["pred_utilization"].apply(self.rule_based_tariff)
        df["tariff_optimal"] = df.apply(
            lambda r: self.optimal_tariff(r["pred_utilization"], r[demand_col]), axis=1
        )
        df["tariff_baseline"] = BASELINE_TARIFF

        for col in ["tariff_rule", "tariff_optimal"]:
            df[f"rev_{col}"] = df[demand_col] * df[col]
        df["rev_baseline"] = df[demand_col] * BASELINE_TARIFF

        df["revenue_gain_pct_rule"] = (df["rev_tariff_rule"] - df["rev_baseline"]) / df["rev_baseline"].replace(0, np.nan) * 100
        df["revenue_gain_pct_opt"] = (df["rev_tariff_optimal"] - df["rev_baseline"]) / df["rev_baseline"].replace(0, np.nan) * 100

        return df


def estimate_elasticity(df: pd.DataFrame, price_col="price", demand_col="volume") -> float:
    """
    OLS log-log regression: log(demand) ~ log(price) + hour + CBD.
    Returns the price coefficient as the elasticity estimate.
    Assumption: this is an association proxy, not a causal elasticity.
    """
    sub = df[[price_col, demand_col, "hour", "CBD"]].dropna()
    sub = sub[(sub[price_col] > 0) & (sub[demand_col] > 0)]

    X = pd.DataFrame({
        "log_price": np.log(sub[price_col]),
        "hour": sub["hour"],
        "CBD": sub["CBD"].astype(float),
    })
    y = np.log(sub[demand_col])

    model = LinearRegression().fit(X, y)
    elasticity = model.coef_[0]
    print(f"Estimated price elasticity (log-log OLS): {elasticity:.4f}")
    print("  Note: proxy estimate from UrbanEV ¥ data; not a causal INR elasticity.")
    return elasticity
