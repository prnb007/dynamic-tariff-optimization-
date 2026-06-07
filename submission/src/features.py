"""Feature engineering for both ACN and UrbanEV panels."""
import numpy as np
import pandas as pd

BASELINE_TARIFF = 15.0  # ₹/kWh (PS-specified fixed baseline)
SURGE_THRESHOLD = 0.80
DISCOUNT_THRESHOLD = 0.30
PEAK_HOURS = list(range(7, 10)) + list(range(17, 21))  # 7-9 AM, 5-8 PM
OFF_PEAK_HOURS = list(range(0, 6)) + list(range(22, 24))


# ---------------------------------------------------------------------------
# ACN features
# ---------------------------------------------------------------------------

def build_acn_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive all ACN session-level features."""
    df = df.copy()

    # Durations (hours)
    df["session_duration_h"] = (df["disconnectTime"] - df["connectionTime"]).dt.total_seconds() / 3600
    df["charging_duration_h"] = (df["doneChargingTime"] - df["connectionTime"]).dt.total_seconds() / 3600
    df["idle_time_h"] = (df["disconnectTime"] - df["doneChargingTime"]).dt.total_seconds() / 3600

    # Sanity flags
    df["valid"] = (
        (df["kWhDelivered"] > 0) &
        (df["session_duration_h"] > 0) &
        (df["session_duration_h"] < 48)
    )

    # Site tag
    if "siteID" in df.columns:
        df["site"] = df["siteID"].astype(str).str.lower().apply(lambda x: "caltech" if "ca" in x else "jpl")

    # Simulated revenue (tariff × kWh at baseline)
    df["revenue_baseline"] = df["kWhDelivered"] * BASELINE_TARIFF

    # Calendar from connectionTime (convert to LA time for context)
    ct_la = df["connectionTime"].dt.tz_convert("America/Los_Angeles")
    df["hour"] = ct_la.dt.hour
    df["dow"] = ct_la.dt.dayofweek  # 0=Mon
    df["weekend"] = (df["dow"] >= 5).astype(int)
    df["date"] = ct_la.dt.date

    df["period"] = pd.cut(
        df["hour"],
        bins=[-1, 5, 9, 16, 20, 23],
        labels=["off_peak_night", "morning_peak", "shoulder", "evening_peak", "off_peak_eve"],
    )

    return df


def build_acn_station_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ACN sessions to station × hour slots."""
    df = df[df["valid"]].copy()
    ct_la = df["connectionTime"].dt.tz_convert("America/Los_Angeles")
    df["slot"] = ct_la.dt.floor("h").dt.tz_convert("UTC")

    # Count concurrent sessions as proxy for utilization
    agg = (
        df.groupby(["stationID", "slot"])
        .agg(
            sessions=("sessionID", "count"),
            kWh_delivered=("kWhDelivered", "sum"),
            avg_charging_h=("charging_duration_h", "mean"),
            avg_session_h=("session_duration_h", "mean"),
        )
        .reset_index()
    )

    agg["hour"] = agg["slot"].dt.hour
    agg["dow"] = agg["slot"].dt.dayofweek
    agg["weekend"] = (agg["dow"] >= 5).astype(int)
    agg["is_peak"] = agg["hour"].isin(PEAK_HOURS).astype(int)
    agg["is_off_peak"] = agg["hour"].isin(OFF_PEAK_HOURS).astype(int)

    # Charger Utilization Rate proxy: avg_charging_h / 1 h (slot window)
    agg["utilization_rate"] = (agg["avg_charging_h"] / 1.0).clip(0, 1)

    # Queue Length Proxy: sessions per station per hour (relative load)
    station_capacity = df.groupby("stationID")["sessionID"].count()
    # Normalize sessions against max observed per station
    station_max = agg.groupby("stationID")["sessions"].transform("max")
    agg["queue_proxy"] = (agg["sessions"] / station_max.replace(0, 1)).clip(0, 1)

    # Revenue per session at baseline
    agg["revenue_per_kwh"] = BASELINE_TARIFF  # baseline constant
    agg["revenue_session"] = agg["kWh_delivered"] * BASELINE_TARIFF

    # Lags (for each station, 1h and 24h)
    agg = agg.sort_values(["stationID", "slot"])
    agg["lag_1h_sessions"] = agg.groupby("stationID")["sessions"].shift(1)
    agg["lag_24h_sessions"] = agg.groupby("stationID")["sessions"].shift(24)
    agg["rolling_7d_sessions"] = (
        agg.groupby("stationID")["sessions"]
        .transform(lambda x: x.shift(1).rolling(24 * 7, min_periods=1).mean())
    )

    return agg


# ---------------------------------------------------------------------------
# UrbanEV features
# ---------------------------------------------------------------------------

def build_urbanev_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add all engineered features to the UrbanEV long panel."""
    df = panel.copy()

    # Charger Utilization Rate = occupancy / total piles
    df["utilization_rate"] = (df["occupancy"] / df["count"].replace(0, np.nan)).clip(0, 1)

    # Queue Length Proxy: occupancy density (piles per km², normalised)
    df["occupancy_density"] = df["occupancy"] / df["area"].replace(0, np.nan)

    # Revenue proxy: volume (kWh delivered in slot) × price (¥/kWh used as elasticity input only)
    df["revenue_proxy"] = df["volume"] * df["price"]

    # Energy cost proxy = price (raw ¥/kWh)
    df["energy_cost_per_kwh"] = df["price"]

    # Calendar
    df["hour"] = df["datetime"].dt.hour
    df["dow"] = df["datetime"].dt.dayofweek
    df["weekend"] = (df["dow"] >= 5).astype(int)
    df["is_peak"] = df["hour"].isin(PEAK_HOURS).astype(int)
    df["is_off_peak"] = df["hour"].isin(OFF_PEAK_HOURS).astype(int)
    df["period"] = pd.cut(
        df["hour"],
        bins=[-1, 5, 9, 16, 20, 23],
        labels=["off_peak_night", "morning_peak", "shoulder", "evening_peak", "off_peak_eve"],
    )

    # Lags & rolling stats (per grid)
    df = df.sort_values(["grid_id", "timestamp"])
    for col in ["occupancy", "volume", "price", "utilization_rate"]:
        df[f"lag_1_{col}"] = df.groupby("grid_id")[col].shift(1)
        df[f"lag_12_{col}"] = df.groupby("grid_id")[col].shift(12)   # 1 hour back (12×5min)
        df[f"lag_288_{col}"] = df.groupby("grid_id")[col].shift(288) # 1 day back
        df[f"roll_2h_{col}"] = (
            df.groupby("grid_id")[col]
            .transform(lambda x: x.shift(1).rolling(24, min_periods=1).mean())
        )

    # Congestion flag ground truth: utilization > 0.8
    df["congested"] = (df["utilization_rate"] > SURGE_THRESHOLD).astype(int)

    return df
