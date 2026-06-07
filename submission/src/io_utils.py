"""Data loading utilities for ACN-Data and UrbanEV datasets."""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

ACN_PATH = Path(__file__).resolve().parents[2] / "Datasets OP_26 Analytics" / "ACN Data_ 25 April 2018 to 16 Dec 2018" / "acndata_sessions.json.xlsx"
URBAN_DIR = Path(__file__).resolve().parents[2] / "Datasets OP_26 Analytics" / "UrbanEV_ SZ_districts"


# ---------------------------------------------------------------------------
# ACN helpers
# ---------------------------------------------------------------------------

def _parse_user_inputs(val):
    """Convert a stringified list-of-dicts userInputs cell to a flat dict."""
    if pd.isna(val) or val == "" or val == "[]":
        return {}
    try:
        items = ast.literal_eval(val) if isinstance(val, str) else val
        if isinstance(items, list) and items:
            return items[0]  # take the first user input entry
        if isinstance(items, dict):
            return items
    except Exception:
        pass
    return {}


def load_acn(path: Path = ACN_PATH) -> pd.DataFrame:
    """Load ACN xlsx, flatten userInputs, and parse timestamps."""
    df = pd.read_excel(path, engine="openpyxl")

    # Flatten userInputs column if present
    if "userInputs" in df.columns:
        ui = df["userInputs"].apply(_parse_user_inputs)
        ui_df = pd.json_normalize(ui)
        ui_df.columns = ["ui_" + c for c in ui_df.columns]
        df = pd.concat([df.drop(columns=["userInputs"]), ui_df], axis=1)

    # Parse GMT timestamp strings
    ts_cols = ["connectionTime", "disconnectTime", "doneChargingTime"]
    for col in ts_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    # Parse requestedDeparture if it came through userInputs
    if "ui_requestedDeparture" in df.columns:
        df["ui_requestedDeparture"] = pd.to_datetime(
            df["ui_requestedDeparture"], utc=True, errors="coerce"
        )

    return df


# ---------------------------------------------------------------------------
# UrbanEV helpers
# ---------------------------------------------------------------------------

def _load_wide(fname: str) -> pd.DataFrame:
    """Load a wide UrbanEV matrix (rows=timestamps, cols=grid_ids), melt to long."""
    path = URBAN_DIR / fname
    df = pd.read_csv(path, index_col=0)
    df.index.name = "timestamp"
    # copy() to defragment before melt avoids PerformanceWarning
    df = df.copy()
    long = df.reset_index().melt(id_vars="timestamp", var_name="grid_id", value_name=fname.replace(".csv", ""))
    long["grid_id"] = long["grid_id"].astype(str)
    long["timestamp"] = long["timestamp"].astype(int)
    return long


def load_urbanev_time() -> pd.DataFrame:
    """Load the timestamp → datetime mapping."""
    df = pd.read_csv(URBAN_DIR / "time.csv")
    df.columns = df.columns.str.lstrip("﻿").str.strip()
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour", "minute", "second"]])
    df.index = range(1, len(df) + 1)
    df.index.name = "timestamp"
    return df[["datetime"]].reset_index()


def load_urbanev_info() -> pd.DataFrame:
    df = pd.read_csv(URBAN_DIR / "information.csv")
    df["grid_id"] = df["grid"].astype(str)
    return df


def load_urbanev_panel() -> pd.DataFrame:
    """Merge all wide UrbanEV matrices into one long panel and attach datetimes."""
    files = ["occupancy.csv", "volume.csv", "duration.csv", "price.csv"]
    panels = [_load_wide(f) for f in files]

    panel = panels[0]
    for p in panels[1:]:
        panel = panel.merge(p, on=["timestamp", "grid_id"], how="outer")

    time_df = load_urbanev_time()
    panel = panel.merge(time_df, on="timestamp", how="left")

    info = load_urbanev_info()[["grid_id", "count", "fast_count", "slow_count", "area", "lon", "la", "CBD", "dynamic_pricing"]]
    panel = panel.merge(info, on="grid_id", how="left")

    panel = panel.sort_values(["grid_id", "timestamp"]).reset_index(drop=True)
    return panel
