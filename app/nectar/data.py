"""
Data access for the dashboard. Reads precomputed parquet only - no model is
ever loaded here.

The cache key includes each file's modification time and size, not just its
name. Caching on the filename alone looks correct and is a trap: when the
pipeline regenerates a parquet the filename does not change, so Streamlit
keeps serving the previous DataFrame. That produced a deployment where new
code and new data were both live and the app still showed the old rankings.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DATA = Path(__file__).resolve().parents[2] / "app_data"


def _sig(p: Path) -> tuple[int, int]:
    s = p.stat()
    return (s.st_mtime_ns, s.st_size)


@st.cache_data(show_spinner=False)
def _parquet(name: str, sig: tuple[int, int]) -> pd.DataFrame:
    return pd.read_parquet(APP_DATA / name)


@st.cache_data(show_spinner=False)
def _json(name: str, sig: tuple[int, int]) -> dict:
    return json.loads((APP_DATA / name).read_text())


def load(name: str) -> pd.DataFrame | None:
    p = APP_DATA / name
    return _parquet(name, _sig(p)) if p.exists() else None


def load_json(name: str) -> dict | None:
    p = APP_DATA / name
    return _json(name, _sig(p)) if p.exists() else None


def require(name: str, label: str) -> pd.DataFrame:
    df = load(name)
    if df is None:
        st.error(
            f"**{label} is not available.** `app_data/{name}` is missing.\n\n"
            "Rebuild it with:\n```\npython run_pipeline.py\n"
            "python -m src.features.export_nectar\n```"
        )
        st.stop()
    return df


# Convenience accessors, so pages read like prose rather than filenames.
def creators() -> pd.DataFrame:      return require("nectar_creators.parquet", "Creator database")
def campaigns() -> pd.DataFrame:     return require("nectar_campaigns.parquet", "Campaigns")
def fit() -> pd.DataFrame:           return require("nectar_fit.parquet", "Campaign fit")
def requests() -> pd.DataFrame:      return require("nectar_requests.parquet", "Requests")
def messages() -> pd.DataFrame:      return require("nectar_messages.parquet", "Deal threads")
def funnel() -> pd.DataFrame:        return require("nectar_funnel.parquet", "Request funnel")
def campaign_summary() -> pd.DataFrame: return require("nectar_campaign_summary.parquet", "Reporting")
def creator_perf() -> pd.DataFrame:  return require("nectar_creator_performance.parquet", "Creator performance")
def monthly() -> pd.DataFrame:       return require("nectar_monthly.parquet", "Monthly performance")
def earnings() -> pd.DataFrame:      return require("nectar_earnings.parquet", "Earnings")
def calibration() -> pd.DataFrame:   return require("nectar_calibration.parquet", "Model calibration")
def meta() -> dict:                  return load_json("nectar_meta.json") or {}
