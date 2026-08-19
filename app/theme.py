"""
Shared visual language and data access for the dashboard.

Colour policy
-------------
Categorical hues are assigned in FIXED slot order and never cycled, so a filter
that removes a series never repaints the survivors. The eight-slot order below is
a validated colourblind-safe sequence (adjacent-pair CVD separation checked in
both light and dark surfaces).

Three rules this file enforces for every chart in the app:
  * sequential encoding is ONE hue, light to dark - never a rainbow
  * diverging encoding is two poles with a NEUTRAL GREY midpoint
  * status colours (good/warn/bad) are reserved and never reused as series colours

Charts here never use a second y-axis. Two measures on different scales get two
charts, not one chart with two scales.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DATA = Path(__file__).resolve().parents[1] / "app_data"

# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------
SERIES = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

# Scatter / bubble / small-multiple forms cap at three slots: the full eight
# cannot clear all-pairs colour separation.
SERIES_ALLPAIRS = SERIES[:3]

SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ["#0d366b", "#256abf", "#86b6ef", "#f0efec", "#f0a09f", "#e34948", "#a02322"]

STATUS = {"good": "#008300", "warning": "#eda100", "serious": "#eb6834", "critical": "#e34948"}

INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#8a8a85"}
SURFACE = "#fcfcfb"
GRID = "#e6e5e1"

TIER_COLOR = {
    "Nano": SERIES[0], "Micro": SERIES[1], "Mid": SERIES[2],
    "Macro": SERIES[3], "Mega": SERIES[6],
}
BAND_COLOR = {"High": STATUS["good"], "Medium": STATUS["warning"], "Low": INK["muted"]}
NETWORK_TIER_COLOR = {
    "Hub": SERIES[6], "Influential": SERIES[0],
    "Connected": SERIES[2], "Peripheral": INK["muted"],
}


def plotly_layout(fig, height: int = 340, showlegend: bool | None = None, ytitle="", xtitle=""):
    """Apply the house style. Recessive grid, thin marks, no chart junk."""
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=28, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif",
                  size=12, color=INK["secondary"]),
        hoverlabel=dict(bgcolor=SURFACE, font_size=12, bordercolor=GRID),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        showlegend=showlegend if showlegend is not None else True,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=GRID,
                     tickfont=dict(size=11, color=INK["muted"]), title_text=xtitle)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
                     linecolor="rgba(0,0,0,0)",
                     tickfont=dict(size=11, color=INK["muted"]), title_text=ytitle)
    return fig


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------


def fmt_count(n: float) -> str:
    n = float(n)
    if n >= 1e7:
        return f"{n / 1e7:.2f} Cr"
    if n >= 1e5:
        return f"{n / 1e5:.2f} L"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return f"{n:,.0f}"


def fmt_inr(n: float) -> str:
    n = float(n)
    if n >= 1e7:
        return f"₹{n / 1e7:.2f} Cr"
    if n >= 1e5:
        return f"₹{n / 1e5:.2f} L"
    if n >= 1e3:
        return f"₹{n / 1e3:.1f}K"
    return f"₹{n:,.0f}"


def fmt_pct(x: float, dp: int = 2) -> str:
    return f"{float(x) * 100:.{dp}f}%"


# --------------------------------------------------------------------------
# Data access
# --------------------------------------------------------------------------


def _signature(p: Path) -> tuple[int, int]:
    """(mtime_ns, size) - the cache key ingredient that makes data changes visible.

    Caching on the filename alone looks correct and is a trap: when the pipeline
    regenerates a parquet, the filename is unchanged, so Streamlit happily serves
    the previous DataFrame. That produced a genuinely confusing deployment where
    new code and new data were both live but the app still showed the old
    rankings. Including the file's mtime and size in the key means any data
    change invalidates the entry automatically.
    """
    stat = p.stat()
    return (stat.st_mtime_ns, stat.st_size)


@st.cache_data(show_spinner=False)
def _read_parquet_cached(name: str, signature: tuple[int, int]) -> pd.DataFrame:
    return pd.read_parquet(APP_DATA / name)


@st.cache_data(show_spinner=False)
def _read_json_cached(name: str, signature: tuple[int, int]) -> dict | None:
    try:
        return json.loads((APP_DATA / name).read_text())
    except Exception:  # noqa: BLE001
        return None


def load(name: str) -> pd.DataFrame | None:
    p = APP_DATA / name
    if not p.exists():
        return None
    return _read_parquet_cached(name, _signature(p))


def load_json(name: str) -> dict | None:
    p = APP_DATA / name
    if not p.exists():
        return None
    return _read_json_cached(name, _signature(p))


def require(name: str, label: str) -> pd.DataFrame:
    df = load(name)
    if df is None:
        st.error(
            f"**{label} is not available.** `app_data/{name}` is missing.\n\n"
            "Run the pipeline to build it:\n```\npython run_pipeline.py\n```"
        )
        st.stop()
    return df


# --------------------------------------------------------------------------
# Freemium gating
# --------------------------------------------------------------------------

FREE = {
    "numeric_score": False, "brand_fit": False, "network": False,
    "price_band": False, "advanced_filters": False,
    "max_profiles": 5, "max_results": 10, "shortlists": 1,
}
PAID = {
    "numeric_score": True, "brand_fit": True, "network": True,
    "price_band": True, "advanced_filters": True,
    "max_profiles": 10**9, "max_results": 10**9, "shortlists": 10**9,
}


def tier_config() -> dict:
    return PAID if st.session_state.get("tier", "Free") == "Paid" else FREE


def sidebar_tier() -> dict:
    """The tier switch. Present on every page so gating is always visible."""
    with st.sidebar:
        st.markdown("### Account")
        tier = st.radio(
            "Plan", ["Free", "Paid"],
            index=1 if st.session_state.get("tier") == "Paid" else 0,
            horizontal=True, label_visibility="collapsed",
            help="Switch plans to see exactly what the freemium gate hides.",
        )
        st.session_state["tier"] = tier
        cfg = tier_config()
        if tier == "Free":
            st.caption(
                f"Free plan · {FREE['max_profiles']} profile views/month · "
                "scores shown as bands only"
            )
        else:
            st.caption("Brand Pro · full scores, brand-fit, network view and price bands")
        st.divider()
        return cfg


def locked(message: str, feature: str = "") -> None:
    """Consistent paywall block."""
    st.info(f"🔒 **{feature or 'Paid feature'}** — {message}\n\nSwitch to **Paid** in the sidebar to preview it.")


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(
        f"<div style='margin-bottom:0.5rem'>"
        f"<div style='font-size:1.6rem;font-weight:650;color:{INK['primary']};"
        f"letter-spacing:-0.02em'>{title}</div>"
        + (f"<div style='color:{INK['secondary']};font-size:0.95rem;margin-top:0.15rem'>{subtitle}</div>"
           if subtitle else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def metric_row(items: list[tuple[str, str, str | None]]) -> None:
    """items = [(label, value, caption|None), ...]"""
    cols = st.columns(len(items))
    for col, (label, value, caption) in zip(cols, items):
        with col:
            st.markdown(
                f"<div style='font-size:0.78rem;color:{INK['muted']};text-transform:uppercase;"
                f"letter-spacing:0.04em'>{label}</div>"
                f"<div style='font-size:1.45rem;font-weight:640;color:{INK['primary']};"
                f"line-height:1.25'>{value}</div>"
                + (f"<div style='font-size:0.78rem;color:{INK['secondary']}'>{caption}</div>"
                   if caption else ""),
                unsafe_allow_html=True,
            )
