"""Discover - the brand-side creator search. Entry point of the dashboard."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from theme import (
    BAND_COLOR, GRID, INK, SERIES, STATUS, TIER_COLOR,
    fmt_count, fmt_inr, fmt_pct, locked, metric_row, page_header,
    plotly_layout, require, sidebar_tier,
)

st.set_page_config(
    page_title="Influencer-Brand Platform",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="expanded",
)

cfg = sidebar_tier()
inf = require("influencers.parquet", "Creator database")

page_header(
    "Discover creators",
    "Ranked by a model trained on sponsored-campaign outcomes — not by follower count.",
)

# ==========================================================================
# Ranking objective
# ==========================================================================
# Which creator is "best" is not a technical question. Ranking purely by
# predicted engagement RATE puts the smallest accounts on top every time,
# because engagement rate falls with audience size; ranking by predicted TOTAL
# engagements collapses back to follower count. The objective belongs to the
# campaign, so it is surfaced here instead of being buried in a default sort.
RANK_MODES = {
    "Engagement rate": {
        "col": "score_rate", "band": "band_rate",
        "help": "Highest predicted engagement rate. Favours smaller creators — best for "
                "authenticity-led campaigns and cost-efficient niche targeting.",
    },
    "Total reach": {
        "col": "score_reach", "band": "band_reach",
        "help": "Highest predicted total engagements (rate × audience). Favours larger "
                "creators — best for awareness campaigns with volume targets.",
    },
    "Balanced": {
        "col": "score_balanced", "band": "band_balanced",
        "help": "Blends both percentile ranks equally. Surfaces creators who are strong "
                "on engagement quality without being tiny.",
    },
}

_rc1, _rc2 = st.columns([1.15, 2])
with _rc1:
    rank_mode = st.radio(
        "Rank by", list(RANK_MODES), horizontal=True,
        index=2, help="What the campaign is optimising for.",
    )
_mode = RANK_MODES[rank_mode]
SCORE_COL = _mode["col"] if _mode["col"] in inf.columns else "performance_score"
BAND_COL = _mode["band"] if _mode["band"] in inf.columns else "performance_band"
with _rc2:
    st.caption(f"**{rank_mode}** — {_mode['help']}")

# ==========================================================================
# Filters
# ==========================================================================
with st.sidebar:
    st.markdown("### Filters")
    niches = st.multiselect("Niche", sorted(inf["primary_niche"].unique()))
    tiers = st.multiselect(
        "Follower tier", ["Nano", "Micro", "Mid", "Macro", "Mega"],
        help="Nano <10K · Micro 10-100K · Mid 100-500K · Macro 0.5-2M · Mega 2M+",
    )

    if cfg["advanced_filters"]:
        st.markdown("**Advanced**")
        er_min = st.slider("Min engagement rate", 0.0, 15.0, 0.0, 0.25, format="%.2f%%") / 100
        vs_bench = st.slider(
            "Min engagement vs benchmark", 0.0, 3.0, 0.0, 0.1,
            help="1.0 = exactly the published average for this size and niche.",
        )
        net_tiers = st.multiselect("Network position", ["Hub", "Influential", "Connected", "Peripheral"])
        geos = st.multiselect("Audience geography", sorted(inf["audience_geo"].dropna().unique()))
        ages = st.multiselect("Audience age band", sorted(inf["audience_age_band"].dropna().unique()))
        max_promo = st.slider(
            "Max ad load", 0.0, 1.0, 1.0, 0.05,
            help="Share of the creator's recent posts that are promotional.",
        )
    else:
        er_min, vs_bench, max_promo = 0.0, 0.0, 1.0
        net_tiers, geos, ages = [], [], []
        st.caption("🔒 Advanced filters (engagement quality, network position, audience geo/demo, ad load) are a paid feature.")

# ---- apply -----------------------------------------------------------------
d = inf.copy()
if niches:
    d = d[d["primary_niche"].isin(niches)]
if tiers:
    d = d[d["follower_tier"].isin(tiers)]
if er_min:
    d = d[d["engagement_rate"] >= er_min]
if vs_bench and "er_vs_benchmark" in d:
    d = d[d["er_vs_benchmark"] >= vs_bench]
if net_tiers and "network_tier" in d:
    d = d[d["network_tier"].isin(net_tiers)]
if geos:
    d = d[d["audience_geo"].isin(geos)]
if ages:
    d = d[d["audience_age_band"].isin(ages)]
if max_promo < 1.0 and "content_promo_rate" in d:
    d = d[d["content_promo_rate"].fillna(0) <= max_promo]

d = d.sort_values(SCORE_COL, ascending=False)
n_total = len(d)
capped = n_total > cfg["max_results"]
shown = d.head(cfg["max_results"])

# ==========================================================================
# Summary
# ==========================================================================
metric_row([
    ("Creators matched", f"{n_total:,}", f"of {len(inf):,} in database"),
    ("Median engagement", fmt_pct(d["engagement_rate"].median()) if n_total else "—",
     "organic, recent posts"),
    ("Median reach", fmt_count(d["followers"].median()) if n_total else "—", "followers"),
    ("Median est. fee", fmt_inr(d["price_estimate_inr"].median()) if (n_total and cfg["price_band"]) else "🔒",
     "per deliverable" if cfg["price_band"] else "paid feature"),
])
st.divider()

if n_total == 0:
    st.warning("No creators match these filters. Try widening the niche or tier selection.")
    st.stop()

# ==========================================================================
# Results
# ==========================================================================
left, right = st.columns([1.55, 1])

with left:
    st.markdown(f"**Ranked results** · showing {len(shown):,} of {n_total:,}")

    table = pd.DataFrame({
        "Creator": shown["handle"],
        "Niche": shown["primary_niche"],
        "Followers": shown["followers"],
        "Engagement": shown["engagement_rate"],
    })
    if cfg["numeric_score"]:
        table["Score"] = shown[SCORE_COL]
    else:
        table["Score"] = shown[BAND_COL]
    if cfg["network"] and "network_tier" in shown:
        table["Network"] = shown["network_tier"]
    if cfg["price_band"]:
        table["Est. fee"] = shown.apply(
            lambda r: f"{fmt_inr(r['price_low_inr'])} – {fmt_inr(r['price_high_inr'])}", axis=1
        )

    col_cfg = {
        "Followers": st.column_config.NumberColumn(format="compact"),
        "Engagement": st.column_config.NumberColumn(format="percent", help="Organic engagement rate"),
    }
    if cfg["numeric_score"]:
        col_cfg["Score"] = st.column_config.ProgressColumn(
            "Score", min_value=0, max_value=100, format="%.0f",
            help=f"Percentile rank under the '{rank_mode}' objective.",
        )
    st.dataframe(table, width="stretch", hide_index=True,
                 column_config=col_cfg, height=430)

    if capped:
        locked(
            f"{n_total - cfg['max_results']:,} further matching creators are hidden on the Free plan.",
            "Unlimited results",
        )

with right:
    st.markdown("**Where the matches sit**")

    # Engagement vs reach. Colour encodes tier (identity), fixed slot order.
    plot_d = d.sample(n=min(700, len(d)), random_state=1)
    fig = px.scatter(
        plot_d, x="followers", y="engagement_rate",
        color="follower_tier",
        color_discrete_map=TIER_COLOR,
        category_orders={"follower_tier": ["Nano", "Micro", "Mid", "Macro", "Mega"]},
        log_x=True, hover_data={"handle": True, "followers": ":,", "engagement_rate": ":.2%"},
    )
    fig.update_traces(marker=dict(size=7, opacity=0.72, line=dict(width=1, color="#fcfcfb")))
    fig.update_yaxes(tickformat=".1%")
    plotly_layout(fig, height=300, ytitle="Engagement rate", xtitle="Followers (log)")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption(
        "Engagement declines with audience size — the reason follower count alone "
        "is a poor ranking signal."
    )

    st.markdown("**Score distribution**")
    if cfg["numeric_score"]:
        h = go.Figure(go.Histogram(
            x=d[SCORE_COL], nbinsx=28,
            marker=dict(color=SERIES[0], line=dict(width=1, color="#fcfcfb")),
        ))
        plotly_layout(h, height=210, showlegend=False,
                      ytitle="Creators", xtitle="Score percentile")
        st.plotly_chart(h, width="stretch", config={"displayModeBar": False})
    else:
        counts = d[BAND_COL].value_counts().reindex(["High", "Medium", "Low"]).fillna(0)
        b = go.Figure(go.Bar(
            x=counts.index, y=counts.values,
            marker=dict(color=[BAND_COLOR[i] for i in counts.index]),
            text=[f"{int(v):,}" for v in counts.values], textposition="outside",
        ))
        plotly_layout(b, height=210, showlegend=False, ytitle="Creators")
        st.plotly_chart(b, width="stretch", config={"displayModeBar": False})
        st.caption("🔒 Numeric scores and the feature-level breakdown are a paid feature.")

st.divider()
st.caption(
    "Data note — the creator universe is **synthetic**, calibrated so that engagement "
    "rates and fee bands reproduce published 2026 benchmarks by follower tier. "
    "The NLP methods behind the content signals are validated on **real, human-labelled** "
    "corpora; see the *Model & Methods* page."
)
