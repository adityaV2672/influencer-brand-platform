"""
Creator-side view: the other half of the two-sided marketplace.

Brands pay for search. Creators get benchmarking against their niche peers for
free (it is what brings them onto the platform), and pay for visibility boost
and brand-interest signals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from theme import (
    GRID, INK, SEQ_BLUE, SERIES, STATUS,
    fmt_count, fmt_inr, fmt_pct, load, locked, metric_row,
    page_header, plotly_layout, require, sidebar_tier,
)

st.set_page_config(page_title="Creator analytics", page_icon="◎", layout="wide")
cfg = sidebar_tier()

inf = require("influencers.parquet", "Creator database")
page_header("Creator analytics", "How you compare with the creators you are competing against.")

labels = {r["influencer_id"]: f"{r['handle']}  ·  {r['primary_niche']}" for _, r in inf.iterrows()}
sel = st.selectbox("Signed in as", list(labels), format_func=lambda k: labels[k],
                   label_visibility="collapsed")
me = inf[inf["influencer_id"] == sel].iloc[0]

# Peer set: same niche AND same follower tier. Comparing a nano food creator to
# a macro tech creator is meaningless, which is what most public "benchmarks" do.
peers = inf[(inf["primary_niche"] == me["primary_niche"]) & (inf["follower_tier"] == me["follower_tier"])]
if len(peers) < 12:
    peers = inf[inf["primary_niche"] == me["primary_niche"]]
    peer_label = f"{me['primary_niche']} creators (all sizes)"
else:
    peer_label = f"{me['follower_tier']}-tier {me['primary_niche']} creators"


def pct_rank(col: str) -> float:
    if col not in peers.columns or pd.isna(me.get(col)):
        return np.nan
    return float((peers[col] < me[col]).mean() * 100)


metric_row([
    ("Followers", fmt_count(me["followers"]), f"{pct_rank('followers'):.0f}th pctile in peer group"),
    ("Engagement rate", fmt_pct(me["engagement_rate"]), f"{pct_rank('engagement_rate'):.0f}th pctile"),
    ("Growth", f"{me['follower_growth_rate'] * 100:+.1f}%/mo", f"{pct_rank('follower_growth_rate'):.0f}th pctile"),
    ("Suggested rate",
     f"{fmt_inr(me['price_low_inr'])} – {fmt_inr(me['price_high_inr'])}" if cfg["price_band"] else "🔒",
     "what brands are likely to pay" if cfg["price_band"] else "paid feature"),
])
st.caption(f"Benchmarked against **{len(peers):,} {peer_label}**.")
st.divider()

a, b = st.columns([1.25, 1])

with a:
    st.markdown(f"**Where you sit among {peer_label}**")
    metrics = [
        ("Engagement rate", "engagement_rate"),
        ("Comment ratio", "comments_to_likes"),
        ("View-through rate", "views_to_followers"),
        ("Growth rate", "follower_growth_rate"),
        ("Posting frequency", "posting_frequency_month"),
    ]
    names, vals = [], []
    for label, col in metrics:
        v = pct_rank(col)
        if not np.isnan(v):
            names.append(label)
            vals.append(v)
    colors = [STATUS["good"] if v >= 60 else STATUS["warning"] if v >= 35 else SERIES[7] for v in vals]
    fig = go.Figure(go.Bar(
        x=vals[::-1], y=names[::-1], orientation="h",
        marker=dict(color=colors[::-1]),
        text=[f"{v:.0f}" for v in vals[::-1]], textposition="outside",
        hovertemplate="%{y}: %{x:.0f}th percentile<extra></extra>",
    ))
    fig.add_vline(x=50, line_width=1, line_dash="dot", line_color=INK["muted"])
    fig.update_xaxes(range=[0, 108])
    plotly_layout(fig, height=290, showlegend=False, xtitle="Percentile within peer group")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption("Green ≥ 60th percentile · amber 35–60 · red below 35.")

    st.markdown("**Your engagement rate against the peer distribution**")
    h = go.Figure(go.Histogram(
        x=peers["engagement_rate"], nbinsx=26,
        marker=dict(color=SEQ_BLUE[2], line=dict(width=1, color="#fcfcfb")),
        name="Peers",
    ))
    h.add_vline(x=float(me["engagement_rate"]), line_width=2.5, line_color=SERIES[1],
                annotation_text="You", annotation_position="top")
    h.update_xaxes(tickformat=".1%")
    plotly_layout(h, height=240, showlegend=False, ytitle="Creators", xtitle="Engagement rate")
    st.plotly_chart(h, width="stretch", config={"displayModeBar": False})

with b:
    st.markdown("**What would move your score most**")
    # Actionable levers ranked by the gap between the creator and their peers.
    levers = [
        ("Engagement rate", "engagement_rate",
         "Fewer, better posts beat more posts. Engagement is the single largest driver in the model."),
        ("Comment ratio", "comments_to_likes",
         "Comments are harder to fake than likes, and the model weights them accordingly. Ask questions."),
        ("Growth rate", "follower_growth_rate",
         "Momentum signals a healthy audience. Flat growth reads as a stagnating account."),
        ("Ad load", "content_promo_rate",
         "An ad-heavy feed suppresses engagement and triggers a brand-safety penalty in matching."),
    ]
    shown_any = False
    for label, col, advice in levers:
        if col not in peers.columns or pd.isna(me.get(col)):
            continue
        p = pct_rank(col)
        # Ad load is inverted: lower is better.
        good = (p >= 55) if col != "content_promo_rate" else (p <= 45)
        if good:
            continue
        shown_any = True
        gap = float(peers[col].median()) - float(me[col])
        with st.container(border=True):
            st.markdown(f"**{label}** — {p:.0f}th percentile")
            st.caption(advice)
            if col == "content_promo_rate":
                st.caption(f"Peer median ad load: {peers[col].median():.0%} · yours: {me[col]:.0%}")
            elif col in ("engagement_rate", "follower_growth_rate"):
                st.caption(f"Peer median: {peers[col].median() * 100:.2f}% · yours: {me[col] * 100:.2f}%")
            else:
                st.caption(f"Peer median: {peers[col].median():.3f} · yours: {me[col]:.3f}")
    if not shown_any:
        st.success("You are at or above the peer median on every tracked lever. Keep the cadence.")

    st.divider()
    if cfg["price_band"]:
        st.markdown("**Your suggested rate vs peers**")
        fig = go.Figure(go.Histogram(
            x=peers["price_estimate_inr"], nbinsx=22,
            marker=dict(color=SEQ_BLUE[3], line=dict(width=1, color="#fcfcfb")),
        ))
        fig.add_vline(x=float(me["price_estimate_inr"]), line_width=2.5, line_color=SERIES[1],
                      annotation_text="You", annotation_position="top")
        fig.update_xaxes(type="log")
        plotly_layout(fig, height=230, showlegend=False, ytitle="Creators", xtitle="Estimated fee (log, INR)")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    else:
        locked("Rate benchmarking against your peer group is part of Creator Pro.", "Rate guidance")

    st.markdown("**Creator Pro**")
    st.caption(
        "Boosted placement in brand search · alerts when a brand views your profile · "
        "full rate benchmarking · category demand trends."
    )
