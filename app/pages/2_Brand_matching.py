"""Brand-side matching: pick a brief, get a ranked, safety-gated shortlist."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from theme import (
    GRID, INK, SERIES, STATUS,
    fmt_count, fmt_inr, fmt_pct, load, load_json, locked,
    metric_row, page_header, plotly_layout, require, sidebar_tier,
)

st.set_page_config(page_title="Brand matching", page_icon="◎", layout="wide")
cfg = sidebar_tier()

inf = require("influencers.parquet", "Creator database")
brands = require("brands.parquet", "Brand database")
fit = load("brand_fit.parquet")

page_header("Brand matching", "Semantic fit, hard safety gates, and predicted campaign performance.")

if fit is None:
    st.warning("The brand-fit matrix has not been built. Run `python run_pipeline.py --only brandfit export`.")
    st.stop()

if not cfg["brand_fit"]:
    locked(
        "Brand-fit matching ranks every creator against your specific brief — category "
        "affinity, audience overlap, semantic similarity and competitor-conflict screening.",
        "Brand-fit matching",
    )
    st.stop()

# ==========================================================================
# Brief
# ==========================================================================
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    blabels = {
        r["brand_id"]: f"{r['brand_name']}  ·  {r['category']}  ·  {fmt_inr(r['budget_inr'])} budget"
        for _, r in brands.iterrows()
    }
    bsel = st.selectbox("Brand brief", list(blabels), format_func=lambda k: blabels[k])
brand = brands[brands["brand_id"] == bsel].iloc[0]

with c2:
    min_fit = st.slider("Min brand-fit", 0.0, 1.0, 0.45, 0.05)
with c3:
    show_blocked = st.checkbox("Show blocked creators", value=False,
                               help="Creators screened out by competitor conflict.")

st.caption(
    f"**Brief** — {brand['category']} · target {brand['target_age_band']} in {brand['target_geo']} · "
    f"keywords: {str(brand['brand_keywords']).replace('|', ', ')} · "
    f"competitors screened: {str(brand['competitor_brands']).replace('|', ', ')}"
)

# ==========================================================================
# Shortlist
# ==========================================================================
m = fit[fit["brand_id"] == bsel].merge(inf, on="influencer_id", how="inner")
n_blocked = int((m["gate_multiplier"] == 0).sum())
if not show_blocked:
    m = m[m["gate_multiplier"] > 0]
m = m[m["brand_fit"] >= min_fit]

# Combined ranking: fit and predicted performance both matter. A perfect-fit
# creator nobody engages with is not a good recommendation, and vice versa.
m["match_score"] = (
    0.55 * m["brand_fit"].rank(pct=True) + 0.45 * m["performance_score"] / 100
).round(4)
m = m.sort_values("match_score", ascending=False)

within_budget = m[m["price_estimate_inr"] <= brand["budget_inr"]]
metric_row([
    ("Creators shortlisted", f"{len(m):,}", f"{n_blocked} blocked on brand safety"),
    ("Median brand-fit", f"{m['brand_fit'].median():.2f}" if len(m) else "—", "0–1 composite"),
    ("Within budget", f"{len(within_budget):,}", f"budget {fmt_inr(brand['budget_inr'])}"),
    ("Est. total for top 5",
     fmt_inr(m.head(5)["price_estimate_inr"].sum()) if len(m) else "—", "sum of point estimates"),
])
st.divider()

if not len(m):
    st.warning("No creators clear these thresholds. Lower the minimum brand-fit.")
    st.stop()

left, right = st.columns([1.5, 1])

with left:
    st.markdown("**Ranked shortlist**")
    top = m.head(40)
    table = pd.DataFrame({
        "Creator": top["handle"],
        "Niche": top["primary_niche"],
        "Followers": top["followers"],
        "Brand fit": top["brand_fit"],
        "Performance": top["performance_score"],
        "Est. fee": top.apply(lambda r: f"{fmt_inr(r['price_low_inr'])} – {fmt_inr(r['price_high_inr'])}", axis=1),
        "Status": np.where(top["gate_multiplier"] == 0, "⛔ Blocked",
                           np.where(top["gate_multiplier"] < 1, "⚠ Ad-saturated", "✓ Clear")),
    })
    st.dataframe(
        table, hide_index=True, width="stretch", height=430,
        column_config={
            "Followers": st.column_config.NumberColumn(format="compact"),
            "Brand fit": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
            "Performance": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
        },
    )

with right:
    st.markdown("**Fit vs predicted performance**")
    # Two independent axes, one mark per creator. No dual-axis chart.
    fig = go.Figure(go.Scatter(
        x=m["brand_fit"], y=m["performance_score"],
        mode="markers",
        marker=dict(
            size=np.clip(np.log10(m["followers"].clip(lower=1)) * 3.0, 6, 20),
            color=SERIES[0], opacity=0.7, line=dict(width=1, color="#fcfcfb"),
        ),
        text=m["handle"],
        hovertemplate="<b>%{text}</b><br>fit %{x:.2f}<br>performance %{y:.0f}<extra></extra>",
    ))
    fig.add_hline(y=75, line_width=1, line_dash="dot", line_color=INK["muted"])
    fig.add_vline(x=float(m["brand_fit"].median()), line_width=1, line_dash="dot", line_color=INK["muted"])
    plotly_layout(fig, height=300, showlegend=False,
                  xtitle="Brand fit", ytitle="Performance score")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption("Marker size = audience size. Top-right quadrant is the shortlist you want.")

    st.markdown("**Budget fit**")
    b = go.Figure(go.Histogram(
        x=m["price_estimate_inr"], nbinsx=26,
        marker=dict(color=SERIES[2], line=dict(width=1, color="#fcfcfb")),
    ))
    b.add_vline(x=float(brand["budget_inr"]), line_width=2, line_color=SERIES[1],
                annotation_text="Budget", annotation_position="top")
    b.update_xaxes(type="log")
    plotly_layout(b, height=220, showlegend=False, ytitle="Creators", xtitle="Estimated fee (log, INR)")
    st.plotly_chart(b, width="stretch", config={"displayModeBar": False})

# ==========================================================================
# Explain one match
# ==========================================================================
st.divider()
st.markdown("**Why was this creator matched?**")
pick = st.selectbox(
    "Creator", m["influencer_id"].head(40),
    format_func=lambda k: m[m["influencer_id"] == k]["handle"].iloc[0],
    label_visibility="collapsed",
)
row = m[m["influencer_id"] == pick].iloc[0]

a, b = st.columns([1, 1])
with a:
    comps = {
        "Semantic similarity": row.get("fit_semantic_similarity", np.nan),
        "Category match": row.get("fit_category_match", np.nan),
        "Audience match": row.get("fit_audience_match", np.nan),
        "Content safety": row.get("fit_content_safety", np.nan),
        "Consistency": row.get("fit_consistency", np.nan),
    }
    comps = {k: float(v) for k, v in comps.items() if pd.notna(v)}
    fig = go.Figure(go.Bar(
        x=list(comps.values())[::-1], y=list(comps.keys())[::-1], orientation="h",
        marker=dict(color=SERIES[0]),
        text=[f"{v:.2f}" for v in list(comps.values())[::-1]], textposition="outside",
    ))
    fig.update_xaxes(range=[0, 1.15])
    plotly_layout(fig, height=250, showlegend=False, xtitle="Component score (0–1)")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    cfgj = load_json("brandfit_config.json") or {}
    w = cfgj.get("component_weights", {})
    if w:
        st.caption("Weights — " + " · ".join(f"{k.replace('_', ' ')} {v:.0%}" for k, v in w.items()))

with b:
    st.metric("Brand fit", f"{row['brand_fit']:.2f}",
              delta=f"{row['brand_fit'] - row['brand_fit_ungated']:+.2f} after safety gates"
              if row["gate_multiplier"] != 1 else None)
    reasons = str(row.get("gate_reasons", "") or "")
    if reasons:
        for reason in reasons.split("; "):
            if reason.startswith("BLOCKED"):
                st.error(reason)
            else:
                st.warning(reason.capitalize())
    else:
        st.success("No brand-safety flags. No competitor conflicts, normal ad load.")

    st.caption(
        "Brand-fit is a **transparent composite**, not a learned model — there is no label "
        "for 'was this a good fit', and brand-safety rules must be hard vetoes rather than "
        "soft weights a similarity score can outvote."
    )
