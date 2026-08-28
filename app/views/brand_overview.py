"""Brand OS — Overview. What is moving across the brand's campaigns today."""
from __future__ import annotations

import streamlit as st

from nectar import charts, data, state, ui
from nectar.theme import GREEN, INK, INK_2, INK_3, LINE

camps = data.campaigns()
reqs = data.requests()
monthly = data.monthly()
summary = data.campaign_summary()
brand = state.campaign()

# ---- header ---------------------------------------------------------------
h1, h2 = st.columns([3.4, 1])
with h1:
    st.markdown(
        ui.page_header(f"Good morning, {brand.brand_name.split()[0]}.",
                       "Here's what's moving across your campaigns."),
        unsafe_allow_html=True)
with h2:
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    if st.button("＋  Create campaign", type="primary", use_container_width=True):
        st.switch_page("views/brand_builder.py")

# ---- KPI row --------------------------------------------------------------
active = camps[camps.status == "Live"]
pipeline = int(reqs[reqs.stage_index >= 4].influencer_id.nunique())
pending = int(reqs[reqs.status.isin(["Sent", "Viewed", "Countered"])].shape[0])
needs_action = int(reqs[reqs.status == "Countered"].shape[0])
spend = float(camps.spent_inr.sum())
budget = float(camps.budget_inr.sum())

k = st.columns(4)
for col, (label, value, delta, tone) in zip(k, [
    ("Active campaigns", f"{len(active)}", f"+{len(camps[camps.status == 'Draft'])} in draft", "flat"),
    ("Creators in pipeline", f"{pipeline}", f"across {len(active)} live campaigns", "flat"),
    ("Pending requests", f"{pending}", f"{needs_action} need action", "warn" if needs_action else "flat"),
    ("Campaign spend", ui.inr(spend), f"{spend / budget:.0%} of committed budget", "good"),
]):
    with col:
        st.markdown(ui.kpi(label, value, delta, tone), unsafe_allow_html=True)

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

# ---- performance ----------------------------------------------------------
# border=True so the chart lives INSIDE the card. An HTML card opened in one
# st.markdown call cannot wrap a widget rendered by the next one.
with st.container(border=True):
    st.markdown(ui.section("Campaign performance", "Last 5 months across all campaigns"),
                unsafe_allow_html=True)
    fig = charts.multiline(monthly, "month",
                           {"reach": "Reach", "engagement": "Engagement", "spend": "Spend"},
                           height=270)
    st.plotly_chart(fig, use_container_width=True, config=charts.CONFIG)
    st.markdown(
        f"<div style='font-size:11.5px;color:{INK_3}'>"
        "Each measure is scaled to its own peak so all three fit one axis. "
        "Hover for the underlying figures.</div>",
        unsafe_allow_html=True)

st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)

# ---- active campaigns -----------------------------------------------------
c1, c2 = st.columns([3, 1])
with c1:
    st.markdown("<div class='n-h2'>Active campaigns</div>", unsafe_allow_html=True)
with c2:
    if st.button("View all  →", use_container_width=True, key="view_all_camps"):
        st.switch_page("views/brand_campaigns.py")

rows = []
for c in camps[camps.status != "Completed"].itertuples():
    s = summary[summary.campaign_id == c.campaign_id]
    reach = float(s.reach.iloc[0]) if len(s) else 0.0
    rows.append([
        f"<div style='font-weight:600'>{ui.esc(c.name)}</div>"
        f"<div style='font-size:12px;color:{INK_3}'>{ui.esc(c.brand_name)} · {ui.esc(c.category)}</div>",
        ui.chip(c.status),
        f"<span class='n-num'>{ui.inr(c.budget_inr)}</span>",
        f"<span class='n-num'>{c.creators_count}</span>",
        f"<span class='n-num'>{ui.count(reach)}</span>",
        (ui.bar(c.progress, GREEN if c.progress >= 0.6 else "#D4A017")
         + f"<span class='n-num' style='margin-left:8px;font-size:12px'>{c.progress:.0%}</span>"),
    ])

st.markdown(
    ui.table(["Campaign", "Status", "Budget", "Creators", "Reach", "Progress"], rows,
             aligns=["left", "left", "right", "right", "right", "left"]),
    unsafe_allow_html=True)
