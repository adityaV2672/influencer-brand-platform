"""Creator OS — Earnings. What was paid, what is owed, and by whom."""
from __future__ import annotations

import streamlit as st

from nectar import charts, creator_ctx as ctx
from nectar import ui
from nectar.theme import AMBER, GREEN, INK, INK_2, INK_3

me = ctx.me()
earn = ctx.my_earnings()
reqs = ctx.my_requests()

st.markdown(ui.page_header("Earnings", "What you have been paid, and what is owed."),
            unsafe_allow_html=True)

if earn.empty:
    st.markdown(ui.empty_state("₹", "No earnings yet.",
                               "Earnings appear once a deal reaches approval."),
                unsafe_allow_html=True)
    st.stop()

MONTHS = ["Mar", "Apr", "May", "Jun", "Jul", "Aug"]
series = earn.set_index("month").reindex(MONTHS).fillna(0.0).reset_index()
paid = float(series.amount.sum())
pending = float(reqs[(reqs.stage_index >= 4) & (reqs.stage_index < 7)].fee_inr.sum())
best = series.loc[series.amount.idxmax()]

k = st.columns(4)
for col, (lbl, val, sub, tone) in zip(k, [
    ("Paid to date", ui.inr(paid), "last six months", "good"),
    ("Pending", ui.inr(pending), "accepted, not yet settled", "warn"),
    ("Best month", ui.inr(best.amount), str(best.month), "flat"),
    ("Average deal", ui.inr(reqs[reqs.stage_index >= 4].fee_inr.mean()),
     "across accepted deals", "flat"),
]):
    with col:
        st.markdown(ui.kpi(lbl, val, sub, tone), unsafe_allow_html=True)

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

with st.container(border=True):
    st.markdown(ui.section("Earnings over time", "Fees on deals that reached payment"),
                unsafe_allow_html=True)
    st.plotly_chart(charts.column(series.month, series.amount, GREEN, height=250),
                    use_container_width=True, config=charts.CONFIG)

st.markdown("<div style='height:18px'></div>"
            "<div class='n-h2'>Recent deals</div><div style='height:10px'></div>",
            unsafe_allow_html=True)

rows = []
for r in reqs[reqs.stage_index >= 4].head(20).itertuples():
    rows.append([
        f"<span style='font-weight:600'>{ui.esc(r.brand_name)}</span>",
        f"<span style='font-size:13px'>{ui.esc(r.campaign_name)}</span>",
        f"<span style='font-size:12.5px;color:{INK_2}'>{ui.esc(r.deliverables)}</span>",
        f"<span class='n-num'>{ui.inr(r.fee_inr)}</span>",
        ui.chip(r.status),
        f"<span class='n-num' style='font-size:12px;color:{INK_3}'>{ui.esc(r.deadline)}</span>",
    ])
st.markdown(
    ui.table(["Brand", "Campaign", "Deliverables", "Amount", "Status", "Date"], rows,
             aligns=["left", "left", "left", "right", "left", "right"]),
    unsafe_allow_html=True)
