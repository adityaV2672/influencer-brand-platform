"""Creator OS — Requests. Brand approaches, and what they are worth."""
from __future__ import annotations

import streamlit as st

from nectar import creator_ctx as ctx
from nectar import state, ui
from nectar.theme import AMBER, GREEN, INK_2, INK_3

me = ctx.me()
reqs = ctx.my_requests()

st.markdown(ui.page_header("Requests", "Every brand that has approached you."),
            unsafe_allow_html=True)

if reqs.empty:
    st.markdown(ui.empty_state("✉", "No requests yet.",
                               "Brands find you through search. A complete profile and "
                               "an open availability window are what get you surfaced."),
                unsafe_allow_html=True)
    st.stop()

open_states = ["Sent", "Viewed", "Countered"]
k = st.columns(4)
for col, (lbl, val, sub, tone) in zip(k, [
    ("Open requests", f"{int(reqs.status.isin(open_states).sum())}", "awaiting your reply", "warn"),
    ("Total offered", ui.inr(reqs.fee_inr.sum()), f"across {len(reqs)} approaches", "flat"),
    ("Median fit", f"{reqs.campaign_fit.median():.0f}th", "percentile across briefs", "good"),
    ("Accepted", f"{int((reqs.stage_index >= 4).sum())}", "of all approaches", "good"),
]):
    with col:
        st.markdown(ui.kpi(lbl, val, sub, tone), unsafe_allow_html=True)

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

choice = st.segmented_control("Filter", ["All", "Open", "Accepted", "Closed"],
                              default="All", label_visibility="collapsed",
                              key="creq_filter")
d = reqs
if choice == "Open":
    d = reqs[reqs.status.isin(open_states)]
elif choice == "Accepted":
    d = reqs[reqs.stage_index >= 4]
elif choice == "Closed":
    d = reqs[reqs.status.isin(["Declined", "Paid"])]

msg = state.drain_toast()
if msg:
    st.success(msg, icon="✓")

if d.empty:
    st.markdown(ui.empty_state("✉", "Nothing in this view.",
                               f"No requests with status “{choice}”."),
                unsafe_allow_html=True)
    st.stop()

for i, r in enumerate(d.itertuples()):
    with st.container(border=True):
        a, b, c = st.columns([2.6, 1.1, 1.1])
        with a:
            st.markdown(
                f"<div style='font-size:15px;font-weight:700'>{ui.esc(r.brand_name)}</div>"
                f"<div style='font-size:12.5px;color:{INK_3};margin-bottom:8px'>"
                f"{ui.esc(r.campaign_name)} · {ui.esc(r.brand_category)}</div>"
                f"<div style='font-size:13px;color:{INK_2}'>{ui.esc(r.deliverables)}"
                f" &nbsp;·&nbsp; due {ui.esc(r.deadline)}</div>",
                unsafe_allow_html=True)
        with b:
            st.markdown(
                f"<div><div style='font-size:11.5px;color:{INK_3}'>Offer</div>"
                f"<div class='n-num' style='font-size:21px'>{ui.inr(r.fee_inr)}</div>"
                f"<div style='font-size:11.5px;color:{GREEN};font-weight:600'>"
                f"{r.campaign_fit:.0f}th pctile on this brief</div></div>",
                unsafe_allow_html=True)
        with c:
            st.markdown(f"<div style='padding-bottom:6px'>{ui.chip(r.status)}</div>",
                        unsafe_allow_html=True)
            if r.status in open_states:
                if st.button("Accept", key=f"acc_{i}", type="primary",
                             use_container_width=True):
                    state.flash(f"Accepted {ui.inr(r.fee_inr)} from {r.brand_name}.")
                    st.rerun()
                if st.button("Counter", key=f"cnt_{i}", use_container_width=True):
                    state.flash(f"Counter drafted for {r.brand_name}.")
                    st.rerun()
