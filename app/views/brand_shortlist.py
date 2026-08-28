"""Brand OS — Shortlist. Creators saved from Discover, ready to be briefed."""
from __future__ import annotations

import streamlit as st

from nectar import data, state, ui
from nectar.theme import GREEN, INK_3

fit = data.fit()
camp = state.campaign()
saved = state.shortlist()

st.markdown(ui.page_header("Shortlist",
                           f"Creators you are considering for {camp.name}."),
            unsafe_allow_html=True)

if not saved:
    st.markdown(ui.empty_state("🔖", "Build your shortlist.",
                               "Save creators you're considering. Bookmark from Discover."),
                unsafe_allow_html=True)
    _, mid, _ = st.columns([1.4, 1, 1.4])
    with mid:
        if st.button("Discover creators", type="primary", use_container_width=True):
            st.switch_page("views/brand_discover.py")
    st.stop()

d = fit[(fit.campaign_id == camp.campaign_id) & (fit.influencer_id.isin(saved))]
d = d.sort_values("rank_best")

# Creators can be shortlisted from one campaign and viewed under another; those
# rows have no fit for the campaign in view, so they are listed separately
# rather than silently dropped.
missing = saved - set(d.influencer_id)

total_low = float(d.brief_fee_inr.sum())
k = st.columns(4)
for col, (lbl, val, sub, tone) in zip(k, [
    ("Shortlisted", f"{len(saved)}", f"{len(d)} scored for this brief", "flat"),
    ("Estimated cost", ui.inr(total_low), f"of {ui.inr(camp.budget_inr)} budget", "flat"),
    ("Median fit", f"{d.campaign_fit.median():.0f}th" if len(d) else "—", "percentile for this brief", "good"),
    ("Combined reach", ui.count(d.followers.sum()) if len(d) else "—", "followers", "flat"),
]):
    with col:
        st.markdown(ui.kpi(lbl, val, sub, tone), unsafe_allow_html=True)

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

rows = []
for r in d.itertuples():
    rows.append([
        ui.creator_cell(r.name, r.nectar_handle, r.initials, r.avatar_color,
                        bool(r.verified), f"{r.nectar_handle} · {r.city}"),
        "".join(ui.tag(c) for c in list(r.categories)[:2]),
        f"<span class='n-num'>{ui.count(r.followers)}</span>",
        f"<span class='n-num'>{r.engagement_rate * 100:.1f}%</span>",
        f"<span class='n-num' style='color:{GREEN}'>{r.campaign_fit:.0f}th</span>",
        f"<span class='n-num'>{ui.inr(r.brief_fee_inr)}</span>",
        ui.chip(r.availability),
    ])
st.markdown(
    ui.table(["Creator", "Categories", "Followers", "Engagement", "Fit",
              "Brief cost", "Availability"], rows,
             aligns=["left", "left", "right", "right", "right", "right", "left"]),
    unsafe_allow_html=True)

if missing:
    st.markdown(
        f"<div class='n-muted' style='margin-top:12px'>{len(missing)} shortlisted "
        f"creator(s) have not been scored against this brief — they were saved from "
        f"another campaign.</div>", unsafe_allow_html=True)

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
a, b, _ = st.columns([1, 1, 2])
with a:
    if st.button("Send requests", type="primary", use_container_width=True):
        state.flash(f"{len(d)} requests queued for {camp.name}.")
        st.switch_page("views/brand_requests.py")
with b:
    if st.button("Clear shortlist", use_container_width=True):
        st.session_state["shortlist"] = set()
        st.rerun()
