"""Brand OS — Campaigns. Every campaign in one place."""
from __future__ import annotations

import streamlit as st

from nectar import data, state, ui
from nectar.theme import GREEN, INK_3, AMBER

camps = data.campaigns()
summary = data.campaign_summary()

h1, h2 = st.columns([3.4, 1])
with h1:
    st.markdown(ui.page_header("Campaigns", "All your campaigns in one place."),
                unsafe_allow_html=True)
with h2:
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    if st.button("＋  Create campaign", type="primary", use_container_width=True):
        st.switch_page("views/brand_builder.py")

tabs = ["All", "Live", "Draft", "Completed"]
choice = st.segmented_control("Filter", tabs, default="All",
                              label_visibility="collapsed", key="camp_filter")
view = camps if choice in (None, "All") else camps[camps.status == choice]

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

if view.empty:
    st.markdown(ui.empty_state("▤", "Nothing here yet.",
                               f"No campaigns with status “{choice}”."),
                unsafe_allow_html=True)
    st.stop()

rows = []
for c in view.itertuples():
    s = summary[summary.campaign_id == c.campaign_id]
    platforms = "Instagram" if "Carousel" in c.deliverable_label or "Story" in c.deliverable_label \
        else "Instagram, YouTube"
    rows.append([
        f"<div style='font-weight:600'>{ui.esc(c.name)}</div>"
        f"<div style='font-size:12px;color:{INK_3}'>{ui.esc(c.category)}</div>",
        ui.chip(c.status),
        f"<span style='color:{INK_3};font-size:13px'>{ui.esc(platforms)}</span>",
        f"<span class='n-num'>{ui.inr(c.budget_inr)}</span>",
        f"<span class='n-num'>{c.creators_count}</span>",
        (ui.bar(c.progress, GREEN if c.progress >= 0.6 else AMBER)
         + f"<span class='n-num' style='margin-left:8px;font-size:12px'>{c.progress:.0%}</span>"),
    ])

st.markdown(
    ui.table(["Campaign", "Status", "Platform", "Budget", "Creators", "Progress"], rows,
             aligns=["left", "left", "left", "right", "right", "left"]),
    unsafe_allow_html=True)

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
st.markdown("<div class='n-h3' style='margin-bottom:8px'>Open a campaign</div>",
            unsafe_allow_html=True)
cols = st.columns(min(4, len(view)))
for col, c in zip(cols, view.itertuples()):
    with col:
        if st.button(c.name, key=f"open_{c.campaign_id}", use_container_width=True):
            st.session_state["campaign_id"] = c.campaign_id
            st.switch_page("views/brand_discover.py")
