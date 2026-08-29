"""
Brand OS — onboarding. Three steps, then straight into a brief.

Deliberately short. A brand arrives wanting a shortlist, not a form, so the
only things asked for are the ones that actually change scoring: who the brand
is (category and description drive semantic and category fit), who it is
trying to reach (audience alignment), what it will not tolerate (the hard
gates), and who its competitors are (the veto).
"""
from __future__ import annotations

import streamlit as st

from nectar import data, state, ui
from nectar.theme import ACCENT_A, INK, INK_2, INK_3, LINE_2, MONO, RED

STEPS = ["Your brand", "Who you want to reach", "Safety & conflicts"]

POLICY = [
    ("Competitor conflict", "A creator with a recent paid partnership with a "
     "named competitor is BLOCKED, not ranked lower. Exclusivity clauses are "
     "contractual, and a score cannot express a contract."),
    ("Audience authenticity floor", "Creators whose audience the model flags as "
     "Suspect are hidden by default. You are paying for reach; reach that is "
     "not real is not reach."),
    ("Content safety", "Built from three things — what the creator posts, what "
     "their audience writes back, and whether that audience is real."),
]


def _rail(step: int) -> None:
    cells = "".join(
        f"<span style='color:{INK if i == step else INK_3};font-weight:"
        f"{700 if i == step else 400}'>{i + 1}. {ui.esc(s)}</span>"
        for i, s in enumerate(STEPS))
    st.markdown(
        f"<div style='display:flex;gap:26px;font-size:12.5px;margin:4px 0 24px;"
        f"font-family:{MONO};letter-spacing:.03em'>{cells}</div>",
        unsafe_allow_html=True)


step = int(st.session_state.get("br_onb_step", 0))
creators = data.creators()

st.markdown(ui.page_header(
    "Set up your brand",
    "Three steps, then your first shortlist.",
    eyebrow="BRAND ONBOARDING"), unsafe_allow_html=True)
_rail(step)

if step == 0:
    with st.container(border=True):
        st.markdown(ui.section("1 · Your brand"), unsafe_allow_html=True)
        c1, c2 = st.columns([1.3, 1])
        with c1:
            st.text_input("Brand name", value="Aster & Co.", key="br_name")
        with c2:
            st.selectbox("Category", sorted(creators.primary_niche.unique()),
                         index=1, key="br_cat")
        st.text_area(
            "What do you make, and who is it for?",
            value="Affordable everyday skincare and makeup. Simple routines, "
                  "honest reviews, nothing over-styled.",
            height=95, key="br_desc",
            help="Matched against what creators actually post about. The words "
                 "themselves are used, so name products and categories.")
        st.text_input("Website or Instagram handle", value="@asterandco", key="br_site")

elif step == 1:
    with st.container(border=True):
        st.markdown(ui.section("2 · Who you want to reach"), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.multiselect("Geography",
                           sorted(creators.audience_geo.dropna().unique()),
                           default=["IN-West", "IN-North"], key="br_geo")
        with c2:
            st.multiselect("Age band",
                           sorted(creators.audience_age_band.dropna().unique()),
                           default=["18-24", "25-34"], key="br_age")
        c3, c4 = st.columns(2)
        with c3:
            st.slider("Typical budget per creator (₹)", 5_000, 500_000, 120_000,
                      5_000, key="br_cap")
        with c4:
            st.slider("Minimum audience size", 500, 200_000, 8_000, 500,
                      key="br_floor")

else:
    with st.container(border=True):
        st.markdown(ui.section(
            "3 · Safety and conflicts",
            "These are gates, not preferences. A creator who fails one is "
            "removed from your results with a reason, not shown at a lower score."),
            unsafe_allow_html=True)
        st.text_input("Competitor brands (comma separated)",
                      value="Maybelline, Lakme", key="br_comp")
        st.checkbox("Hide creators whose audience is flagged Suspect",
                    value=True, key="br_hide_suspect")
        st.checkbox("Require verified metrics (creator has connected their account)",
                    value=False, key="br_require_verified",
                    help="Narrows the pool to the 61% of creators who have "
                         "connected. Their engagement figures are measured "
                         "rather than inferred.")
        for title, body in POLICY:
            st.markdown(
                f"<div style='padding:12px 0;border-top:1px solid {LINE_2}'>"
                f"<div style='font-size:13px;font-weight:600'>{ui.esc(title)}</div>"
                f"<div style='font-size:12.5px;color:{INK_2};line-height:1.55;"
                f"margin-top:3px'>{ui.esc(body)}</div></div>", unsafe_allow_html=True)

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
back, fwd = st.columns(2)
with back:
    if step > 0 and st.button("Back", use_container_width=True, key=f"brb{step}"):
        st.session_state["br_onb_step"] = step - 1
        st.rerun()
with fwd:
    if step < len(STEPS) - 1:
        if st.button("Continue", type="primary", use_container_width=True,
                     key=f"brf{step}"):
            st.session_state["br_onb_step"] = step + 1
            st.rerun()
    else:
        if st.button("Write my first brief", type="primary",
                     use_container_width=True, key="br_done"):
            state.flash("Brand set up. Describe the campaign and we will score "
                        "every creator against it.")
            st.switch_page("views/brand_builder.py")
