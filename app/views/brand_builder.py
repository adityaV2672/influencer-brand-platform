"""
Brand OS — Campaign builder.

A brief is four decisions: what the campaign is for, who it should reach, what
the creator has to make, and what a creator may be paid. The last two are what
the matching engine actually consumes — the audience floor and the per-creator
cap decide who is eligible before fit is even considered — so the form shows
the size of the eligible pool updating as those are set, rather than hiding the
consequence until the shortlist is generated.
"""
from __future__ import annotations

import streamlit as st

from nectar import data, state, ui
from nectar.theme import GREEN, INK, INK_2, INK_3, LINE

creators = data.creators()
camps = data.campaigns()

st.markdown(ui.page_header("Create campaign",
                           "Set the brief. The eligible pool updates as you go."),
            unsafe_allow_html=True)

form, side = st.columns([2.1, 1], gap="large")

with form:
    with st.container(border=True):
        st.markdown(ui.section("1 · Campaign"), unsafe_allow_html=True)
        name = st.text_input("Campaign name", value="Festive Glow Edit")
        c1, c2 = st.columns(2)
        with c1:
            category = st.selectbox("Category", sorted(creators.primary_niche.unique()),
                                    index=1)
        with c2:
            objective = st.selectbox("Objective",
                                     ["Awareness", "Consideration", "Conversion"])
        st.text_area("Brief", value="Show the product in an everyday routine. "
                                    "No scripted claims. Disclose the partnership.",
                     height=80)

    with st.container(border=True):
        st.markdown(ui.section("2 · Audience"), unsafe_allow_html=True)
        a1, a2 = st.columns(2)
        with a1:
            geo = st.multiselect("Geography", sorted(creators.audience_geo.dropna().unique()),
                                 default=["IN-West", "IN-North"])
        with a2:
            age = st.multiselect("Age band",
                                 sorted(creators.audience_age_band.dropna().unique()),
                                 default=["18-24", "25-34"])
        min_followers = st.slider("Minimum audience size", 500, 200_000, 8_000, 500,
                                  help="Creators below this are not eligible, however "
                                       "well they score on fit.")

    with st.container(border=True):
        st.markdown(ui.section("3 · Deliverables"), unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)
        with d1:
            n_reel = st.number_input("Reels", 0, 8, 2)
        with d2:
            n_story = st.number_input("Stories", 0, 10, 3)
        with d3:
            n_carousel = st.number_input("Carousels", 0, 8, 0)

    with st.container(border=True):
        st.markdown(ui.section("4 · Budget"), unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        with b1:
            budget = st.number_input("Total budget (₹)", 50_000, 20_000_000,
                                     750_000, 50_000)
        with b2:
            cap = st.number_input("Maximum per creator (₹)", 5_000, 2_000_000,
                                  160_000, 5_000)

# ---- live eligibility -----------------------------------------------------
d = creators.copy()
brief_fee = (d.rate_reel * n_reel + d.rate_story * n_story
             + d.rate_carousel * n_carousel)
elig = d[(d.primary_niche == category) | (d.secondary_niche == category)]
elig = elig[elig.followers >= min_followers]
if geo:
    elig = elig[elig.audience_geo.isin(geo)]
if age:
    elig = elig[elig.audience_age_band.isin(age)]
elig = elig[brief_fee.loc[elig.index] <= cap]
fees = brief_fee.loc[elig.index]

with side:
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("<div class='n-eyebrow'>Eligible pool</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='n-num' style='font-size:34px;line-height:1.25'>{len(elig):,}</div>"
            f"<div style='font-size:12.5px;color:{INK_3};margin-bottom:14px'>"
            f"creators clear every gate</div>", unsafe_allow_html=True)

        affordable = int(budget // fees.median()) if len(fees) and fees.median() else 0
        for label, value in [
            ("Brief price (median creator)", ui.inr(fees.median()) if len(fees) else "—"),
            ("Creators this budget buys", f"{min(affordable, len(elig))}"),
            ("Combined reach", ui.count(elig.followers.sum()) if len(elig) else "—"),
            ("Median engagement", f"{elig.engagement_rate.median() * 100:.1f}%"
             if len(elig) else "—"),
        ]:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:7px 0;border-top:1px solid {LINE};font-size:12.5px'>"
                f"<span style='color:{INK_2}'>{ui.esc(label)}</span>"
                f"<span class='n-num'>{value}</span></div>", unsafe_allow_html=True)

        if len(elig) < 15:
            st.markdown(
                f"<div style='margin-top:12px;font-size:12px;color:#B8860B'>"
                f"A pool this small will not survive the response funnel — roughly "
                f"40% of creators approached accept. Lower the audience floor or "
                f"raise the per-creator cap.</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        if st.button("Create campaign", type="primary", use_container_width=True):
            state.flash(f"“{name}” created as a draft with {len(elig)} eligible creators.")
            st.switch_page("views/brand_campaigns.py")

    st.markdown(
        f"<div class='n-muted' style='margin-top:14px;line-height:1.6'>"
        f"The pool is computed live against the full creator database using the "
        f"fee model's per-deliverable rates. It is the same gate the shortlist "
        f"uses, shown before you commit rather than after.</div>",
        unsafe_allow_html=True)
