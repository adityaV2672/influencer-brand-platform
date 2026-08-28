"""
Brand OS - Find creators.

The intake page for a brand that has never used the platform. It asks four
things in the brand's own words - who you are, what the campaign is, who it
must reach, and what you will pay - and returns a ranked, explained shortlist
scored against all 2,000 creators.

This replaced the earlier dropdown-only Campaign builder. The builder could
only express a brief the taxonomy already had a word for; a new brand's
description is the one thing the platform does not have precomputed, and it is
the first thing a brand actually types.

The matching itself is in nectar/match.py, including an honest account of the
one place it is weaker than the batch engine.
"""
from __future__ import annotations

import streamlit as st

from nectar import data, match, state, ui
from nectar.theme import AMBER, GREEN, INK, INK_2, INK_3, LINE

creators = data.creators()

st.markdown(ui.page_header(
    "Find creators",
    "Describe your brand and the campaign. Every creator is scored against it.",
    eyebrow="New brief"), unsafe_allow_html=True)

form, side = st.columns([2.1, 1], gap="large")

# --------------------------------------------------------------------------
# The brief
# --------------------------------------------------------------------------
with form:
    with st.container(border=True):
        st.markdown(ui.section("1 · Your brand"), unsafe_allow_html=True)
        b1, b2 = st.columns([1.3, 1])
        with b1:
            brand_name = st.text_input("Brand name", value="Aster & Co.")
        with b2:
            category = st.selectbox("Category", sorted(creators.primary_niche.unique()),
                                    index=1)
        brand_text = st.text_area(
            "What does your brand make, and who is it for?",
            value="Affordable everyday skincare and makeup. Simple routines, "
                  "honest reviews, nothing over-styled.",
            height=90,
            help="Written in your own words. The words themselves are matched "
                 "against what creators actually post about.")
        competitors = st.text_input(
            "Competitor brands (comma separated)",
            value="Maybelline, Lakme",
            help="A creator with a recent paid partnership with one of these is "
                 "removed from the shortlist, not just ranked lower.")

    with st.container(border=True):
        st.markdown(ui.section("2 · The campaign"), unsafe_allow_html=True)
        campaign_name = st.text_input("Campaign name", value="Everyday Glow")
        campaign_text = st.text_area(
            "What should the campaign say and show?",
            value="Launch of a new moisturiser. Show it in a real morning "
                  "routine. No scripted claims. Disclose the partnership.",
            height=90)
        g1, g2 = st.columns(2)
        with g1:
            objective = st.selectbox("Goal", list(match.OBJECTIVES),
                                     index=1,
                                     help="What the shortlist should optimise for "
                                          "once fit is satisfied.")
        with g2:
            st.markdown(
                f"<div style='font-size:12px;color:{INK_3};padding-top:30px;"
                f"line-height:1.5'>Ranks on "
                f"{ui.esc(match.OBJECTIVES[objective][1])}.</div>",
                unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(ui.section("3 · Audience"), unsafe_allow_html=True)
        a1, a2 = st.columns(2)
        with a1:
            geo = st.multiselect("Geography",
                                 sorted(creators.audience_geo.dropna().unique()),
                                 default=["IN-West", "IN-North"])
        with a2:
            age = st.multiselect("Age band",
                                 sorted(creators.audience_age_band.dropna().unique()),
                                 default=["18-24", "25-34"])
        min_followers = st.slider("Minimum audience size", 500, 200_000, 8_000, 500,
                                  help="A hard floor. Creators below it are not "
                                       "eligible however well they score.")

    with st.container(border=True):
        st.markdown(ui.section("4 · Deliverables and budget"), unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)
        with d1:
            n_reel = st.number_input("Reels", 0, 8, 2)
        with d2:
            n_story = st.number_input("Stories", 0, 10, 3)
        with d3:
            n_carousel = st.number_input("Carousels", 0, 8, 0)
        b3, b4 = st.columns(2)
        with b3:
            budget = st.number_input("Total budget (₹)", 50_000, 20_000_000,
                                     750_000, 50_000)
        with b4:
            cap = st.number_input("Maximum per creator (₹)", 5_000, 2_000_000,
                                  160_000, 5_000)

brief = match.Brief(
    category=category,
    brand_text=brand_text,
    campaign_text=campaign_text,
    competitors=[c.strip() for c in competitors.split(",") if c.strip()],
    geos=geo, ages=age, min_followers=int(min_followers),
    budget=int(budget), cap=int(cap),
    n_reel=int(n_reel), n_story=int(n_story), n_carousel=int(n_carousel),
    objective=objective,
)

ranked, info = match.score(brief)
eligible = ranked[ranked.eligible]

# --------------------------------------------------------------------------
# Live consequences of the brief
# --------------------------------------------------------------------------
with side:
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("<div class='n-eyebrow'>Eligible pool</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='n-num' style='font-size:34px;line-height:1.25'>"
            f"{info['eligible']:,}</div>"
            f"<div style='font-size:12.5px;color:{INK_3};margin-bottom:14px'>"
            f"of {info['pool']:,} creators clear every gate</div>",
            unsafe_allow_html=True)

        rows = [
            ("Brief price (median)", ui.inr(info["median_fee"]) if info["median_fee"] else "—"),
            ("Creators this budget buys", f"{info['creators_affordable']}"),
            ("Blocked on competitor conflict", f"{info['blocked']}"),
            ("Combined reach", ui.count(eligible.followers.sum()) if len(eligible) else "—"),
            ("Median engagement", f"{eligible.engagement_rate.median() * 100:.1f}%"
             if len(eligible) else "—"),
        ]
        for label, value in rows:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:7px 0;border-top:1px solid {LINE};font-size:12.5px'>"
                f"<span style='color:{INK_2}'>{ui.esc(label)}</span>"
                f"<span class='n-num'>{value}</span></div>", unsafe_allow_html=True)

        if info["eligible"] < 15:
            st.markdown(
                f"<div style='margin-top:12px;font-size:12px;color:{AMBER}'>"
                f"A pool this small will not survive the response funnel — roughly "
                f"40% of creators approached accept. Lower the audience floor or "
                f"raise the per-creator cap.</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        if st.button("Save as campaign", type="primary", use_container_width=True):
            state.flash(f"“{campaign_name}” saved as a draft with "
                        f"{info['eligible']} eligible creators.")
            st.switch_page("views/brand_campaigns.py")

    # ---- what the matcher actually read ----------------------------------
    with st.container(border=True):
        st.markdown("<div class='n-eyebrow'>Words matched</div>", unsafe_allow_html=True)
        if info["fallback"]:
            st.markdown(
                f"<div style='font-size:12.5px;color:{AMBER};line-height:1.6'>"
                f"None of your wording appears in what creators post about, so the "
                f"text is being ignored and the shortlist is ranked on category, "
                f"audience, safety and consistency only. Try naming products, "
                f"categories or hashtags.</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                "".join(ui.chip(t) for t in info["matched"][:14]),
                unsafe_allow_html=True)
        if info["ignored"]:
            st.markdown(
                f"<div style='margin-top:10px;font-size:11.5px;color:{INK_3};"
                f"line-height:1.55'><b>Not in the creator vocabulary:</b> "
                f"{ui.esc(', '.join(info['ignored'][:12]))}"
                f"{'…' if len(info['ignored']) > 12 else ''}</div>",
                unsafe_allow_html=True)
        st.markdown(
            f"<div style='margin-top:10px;font-size:11.5px;color:{INK_3};"
            f"line-height:1.55'>Typed briefs are matched on shared words, not "
            f"shared meaning — the hosted app runs no language model. "
            f"{info['vocab_size']:,} terms are in the vocabulary.</div>",
            unsafe_allow_html=True)

# --------------------------------------------------------------------------
# The shortlist
# --------------------------------------------------------------------------
st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
st.markdown(ui.section(
    "Best-fit creators",
    f"Ranked on fit and on {match.OBJECTIVES[objective][1]}. "
    f"Showing the top 12 of {info['eligible']:,} eligible."), unsafe_allow_html=True)

if not len(eligible):
    st.markdown(ui.empty_state(
        "🔍", "No creator clears every gate",
        "Lower the minimum audience size, raise the per-creator cap, or remove a "
        "competitor from the conflict list."), unsafe_allow_html=True)
else:
    show = eligible.head(12)
    for start in range(0, len(show), 3):
        cols = st.columns(3, gap="medium")
        for col, (_, r) in zip(cols, show.iloc[start:start + 3].iterrows()):
            with col, st.container(border=True):
                st.markdown(
                    ui.creator_cell(r["name"], r.nectar_handle, r.initials,
                                    r.avatar_color, bool(r.verified),
                                    sub=f"{r.primary_niche} · {r.city}"),
                    unsafe_allow_html=True)
                st.markdown(
                    f"<div style='display:flex;gap:8px;margin:12px 0 4px 0'>"
                    f"{ui.fit_tile('Match', float(r.match_display), objective)}"
                    f"{ui.fit_tile('Brand fit', float(r.fit_pct * 100), 'Fit')}"
                    f"</div>", unsafe_allow_html=True)
                st.markdown(ui.metric_strip([
                    ("Followers", ui.count(r.followers)),
                    ("Engagement", f"{r.engagement_rate * 100:.1f}%"),
                    ("Brief price", ui.inr(r.fee)),
                ]), unsafe_allow_html=True)
                st.markdown(ui.reason_list(match.reasons(r, brief, info)[:4]),
                            unsafe_allow_html=True)

    with st.expander("How this score is built"):
        st.markdown(
            f"The composite is the same one used everywhere else in the app: "
            f"semantic similarity **{match.COMPONENT_WEIGHTS['semantic_similarity']:.0%}**, "
            f"category match **{match.COMPONENT_WEIGHTS['category_match']:.0%}**, "
            f"audience overlap **{match.COMPONENT_WEIGHTS['audience_match']:.0%}**, "
            f"content safety **{match.COMPONENT_WEIGHTS['content_safety']:.0%}**, "
            f"consistency **{match.COMPONENT_WEIGHTS['consistency']:.0%}**. "
            f"Competitor conflicts and ad saturation multiply that composite "
            f"rather than being averaged into it, so a paid competitor "
            f"partnership is a veto and not a deduction.\n\n"
            f"**Match** blends brand fit with your goal at "
            f"{1 - match.GOAL_WEIGHT:.0%} / {match.GOAL_WEIGHT:.0%}, both as "
            f"percentiles of the full creator base. Fit decides who is "
            f"appropriate; the goal decides who is most useful among them.\n\n"
            f"One honest caveat: for a brief typed here the semantic term is "
            f"TF-IDF over creator captions, not the SBERT embedding used for "
            f"the precomputed campaigns. The hosted app deliberately loads no "
            f"model, so text arriving at request time cannot be embedded. "
            f"That is why the words your brief did and did not match are shown "
            f"above rather than folded into a single confident number.")
