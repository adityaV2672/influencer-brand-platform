"""
Brand OS — Discover. The ranked creator search, and the heart of the product.

Three things are worth knowing about this page:

  * Campaign Fit and Organisation Fit are model output, not decoration. Fit
    comes from the brand-fit composite (SBERT semantic similarity, category
    affinity, audience overlap, content safety, posting consistency); the
    ranking objective toggle re-sorts on quantities the performance model
    predicts.
  * "Why this creator?" is generated from the creator's own position on the
    features the model weights, so every claim on a card traces to a number.
  * Creators screened out on brand safety are hidden by default rather than
    deleted, and can be shown with their reasons - a brand needs to know who
    was excluded and why.
"""
from __future__ import annotations

import streamlit as st

from nectar import data, state, ui
from nectar.theme import GREEN, INK, INK_2, INK_3, LINE

fit = data.fit()
camps = data.campaigns()

PLATFORMS = ["Instagram", "YouTube", "Moj", "Josh", "Snapchat"]
CATEGORIES = ["Beauty", "Fashion", "Fitness", "Travel", "Food", "Technology",
              "Gaming", "Finance", "Education", "Parenting", "Home & Decor", "Automotive"]
AVAILABILITY = ["Available", "Busy", "Unavailable"]

rail, body = st.columns([0.85, 4], gap="large")

# ---- filter rail ----------------------------------------------------------
with rail:
    st.markdown("<div class='n-eyebrow' style='margin-top:6px'>Filters</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='n-h3' style='margin:14px 0 2px 0'>Platform</div>",
                unsafe_allow_html=True)
    f_plat = [p for p in PLATFORMS if st.checkbox(p, key=f"pl_{p}")]
    st.markdown("<div class='n-h3' style='margin:14px 0 2px 0'>Category</div>",
                unsafe_allow_html=True)
    f_cat = [c for c in CATEGORIES[:8] if st.checkbox(c, key=f"ct_{c}")]
    st.markdown("<div class='n-h3' style='margin:14px 0 2px 0'>Availability</div>",
                unsafe_allow_html=True)
    f_avail = [a for a in AVAILABILITY if st.checkbox(a, key=f"av_{a}")]
    st.markdown("<div class='n-h3' style='margin:14px 0 2px 0'>Brief eligibility</div>",
                unsafe_allow_html=True)
    # Discover used to rank every unblocked creator, including ones the brief's
    # own per-creator fee cap and audience floor would reject - so the top of
    # the list could be an 800-follower account on a 7.5 lakh campaign that the
    # request pipeline would never actually approach. It now applies the same
    # two gates the campaign builder shows live and the shortlist enforces.
    ignore_gates = st.checkbox("Ignore fee cap and audience floor",
                               key="ignore_gates",
                               help="Show creators this brief cannot afford or "
                                    "that fall below its minimum audience.")
    show_blocked = st.checkbox("Show screened-out creators", key="show_blocked",
                               help="Creators blocked on competitor conflict.")

# ---- body -----------------------------------------------------------------
with body:
    camp = state.campaign()
    st.markdown(ui.page_header("Find your creators.", f"Ranked for {camp.name}."),
                unsafe_allow_html=True)

    s1, s2, s3 = st.columns([1.9, 1.3, 2.0])
    with s1:
        query = st.text_input("Search", placeholder="Search creators…",
                              label_visibility="collapsed", key="disc_q")
    with s2:
        names = list(camps.name)
        idx = names.index(camp.name) if camp.name in names else 0
        picked = st.selectbox("Campaign", names, index=idx,
                              label_visibility="collapsed", key="disc_camp")
        if picked != camp.name:
            st.session_state["campaign_id"] = \
                camps[camps.name == picked].campaign_id.iloc[0]
            st.rerun()
    with s3:
        objective = st.segmented_control(
            "Rank by", ["Best match", "Engagement", "Reach"], default="Best match",
            label_visibility="collapsed", key="disc_rank")

    # ---- apply ------------------------------------------------------------
    d = fit[fit.campaign_id == camp.campaign_id].copy()
    if not show_blocked:
        d = d[~d.blocked]
    if not ignore_gates:
        d = d[d.eligible]
    if query:
        q = query.lower()
        d = d[d.name.str.lower().str.contains(q) | d.nectar_handle.str.lower().str.contains(q)]
    if f_plat:
        d = d[d.platform_names.map(lambda ps: any(p in ps for p in f_plat))]
    if f_cat:
        d = d[d.categories.map(lambda cs: any(c in cs for c in f_cat))]
    if f_avail:
        d = d[d.availability.isin(f_avail)]

    sort_col = {"Best match": "rank_best", "Engagement": "rank_engagement",
                "Reach": "rank_reach"}.get(objective or "Best match", "rank_best")
    d = d.sort_values(sort_col)

    gate_note = ("" if ignore_gates else
                 f" &nbsp;·&nbsp; within {ui.inr(camp.max_per_creator)} per creator "
                 f"and {ui.count(camp.min_followers)}+ audience")
    st.markdown(
        f"<div style='font-family:JetBrains Mono,monospace;font-size:12px;"
        f"color:{INK_3};margin:6px 0 14px 0'>{len(d):,} creators ranked{gate_note}</div>",
        unsafe_allow_html=True)

    if d.empty:
        st.markdown(ui.empty_state("⌕", "No creators match these filters.",
                                   "Widen the category or availability filters, or "
                                   "switch the ranking objective."),
                    unsafe_allow_html=True)
        st.stop()

    msg = state.drain_toast()
    if msg:
        st.success(msg, icon="✓")

    # ---- card grid --------------------------------------------------------
    shown = d.head(24)
    rows = [shown.iloc[i:i + 3] for i in range(0, len(shown), 3)]
    for chunk in rows:
        cols = st.columns(3, gap="medium")
        for col, r in zip(cols, chunk.itertuples()):
            with col:
                with st.container(border=True):
                    st.markdown(ui.creator_card_html(r), unsafe_allow_html=True)
                    b1, b2 = st.columns([1, 1])
                    with b1:
                        if st.button("View profile",
                                     key=f"vp_{r.campaign_id}_{r.influencer_id}",
                                     use_container_width=True):
                            st.session_state["creator_id"] = r.influencer_id
                            st.switch_page("views/creator_profile.py")
                    with b2:
                        saved = r.influencer_id in state.shortlist()
                        if st.button("Saved ✓" if saved else "Shortlist",
                                     key=f"sl_{r.campaign_id}_{r.influencer_id}",
                                     use_container_width=True,
                                     type="secondary" if saved else "primary"):
                            added = state.toggle_shortlist(r.influencer_id)
                            state.flash(f"{r.name} "
                                        f"{'added to' if added else 'removed from'} "
                                        f"your shortlist.")
                            st.rerun()
                st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    if len(d) > 24:
        st.markdown(
            f"<div class='n-muted' style='text-align:center;padding:10px'>"
            f"Showing the top 24 of {len(d)} ranked creators.</div>",
            unsafe_allow_html=True)
