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

from nectar import data, events, state, ui
from nectar.theme import GREEN, INK, INK_2, INK_3, LINE

fit = data.fit()
camps = data.campaigns()

# The three-tier scores live in their own tables so the older ranking columns
# keep working. Merged in here rather than recomputed per card: a card is drawn
# 24 times a page and a merge is drawn once.
_cq = data.load("nectar_creator_quality.parquet")
_aq = data.load("nectar_audience_quality.parquet")
_cf2 = data.load("nectar_campaign_fit.parquet")
_conn = data.load("nectar_connections.parquet")
for _t, _cols in ((_cq, ["influencer_id", "creator_quality", "creator_quality_band",
                         "creator_quality_reasons", "verified_metrics"]),
                  (_aq, ["influencer_id", "audience_quality_score", "audience_band"]),
                  (_conn, ["influencer_id", "account_connected"])):
    if _t is not None:
        _t = _t.copy(); _t["influencer_id"] = _t.influencer_id.astype(str)
        fit["influencer_id"] = fit.influencer_id.astype(str)
        fit = fit.merge(_t[[c for c in _cols if c in _t.columns]],
                        on="influencer_id", how="left")
if _cf2 is not None:
    _c = _cf2.copy(); _c["influencer_id"] = _c.influencer_id.astype(str)
    # nectar_fit already carries campaign_fit and org_fit from the older
    # composite, so the new columns are renamed rather than merged on top of
    # them - pandas would otherwise suffix both to _x/_y and the creator card,
    # which reads r.campaign_fit, would break on every row.
    fit = fit.merge(
        _c[["campaign_id", "influencer_id", "campaign_fit_pct", "campaign_fit_band",
            "campaign_fit_reasons", "org_fit", "block_reasons"]].rename(columns={
                "campaign_fit_pct": "v2_campaign_fit_pct",
                "campaign_fit_band": "v2_campaign_fit_band",
                "campaign_fit_reasons": "v2_reasons",
                "org_fit": "v2_org_fit",
                "block_reasons": "v2_block_reasons"}),
        on=["campaign_id", "influencer_id"], how="left")

PLATFORMS = ["Instagram", "YouTube", "Moj", "Josh", "Snapchat"]
CATEGORIES = ["Beauty", "Fashion", "Fitness", "Travel", "Food", "Technology",
              "Gaming", "Finance", "Education", "Parenting", "Home & Decor", "Automotive"]
AVAILABILITY = ["Available", "Busy", "Unavailable"]

def _score_strip(r) -> str:
    """Campaign Fit, Organisation Fit and Creator Quality, side by side.

    Three numbers rather than one because they answer different questions and a
    brand needs to be able to see them disagree - a creator can be excellent in
    general, a poor long-term match for the brand, and still right for this one
    brief.
    """
    def tile(label, value, sub, tone=INK):
        shown = "—" if value is None or (isinstance(value, float) and value != value) \
            else f"{float(value):.0f}"
        return (f"<div style='flex:1;min-width:0'>"
                f"<div style='font-size:10.5px;color:{INK_3};letter-spacing:.03em'>"
                f"{ui.esc(label)}</div>"
                f"<div class='n-num' style='font-size:19px;color:{tone};line-height:1.3'>"
                f"{shown}</div>"
                f"<div style='font-size:10.5px;color:{INK_3}'>{ui.esc(sub)}</div></div>")

    aq = getattr(r, "audience_quality_score", None)
    band = str(getattr(r, "audience_band", "") or "")
    aq_tone = {"Suspect": "#C2413F", "Mixed": "#B8860B"}.get(band, GREEN)
    verified = bool(getattr(r, "account_connected", False))
    return (f"<div style='display:flex;gap:10px;margin:12px 0 10px 0;"
            f"padding-top:11px;border-top:1px solid {LINE}'>"
            + tile("CAMPAIGN FIT", getattr(r, "v2_campaign_fit_pct", None), "percentile")
            + tile("ORG FIT", getattr(r, "v2_org_fit", None),
                   str(getattr(r, "v2_campaign_fit_band", "") or ""))
            + tile("QUALITY", getattr(r, "creator_quality", None),
                   str(getattr(r, "creator_quality_band", "") or ""))
            + tile("AUDIENCE", aq, band or "unrated", aq_tone)
            + "</div>"
            + (f"<div style='font-family:JetBrains Mono,monospace;font-size:9.5px;"
               f"letter-spacing:.06em;color:{GREEN};margin-bottom:6px'>"
               f"✓ VERIFIED METRICS</div>" if verified else
               f"<div style='font-family:JetBrains Mono,monospace;font-size:9.5px;"
               f"letter-spacing:.06em;color:{INK_3};margin-bottom:6px'>"
               f"INFERRED FROM PUBLIC DATA</div>"))


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
    # two gates the Find creators page shows live and the shortlist enforces.
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
    # An impression is the denominator every downstream rate needs: without it
    # a shortlist rate cannot be computed, only a shortlist count.
    if not st.session_state.get(f"_seen_{camp.campaign_id}"):
        for _r in shown.itertuples():
            events.log("viewed", actor="brand", actor_id=str(camp.brand_id),
                       brand_id=str(camp.brand_id),
                       campaign_id=str(camp.campaign_id),
                       influencer_id=str(_r.influencer_id), surface="discover")
        st.session_state[f"_seen_{camp.campaign_id}"] = True
    rows = [shown.iloc[i:i + 3] for i in range(0, len(shown), 3)]
    for chunk in rows:
        cols = st.columns(3, gap="medium")
        for col, r in zip(cols, chunk.itertuples()):
            with col:
                with st.container(border=True):
                    st.markdown(ui.creator_card_html(r), unsafe_allow_html=True)
                    st.markdown(_score_strip(r), unsafe_allow_html=True)
                    _reasons = str(getattr(r, "v2_reasons", "") or "")
                    if _reasons:
                        st.markdown(
                            ui.reason_list(_reasons.split(" · ")[:3],
                                           tone="warn" if getattr(r, "blocked", False)
                                           else "good"),
                            unsafe_allow_html=True)
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
                            # Both directions are recorded. A rejection is as
                            # informative to a ranker as a shortlist, and a log
                            # that only keeps the positives teaches a model that
                            # everything is good.
                            events.log("shortlisted" if added else "declined",
                                       actor="brand", actor_id=str(camp.brand_id),
                                       brand_id=str(camp.brand_id),
                                       campaign_id=str(camp.campaign_id),
                                       influencer_id=str(r.influencer_id),
                                       surface="discover",
                                       note=str(getattr(r, "v2_campaign_fit_pct", "")))
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
