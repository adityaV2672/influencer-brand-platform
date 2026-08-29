"""
Creator OS — Discover. The mirror image of the brand-side search: briefs this
creator matches, whether or not the brand has approached them yet.

Same fit composite, same numbers. A creator seeing 82% against a brief is
seeing exactly what the brand sees, which is the point of a two-sided market
that claims to remove the middleman.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from nectar import creator_ctx as ctx
from nectar import data, events, state, ui
from nectar.theme import AMBER, GREEN, INK, INK_2, INK_3, LINE

me = ctx.me()
mine = ctx.my_fit()
camps = data.campaigns()
reqs = ctx.my_requests()
approached = set(reqs.campaign_name.dropna())

st.markdown(ui.page_header("Discover briefs",
                           "Live campaigns ranked by how well you match them."),
            unsafe_allow_html=True)

if mine.empty:
    st.markdown(ui.empty_state("⌕", "No live briefs match you yet.",
                               "Briefs appear as brands open campaigns in your categories."),
                unsafe_allow_html=True)
    st.stop()

d = mine.merge(camps[["campaign_id", "name", "status", "budget_inr", "objective",
                      "deliverable_label", "category", "brand_name", "end_date"]],
               on="campaign_id", suffixes=("", "_c"))
# The three-tier scores and, more useful to a creator, the SPECIFIC reason a
# brief is out of reach. The old page guessed between two possibilities
# ("your rate exceeds the cap" or "the audience floor is higher"); the scoring
# engine knows which, and a creator can only act on the real one.
_cf2 = data.load("nectar_campaign_fit.parquet")
if _cf2 is not None:
    _c = _cf2[_cf2.influencer_id.astype(str) == str(me.influencer_id)].copy()
    d = d.merge(
        _c[["campaign_id", "campaign_fit_pct", "campaign_fit_reasons",
            "block_reasons", "blocked", "c_deliverable_fit", "c_availability_fit",
            "c_budget_fit"]].rename(columns={
                "campaign_fit_pct": "v2_pct", "campaign_fit_reasons": "v2_reasons",
                "block_reasons": "v2_block", "blocked": "v2_blocked"}),
        on="campaign_id", how="left")

d = d[d.status != "Draft"].sort_values("campaign_fit", ascending=False)

msg = state.drain_toast()
if msg:
    st.success(msg, icon="✓")

for i, r in enumerate(d.itertuples()):
    already = r.name_c in approached
    with st.container(border=True):
        a, b, c = st.columns([2.5, 1.1, 1.1])
        with a:
            st.markdown(
                f"<div style='font-size:15.5px;font-weight:700'>{ui.esc(r.name_c)}</div>"
                f"<div style='font-size:12.5px;color:{INK_3}'>{ui.esc(r.brand_name)} · "
                f"{ui.esc(r.category)} · {ui.esc(r.objective)}</div>"
                f"<div style='font-size:13px;color:{INK_2};margin-top:8px'>"
                f"{ui.esc(r.deliverable_label)} &nbsp;·&nbsp; closes {ui.esc(r.end_date)}"
                f"</div>"
                f"<div style='margin-top:10px'>"
                + ui.reason_list(
                    str(getattr(r, "v2_reasons", "") or "").split(" · ")[:3]
                    if getattr(r, "v2_reasons", None)
                    else list(r.what_helped)[:2]) + "</div>",
                unsafe_allow_html=True)
        with b:
            st.markdown(
                f"<div><div style='font-size:11.5px;color:{INK_3}'>Your fit (pctile)</div>"
                f"<div class='n-num' style='font-size:26px;color:{GREEN}'>"
                f"{(getattr(r, 'v2_pct', None) if pd.notna(getattr(r, 'v2_pct', float('nan'))) else r.campaign_fit):.0f}"
                f"<span style='font-size:15px'>th</span></div>"
                f"<div style='font-size:11.5px;color:{INK_3}'>ranked "
                f"#{int(r.rank_best)} of {len(mine) and 2000}</div>"
                f"<div style='font-size:12.5px;margin-top:8px;color:{INK_2}'>"
                f"Brief pays <b class='n-num'>{ui.inr(r.brief_fee_inr)}</b></div></div>",
                unsafe_allow_html=True)
        with c:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            _block = str(getattr(r, "v2_block", "") or "")
            if bool(getattr(r, "v2_blocked", False)) or r.blocked:
                st.markdown(ui.chip("Not eligible"), unsafe_allow_html=True)
                st.markdown(
                    f"<div style='font-size:11.5px;color:{INK_3};margin-top:6px;"
                    f"line-height:1.5'>"
                    f"{ui.esc(_block or 'Competitor conflict — you cannot be matched to this brief.')}"
                    f"</div>", unsafe_allow_html=True)
            elif already:
                st.markdown(ui.chip("Approached"), unsafe_allow_html=True)
                if st.button("Open request", key=f"open_{i}", use_container_width=True):
                    st.switch_page("views/creator_requests.py")
            elif not r.eligible:
                st.markdown(ui.chip("Not eligible"), unsafe_allow_html=True)
                st.markdown(
                    f"<div style='font-size:11.5px;color:{INK_3};margin-top:6px;"
                    f"line-height:1.5'>"
                    f"{ui.esc(_block or 'Outside this brief’s budget or audience floor.')}"
                    f"</div>", unsafe_allow_html=True)
            else:
                if st.button("Pitch for this", key=f"pitch_{i}", type="primary",
                             use_container_width=True):
                    events.log("contacted", actor="creator",
                               actor_id=str(me.influencer_id),
                               campaign_id=str(r.campaign_id),
                               influencer_id=str(me.influencer_id),
                               surface="creator_discover", note="pitch")
                    state.flash(f"Pitch sent to {r.brand_name} for {r.name_c}.")
                    st.rerun()
