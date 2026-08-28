"""
Creator OS — Discover. The mirror image of the brand-side search: briefs this
creator matches, whether or not the brand has approached them yet.

Same fit composite, same numbers. A creator seeing 82% against a brief is
seeing exactly what the brand sees, which is the point of a two-sided market
that claims to remove the middleman.
"""
from __future__ import annotations

import streamlit as st

from nectar import creator_ctx as ctx
from nectar import data, state, ui
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
                + ui.reason_list(list(r.what_helped)[:2]) + "</div>",
                unsafe_allow_html=True)
        with b:
            st.markdown(
                f"<div><div style='font-size:11.5px;color:{INK_3}'>Your fit (pctile)</div>"
                f"<div class='n-num' style='font-size:26px;color:{GREEN}'>"
                f"{r.campaign_fit:.0f}<span style='font-size:15px'>th</span></div>"
                f"<div style='font-size:11.5px;color:{INK_3}'>ranked "
                f"#{int(r.rank_best)} of {len(mine) and 2000}</div>"
                f"<div style='font-size:12.5px;margin-top:8px;color:{INK_2}'>"
                f"Brief pays <b class='n-num'>{ui.inr(r.brief_fee_inr)}</b></div></div>",
                unsafe_allow_html=True)
        with c:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            if r.blocked:
                st.markdown(ui.chip("Blocked"), unsafe_allow_html=True)
                st.markdown(
                    f"<div style='font-size:11.5px;color:{INK_3};margin-top:6px'>"
                    f"Competitor conflict — you cannot be matched to this brief.</div>",
                    unsafe_allow_html=True)
            elif already:
                st.markdown(ui.chip("Approached"), unsafe_allow_html=True)
                if st.button("Open request", key=f"open_{i}", use_container_width=True):
                    st.switch_page("views/creator_requests.py")
            elif not r.eligible:
                st.markdown(ui.chip("Not eligible"), unsafe_allow_html=True)
                reason = ("your rate exceeds this brief's per-creator cap"
                          if not r.within_budget else
                          "this brief has a higher audience floor")
                st.markdown(
                    f"<div style='font-size:11.5px;color:{INK_3};margin-top:6px'>"
                    f"{reason}.</div>", unsafe_allow_html=True)
            else:
                if st.button("Pitch for this", key=f"pitch_{i}", type="primary",
                             use_container_width=True):
                    state.flash(f"Pitch sent to {r.brand_name} for {r.name_c}.")
                    st.rerun()
