"""Creator OS — Deals. The same negotiation thread, from the other side."""
from __future__ import annotations

import streamlit as st

from nectar import creator_ctx as ctx
from nectar import data, state, ui
from nectar.theme import (
    AMBER, CARD, GREEN, INK, INK_2, INK_3, LINE, LINE_2, status_style,
)

me = ctx.me()
reqs = ctx.my_requests()
msgs = data.messages()

live = reqs[reqs.stage_index >= 3].copy()
if live.empty:
    st.markdown(ui.empty_state("🤝", "No deals in flight.",
                               "A deal opens here once you accept or counter a request."),
                unsafe_allow_html=True)
    st.stop()

live["needs_action"] = (live.status == "Countered").astype(int)
live = live.sort_values(["needs_action", "stage_index"], ascending=[False, False])

if st.session_state.get("open_deal") not in set(live.request_id):
    st.session_state["open_deal"] = live.request_id.iloc[0]

st.markdown(ui.page_header("Deals", "Live partnerships and what you are owed."),
            unsafe_allow_html=True)

left, mid = st.columns([1.1, 2.4], gap="large")

with left:
    st.markdown("<div class='n-eyebrow'>Your deals</div>", unsafe_allow_html=True)
    for r in live.head(12).itertuples():
        selected = r.request_id == st.session_state["open_deal"]
        fg, _ = status_style(r.status)
        st.markdown(
            f"<div style='padding:9px 10px 4px 10px;border-radius:10px;"
            f"background:{LINE_2 if selected else 'transparent'}'>"
            f"<div style='display:flex;justify-content:space-between'>"
            f"<span style='font-size:13px;font-weight:600'>{ui.esc(r.brand_name)}</span>"
            f"<span class='n-num' style='font-size:12.5px'>{ui.inr(r.fee_inr)}</span></div>"
            f"<div style='font-size:11.5px;color:{INK_3}'>{ui.esc(r.campaign_name)}</div>"
            f"<div style='font-size:11.5px;color:{fg};font-weight:600'>{ui.esc(r.status)}</div>"
            f"</div>", unsafe_allow_html=True)
        if not selected and st.button("Open", key=f"cd_{r.request_id}",
                                      use_container_width=True):
            st.session_state["open_deal"] = r.request_id
            st.rerun()

deal = live[live.request_id == st.session_state["open_deal"]].iloc[0]
thread = msgs[msgs.request_id == deal.request_id].sort_values("seq")

with mid:
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:11px;padding-bottom:14px;"
        f"border-bottom:1px solid {LINE}'>"
        f"<div style='flex:1'><div style='font-size:17px;font-weight:700'>"
        f"{ui.esc(deal.brand_name)}</div>"
        f"<div style='font-size:12.5px;color:{INK_3}'>{ui.esc(deal.campaign_name)} · "
        f"{ui.esc(deal.deliverables)}</div></div>"
        f"{ui.chip(deal.status)}</div><div style='height:16px'></div>",
        unsafe_allow_html=True)

    if thread.empty:
        st.markdown(
            f"<div style='background:{CARD};border:1px solid {LINE};border-radius:14px;"
            f"padding:14px 16px;font-size:13.5px'>"
            f"{ui.esc(deal.brand_name)} sent a brief for {ui.esc(deal.campaign_name)}."
            f"<div style='border-top:1px solid {LINE};margin-top:11px;padding-top:10px'>"
            f"<div class='n-num' style='font-size:19px'>{ui.inr(deal.fee_inr)}</div>"
            f"<div style='font-size:11.5px;color:{INK_3}'>{ui.esc(deal.deliverables)} · "
            f"due {ui.esc(deal.deadline)}</div></div></div>",
            unsafe_allow_html=True)
    for m in thread.itertuples():
        mine = m.sender == "creator"
        bg = INK if mine else CARD
        fg = "#ffffff" if mine else INK
        border = "none" if mine else f"1px solid {LINE}"
        align = "flex-end" if mine else "flex-start"
        offer = ""
        if m.offer_inr == m.offer_inr:
            rule = "rgba(255,255,255,0.16)" if mine else LINE
            sub = "rgba(255,255,255,0.6)" if mine else INK_3
            offer = (f"<div style='border-top:1px solid {rule};margin-top:11px;"
                     f"padding-top:10px'>"
                     f"<div class='n-num' style='font-size:19px'>{ui.inr(m.offer_inr)}</div>"
                     f"<div style='font-size:11.5px;color:{sub}'>{ui.esc(m.offer_note or '')}</div>"
                     f"</div>")
        st.markdown(
            f"<div style='display:flex;justify-content:{align};margin-bottom:6px'>"
            f"<div style='max-width:78%;background:{bg};color:{fg};border:{border};"
            f"border-radius:14px;padding:13px 15px;font-size:13.5px;line-height:1.5'>"
            f"{ui.esc(m.body)}{offer}</div></div>"
            f"<div style='text-align:{'right' if mine else 'left'};font-size:11.5px;"
            f"color:{INK_3};margin-bottom:16px'>{ui.esc(m.sender_name)} · "
            f"{ui.esc(m.timestamp)}</div>", unsafe_allow_html=True)

    a, b, _ = st.columns([1, 1, 1.2])
    with a:
        if st.button("Submit deliverable", type="primary", use_container_width=True,
                     key="submit_deliv"):
            state.flash("Deliverable submitted for review.")
            st.rerun()
    with b:
        if st.button("Message brand", use_container_width=True, key="msg_brand"):
            state.flash("Message sent.")
            st.rerun()
    msg = state.drain_toast()
    if msg:
        st.success(msg, icon="✓")
