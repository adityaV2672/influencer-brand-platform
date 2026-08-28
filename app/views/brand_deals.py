"""
Brand OS — Deals. The negotiation thread.

Three panes, matching the reference: the active deal list, the message thread,
and a summary of the terms on the table. The offers in the thread are the fee
model's numbers, not free text.
"""
from __future__ import annotations

import streamlit as st

from nectar import data, state, ui
from nectar.theme import (
    AMBER, CARD, GREEN, INK, INK_2, INK_3, LINE, LINE_2, status_style,
)

reqs = data.requests()
msgs = data.messages()
fit = data.fit()

# Deals needing a decision come first. Sorting purely by funnel stage put the
# finished, paid deals at the top and buried the counter-offers that are the
# only rows a user can actually act on.
live = reqs[reqs.stage_index >= 3].copy()
live["needs_action"] = (live.status == "Countered").astype(int)
live = live.sort_values(["needs_action", "stage_index"], ascending=[False, False])
if live.empty:
    st.markdown(ui.empty_state("🤝", "No deals in flight.",
                               "Deals appear here once a creator responds to a request."),
                unsafe_allow_html=True)
    st.stop()

if st.session_state.get("open_deal") not in set(live.request_id):
    st.session_state["open_deal"] = live.request_id.iloc[0]

left, mid, right = st.columns([1.05, 2.1, 1.15], gap="medium")

# ---- deal list ------------------------------------------------------------
with left:
    st.markdown("<div class='n-eyebrow'>Active deals</div>", unsafe_allow_html=True)
    for r in live.head(12).itertuples():
        selected = r.request_id == st.session_state["open_deal"]
        fg, _ = status_style(r.status)
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:9px;padding:9px 8px 2px 8px;"
            f"border-radius:10px;background:{LINE_2 if selected else 'transparent'}'>"
            f"{ui.avatar(r.initials, r.avatar_color, 30)}"
            f"<div style='flex:1;min-width:0'>"
            f"<div style='font-size:13px;font-weight:600'>{ui.esc(r.creator_name)}</div>"
            f"<div style='font-size:11.5px;color:{INK_3}'>{ui.esc(r.campaign_name)}</div>"
            f"<div style='font-size:11.5px;color:{fg};font-weight:600'>{ui.esc(r.status)}</div>"
            f"</div>"
            f"<div class='n-num' style='font-size:12.5px'>{ui.inr(r.fee_inr)}</div></div>",
            unsafe_allow_html=True)
        if not selected and st.button("Open", key=f"deal_{r.request_id}",
                                      use_container_width=True):
            st.session_state["open_deal"] = r.request_id
            st.rerun()

deal = live[live.request_id == st.session_state["open_deal"]].iloc[0]
thread = msgs[msgs.request_id == deal.request_id].sort_values("seq")

# ---- thread ---------------------------------------------------------------
with mid:
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:11px;padding-bottom:14px;"
        f"border-bottom:1px solid {LINE}'>"
        f"{ui.avatar(deal.initials, deal.avatar_color, 38)}"
        f"<div style='flex:1'><div style='font-size:17px;font-weight:700'>"
        f"{ui.esc(deal.creator_name)}</div>"
        f"<div style='font-size:12.5px;color:{INK_3}'>{ui.esc(deal.campaign_name)} · "
        f"{ui.esc(deal.request_id)}</div></div>"
        f"{ui.chip(deal.status)}</div><div style='height:16px'></div>",
        unsafe_allow_html=True)

    for m in thread.itertuples():
        brand_side = m.sender == "brand"
        bg = INK if brand_side else CARD
        fg = "#ffffff" if brand_side else INK
        border = "none" if brand_side else f"1px solid {LINE}"
        align = "flex-end" if brand_side else "flex-start"
        offer = ""
        if m.offer_inr == m.offer_inr:      # not NaN
            rule = "rgba(255,255,255,0.16)" if brand_side else LINE
            sub = "rgba(255,255,255,0.6)" if brand_side else INK_3
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
            f"<div style='text-align:{'right' if brand_side else 'left'};font-size:11.5px;"
            f"color:{INK_3};margin-bottom:16px'>{ui.esc(m.sender_name)} · "
            f"{ui.esc(m.timestamp)}</div>",
            unsafe_allow_html=True)

    if deal.status == "Countered":
        a, b, _ = st.columns([1.1, 1, 1])
        with a:
            if st.button(f"Accept counter", type="primary", use_container_width=True,
                         key="accept_counter"):
                state.flash(f"Counter accepted — {deal.creator_name} at "
                            f"{ui.inr(deal.counter_fee_inr)}.")
                st.rerun()
        with b:
            if st.button("Counter again", use_container_width=True, key="counter_again"):
                state.flash("Counter-offer drafted.")
                st.rerun()
    msg = state.drain_toast()
    if msg:
        st.success(msg, icon="✓")

# ---- summary --------------------------------------------------------------
with right:
    st.markdown("<div class='n-eyebrow'>Deal summary</div>", unsafe_allow_html=True)
    frow = fit[(fit.campaign_id == deal.campaign_id) &
               (fit.influencer_id == deal.influencer_id)]

    def line(label, value, colour=INK):
        return (f"<div style='margin-bottom:13px'>"
                f"<div style='font-size:11.5px;color:{INK_3}'>{ui.esc(label)}</div>"
                f"<div style='font-size:14px;font-weight:600;color:{colour}'>{value}</div></div>")

    html = [f"<div style='padding-top:6px'>",
            f"<div style='display:flex;align-items:center;gap:9px;margin-bottom:16px'>"
            f"{ui.avatar(deal.initials, deal.avatar_color, 32)}"
            f"<div><div style='font-size:13.5px;font-weight:600'>{ui.esc(deal.creator_name)}</div>"
            f"<div style='font-size:12px;color:{INK_3}'>{ui.esc(deal.creator_handle)}</div></div></div>"]
    html.append(
        f"<div style='display:flex;gap:16px;margin-bottom:16px'>"
        f"<div><div style='font-size:11.5px;color:{INK_3}'>Campaign Fit</div>"
        f"<div class='n-num' style='font-size:21px;color:{GREEN}'>{deal.campaign_fit:.0f}th</div></div>"
        f"<div><div style='font-size:11.5px;color:{INK_3}'>Org Fit</div>"
        f"<div class='n-num' style='font-size:21px;color:{GREEN}'>{deal.org_fit:.0f}th</div></div>"
        f"</div>")
    html.append(line("Fee", f"<span class='n-num'>{ui.inr(deal.fee_inr)}</span>"))
    html.append(line("Deliverables", ui.esc(deal.deliverables)))
    html.append(line("Deadline", ui.esc(deal.deadline)))
    html.append(line("Payment", ui.esc(deal.payment)))
    html.append(line("Usage rights", ui.esc(deal.usage_rights)))
    html.append(line("Exclusivity", ui.esc(deal.exclusivity)))
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)

    if deal.counter_fee_inr == deal.counter_fee_inr:
        st.markdown(
            f"<div style='border:1px solid {LINE};border-radius:12px;padding:13px 15px'>"
            f"<div class='n-eyebrow'>Creator's counter</div>"
            f"<div class='n-num' style='font-size:19px;margin-top:4px'>"
            f"{ui.inr(deal.counter_fee_inr)}</div>"
            f"<div style='font-size:11.5px;color:{INK_3}'>{ui.esc(deal.deliverables)}</div>"
            f"<div style='font-size:11.5px;color:{AMBER};font-weight:600;margin-top:6px'>"
            f"+{(deal.counter_fee_inr / deal.fee_inr - 1) * 100:.0f}% on the opening offer</div>"
            f"</div>", unsafe_allow_html=True)

    if len(frow):
        f = frow.iloc[0]
        st.markdown(
            f"<div style='margin-top:14px'><div class='n-eyebrow'>Why the model rated them</div>"
            + ui.reason_list(list(f.what_helped)[:3])
            + (ui.reason_list(list(f.what_held_back)[:2], tone="warn")
               if len(f.what_held_back) else "")
            + "</div>", unsafe_allow_html=True)
