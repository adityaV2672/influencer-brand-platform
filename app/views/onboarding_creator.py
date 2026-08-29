"""
Creator OS — onboarding.

This page is the platform's answer to the data problem, so it is worth saying
what that problem is. Saves, shares, watch time and audience demographics are
returned by Instagram only to the account owner; no third party can scrape
them. A marketplace that wants those signals has to be GIVEN them, which means
the creator has to want to give them.

So the trade is stated plainly on the connect step: connect your account, and
the numbers only you can see become the reason a brand picks you over someone
with more followers. 61% of creators on the platform have taken that trade.

The connection itself is SIMULATED. No Meta app, no OAuth, no token exchange -
that needs business verification and app review for the
instagram_business_manage_insights scope. The scope names shown are the real
ones so the permission model is not fiction, but nothing is authorised.
"""
from __future__ import annotations

import streamlit as st

from nectar import data, state, ui
from nectar.theme import (ACCENT_A, AMBER, AMBER_BG, GREEN, GREEN_BG, INK,
                          INK_2, INK_3, LINE, LINE_2, MONO)

STEPS = ["Your profile", "Connect Instagram", "Rates & formats",
         "Availability", "Review"]

SCOPES = [
    ("instagram_business_basic", "Profile, follower count and public posts",
     "Public anyway — this is what a brand can already see"),
    ("instagram_business_manage_insights",
     "Saves, shares, reach, watch time, audience age / gender / location",
     "The signals no third party can obtain. This is what changes your ranking"),
    ("instagram_business_manage_comments", "Comment text on your posts",
     "Used to measure audience authenticity and tone, never to reply for you"),
]

UNLOCKS = [
    ("Verified engagement", "Saves and shares are weighted above likes, because "
     "a save is a considered act and a like is not"),
    ("Audience quality score", "Proves your following is real, which is the "
     "single thing brands most want checked"),
    ("Audience match", "Briefs targeted at your actual audience reach you first"),
    ("Verified badge", "Brands can filter for creators with verified metrics"),
]


def _step_header() -> int:
    step = int(st.session_state.get("cr_onb_step", 0))
    cells = "".join(
        f"<div class='ob-step {'done' if i < step else 'now' if i == step else ''}'>"
        f"<span class='n'>{i + 1}</span><span class='t'>{ui.esc(s)}</span></div>"
        for i, s in enumerate(STEPS))
    st.markdown(
        f"<style>.ob-rail{{display:flex;gap:6px;margin:6px 0 26px}}"
        f".ob-step{{flex:1;display:flex;align-items:center;gap:9px;padding:11px 13px;"
        f"border-radius:10px;background:#fff;border:1px solid {LINE};font-size:12.5px;"
        f"color:{INK_3}}}"
        f".ob-step .n{{width:20px;height:20px;border-radius:50%;background:{LINE_2};"
        f"display:inline-flex;align-items:center;justify-content:center;font-size:11px;"
        f"font-weight:700;color:{INK_3};font-family:{MONO}}}"
        f".ob-step.now{{border-color:{INK};color:{INK}}}"
        f".ob-step.now .n{{background:{INK};color:#fff}}"
        f".ob-step.done .n{{background:{GREEN};color:#fff}}"
        f".ob-step.done{{color:{INK_2}}}</style>"
        f"<div class='ob-rail'>{cells}</div>", unsafe_allow_html=True)
    return step


def _nav(step: int, forward_label: str = "Continue") -> None:
    back, fwd = st.columns([1, 1])
    with back:
        if step > 0 and st.button("Back", use_container_width=True, key=f"ob_b{step}"):
            st.session_state["cr_onb_step"] = step - 1
            st.rerun()
    with fwd:
        if step < len(STEPS) - 1 and st.button(forward_label, type="primary",
                                               use_container_width=True,
                                               key=f"ob_f{step}"):
            st.session_state["cr_onb_step"] = step + 1
            st.rerun()


me = state.signed_in_creator()
conn = data.load("nectar_connections.parquet")
connected = False
if conn is not None:
    row = conn[conn.influencer_id.astype(str) == str(me.influencer_id)]
    connected = bool(row.account_connected.iloc[0]) if len(row) else False
connected = bool(st.session_state.get("cr_connected", connected))

st.markdown(ui.page_header(
    "Set up your Nectar profile",
    "Five steps. The only one that changes how brands see you is step two.",
    eyebrow="CREATOR ONBOARDING"), unsafe_allow_html=True)

step = _step_header()

# --------------------------------------------------------------------------
if step == 0:
    with st.container(border=True):
        st.markdown(ui.section("1 · Who you are"), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.text_input("Display name", value=str(me.name), key="ob_name")
            st.text_input("Instagram handle", value=str(me.nectar_handle), key="ob_handle")
        with c2:
            st.text_input("City", value=str(me.city), key="ob_city")
            st.selectbox("Primary niche",
                         sorted(data.creators().primary_niche.unique()),
                         index=sorted(data.creators().primary_niche.unique())
                         .index(str(me.primary_niche)), key="ob_niche")
        st.text_area("Bio", value=str(me.bio), height=80, key="ob_bio")
    _nav(step)

# --------------------------------------------------------------------------
elif step == 1:
    left, right = st.columns([1.35, 1], gap="large")
    with left, st.container(border=True):
        st.markdown(ui.section(
            "2 · Connect your Instagram",
            "Instagram gives some numbers only to you. Nectar cannot see them "
            "unless you share them."), unsafe_allow_html=True)
        for scope, what, why in SCOPES:
            st.markdown(
                f"<div style='padding:13px 0;border-top:1px solid {LINE_2}'>"
                f"<div style='font-family:{MONO};font-size:11px;color:{ACCENT_A};"
                f"letter-spacing:.04em'>{ui.esc(scope)}</div>"
                f"<div style='font-size:13.5px;margin-top:5px'>{ui.esc(what)}</div>"
                f"<div style='font-size:12px;color:{INK_3};margin-top:3px'>"
                f"{ui.esc(why)}</div></div>", unsafe_allow_html=True)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        if connected:
            st.markdown(
                f"<div style='background:{GREEN_BG};color:{GREEN};padding:12px 14px;"
                f"border-radius:10px;font-size:13px;font-weight:600'>"
                f"✓ Account connected — insights are flowing</div>",
                unsafe_allow_html=True)
            if st.button("Disconnect", use_container_width=True, key="ob_disc"):
                st.session_state["cr_connected"] = False
                st.rerun()
        else:
            if st.button("Connect Instagram account", type="primary",
                         use_container_width=True, key="ob_conn"):
                st.session_state["cr_connected"] = True
                state.flash("Account connected. Your insights now feed your scores.")
                st.rerun()
        st.markdown(
            f"<div style='margin-top:12px;font-size:11.5px;color:{INK_3};"
            f"line-height:1.6'><b>Prototype note.</b> The connection is simulated. "
            f"A live build needs a Meta app with business verification and app "
            f"review for the insights scope; the scope names above are the real "
            f"ones so the permission model is accurate.</div>",
            unsafe_allow_html=True)
    with right, st.container(border=True):
        st.markdown(ui.section("What connecting unlocks"), unsafe_allow_html=True)
        for title, body in UNLOCKS:
            st.markdown(
                f"<div style='padding:11px 0;border-top:1px solid {LINE_2}'>"
                f"<div style='font-size:13.5px;font-weight:600'>{ui.esc(title)}</div>"
                f"<div style='font-size:12.5px;color:{INK_2};line-height:1.55;"
                f"margin-top:3px'>{ui.esc(body)}</div></div>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='margin-top:14px;padding:12px 13px;background:{AMBER_BG};"
            f"border-radius:10px;font-size:12.5px;color:{AMBER};line-height:1.55'>"
            f"Creators who have not connected are still listed. Their engagement "
            f"is inferred from public data and marked as such, and they rank "
            f"below verified creators on equal fit.</div>", unsafe_allow_html=True)
    _nav(step)

# --------------------------------------------------------------------------
elif step == 2:
    cap = data.load("nectar_capability.parquet")
    mine = cap[cap.influencer_id.astype(str) == str(me.influencer_id)] if cap is not None else None
    with st.container(border=True):
        st.markdown(ui.section(
            "3 · What you make, and what it costs",
            "Formats you do not offer become a hard block on briefs that need "
            "them — a brand is never shown a creator who cannot deliver."),
            unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        for col, fmt, rate in ((c1, "Reel", me.rate_reel), (c2, "Story", me.rate_story),
                               (c3, "Carousel", me.rate_carousel)):
            with col:
                offered = bool(getattr(mine.iloc[0], f"offers_{fmt.lower()}", True)
                               if mine is not None and len(mine) else True)
                st.checkbox(f"I make {fmt}s", value=offered, key=f"ob_off_{fmt}")
                st.number_input(f"{fmt} rate (₹)", 200, 2_000_000, int(rate or 0),
                                500, key=f"ob_rate_{fmt}")
                if mine is not None and len(mine):
                    s = float(getattr(mine.iloc[0], f"strength_{fmt.lower()}", 0) or 0)
                    if s:
                        st.markdown(
                            f"<div style='font-size:11.5px;color:{INK_3}'>Your "
                            f"{fmt}s perform at <b style='color:{INK}'>{s:.2f}×</b> "
                            f"your own baseline</div>", unsafe_allow_html=True)
    _nav(step)

# --------------------------------------------------------------------------
elif step == 3:
    with st.container(border=True):
        st.markdown(ui.section(
            "4 · When you are free",
            "Briefs whose dates fall entirely inside a booked block are hidden "
            "from you, and you are hidden from them."), unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.selectbox("Current status", ["Available", "Busy", "Unavailable"],
                         index=["Available", "Busy", "Unavailable"].index(
                             str(me.availability)), key="ob_status")
        with c2:
            st.date_input("Booked from", key="ob_bfrom")
        with c3:
            st.date_input("Booked until", key="ob_bto")
        st.slider("Notice you need before a campaign starts (days)", 1, 30,
                  int(getattr(me, "lead_time_days", 7) or 7), key="ob_lead")
    _nav(step, "Review")

# --------------------------------------------------------------------------
else:
    q = data.load("nectar_creator_quality.parquet")
    row = q[q.influencer_id.astype(str) == str(me.influencer_id)] if q is not None else None
    with st.container(border=True):
        st.markdown(ui.section("5 · How brands will see you"), unsafe_allow_html=True)
        if row is not None and len(row):
            r = row.iloc[0]
            a, b = st.columns([1, 2])
            with a:
                st.markdown(
                    f"<div style='text-align:center;padding:14px'>"
                    f"<div style='font-size:11.5px;color:{INK_3}'>CREATOR QUALITY</div>"
                    f"<div class='n-num' style='font-size:46px;line-height:1.1'>"
                    f"{r.creator_quality:.0f}</div>"
                    f"<div style='font-size:13px;font-weight:600'>"
                    f"{ui.esc(r.creator_quality_band)}</div></div>",
                    unsafe_allow_html=True)
            with b:
                st.markdown(ui.reason_list(
                    str(r.creator_quality_reasons).split(" · ")), unsafe_allow_html=True)
        st.markdown(
            f"<div style='margin-top:8px;padding:12px 14px;background:"
            f"{GREEN_BG if connected else AMBER_BG};border-radius:10px;font-size:13px;"
            f"color:{GREEN if connected else AMBER};line-height:1.55'>"
            + ("Your account is connected, so your engagement figures are "
               "verified rather than inferred." if connected else
               "You have not connected your account. Your engagement is inferred "
               "from public data, and verified creators rank above you on equal "
               "fit. Step two takes about ten seconds.")
            + "</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back", use_container_width=True, key="ob_b_last"):
            st.session_state["cr_onb_step"] = 3
            st.rerun()
    with c2:
        if st.button("Finish and see my briefs", type="primary",
                     use_container_width=True, key="ob_done"):
            state.flash("Profile saved. Here are the briefs that fit you.")
            st.switch_page("views/creator_discover.py")
