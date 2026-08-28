"""Settings. Real controls where the app has them; honest labels where it does not."""
from __future__ import annotations

import streamlit as st

from nectar import data, state, ui
from nectar.theme import AMBER, GREEN, INK_2, INK_3, LINE

meta = data.meta()

st.markdown(ui.page_header("Settings", "Account, workspace and data."),
            unsafe_allow_html=True)

a, b = st.columns([1.5, 1], gap="large")

with a:
    with st.container(border=True):
        st.markdown(ui.section("Workspace"), unsafe_allow_html=True)
        role = st.radio("Signed in as", ["Brand / Agency", "Creator"],
                        index=0 if state.role() == "brand" else 1, horizontal=True)
        want = "brand" if role.startswith("Brand") else "creator"
        if want != state.role():
            state.set_role(want)
            st.rerun()

        camps = data.campaigns()
        names = list(camps.name)
        cur = state.campaign().name
        pick = st.selectbox("Default campaign", names,
                            index=names.index(cur) if cur in names else 0)
        if pick != cur:
            st.session_state["campaign_id"] = camps[camps.name == pick].campaign_id.iloc[0]
            st.rerun()

    with st.container(border=True):
        st.markdown(ui.section("Matching policy",
                               "How the shortlist is gated before fit is considered"),
                    unsafe_allow_html=True)
        cfg = data.load_json("brandfit_config.json") or {}
        w = cfg.get("component_weights", {})
        for k, v in w.items():
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:6px 0;"
                f"border-bottom:1px solid {LINE};font-size:13px'>"
                f"<span>{k.replace('_', ' ').capitalize()}</span>"
                f"<span class='n-num'>{v:.0%}</span></div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='n-muted' style='margin-top:10px'>Weights are fixed in "
            f"<code>src/models/brandfit.py</code>. They are configuration, not a "
            f"learned parameter — there is no label for “was this a good fit”, so "
            f"there is nothing to learn them from.</div>", unsafe_allow_html=True)
        for g in cfg.get("gates", []):
            st.markdown(f"<div style='font-size:12.5px;color:{INK_2};margin-top:6px'>"
                        f"• {ui.esc(g)}</div>", unsafe_allow_html=True)

with b:
    with st.container(border=True):
        st.markdown(ui.section("Build"), unsafe_allow_html=True)
        for lbl, val in [
            ("Creators", f"{meta.get('n_creators', 0):,}"),
            ("Campaigns", f"{meta.get('n_campaigns', 0)}"),
            ("Requests", f"{meta.get('n_requests', 0)}"),
        ]:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:6px 0;"
                f"font-size:13px'><span style='color:{INK_2}'>{lbl}</span>"
                f"<span class='n-num'>{val}</span></div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(ui.section("Not implemented"), unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size:13px;color:{INK_2};line-height:1.7'>"
            f"Billing, team seats, notification preferences, API keys and SSO are "
            f"part of the product design and are <b>not built</b>. They are listed "
            f"here rather than rendered as dead toggles, because a settings page full "
            f"of switches that do nothing is worse than one that says so."
            f"</div>", unsafe_allow_html=True)
