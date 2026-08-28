"""The application shell: the dark sidebar, the role switch, and the sign-in screen."""
from __future__ import annotations

import streamlit as st

from . import data, state
from .theme import CSS, GRADIENT, INK, LINE
from .ui import esc, logo_svg


def inject_css() -> None:
    # Collapse blank lines: a blank line inside the markdown block terminates
    # the raw-HTML span and Streamlit escapes the rest of the stylesheet, which
    # renders the whole design system as visible text on the page.
    css = "\n".join(line for line in CSS.splitlines() if line.strip())
    st.markdown(css, unsafe_allow_html=True)


def _brand_mark() -> str:
    return (f"<div class='nectar-brand'>{logo_svg(21)}"
            "<span class='wordmark'>nectar</span></div>")


def render_sidebar(nav: dict) -> None:
    """nav = {"main": [(page, label, icon)...], "methods": [...]} """
    with st.sidebar:
        st.markdown(_brand_mark(), unsafe_allow_html=True)

        for page, label, icon in nav["main"]:
            st.page_link(page, label=label, icon=icon)

        st.markdown("<div class='nectar-navgroup'>Model &amp; methods</div>",
                    unsafe_allow_html=True)
        for page, label, icon in nav["methods"]:
            st.page_link(page, label=label, icon=icon)

        st.markdown("<div class='nectar-sidebar-rule'></div>", unsafe_allow_html=True)
        for page, label, icon in nav["footer"]:
            st.page_link(page, label=label, icon=icon)

        if state.role() == "creator":
            _creator_picker()
        _account_block()


def _creator_picker() -> None:
    """Sign in as any creator. Lives in the shell rather than in each page so
    it renders above the account block instead of being appended to the bottom
    of the sidebar after the page has run."""
    creators = data.creators()
    reqs = data.requests()
    busy = reqs.influencer_id.value_counts()
    ranked = creators.assign(_n=creators.influencer_id.map(busy).fillna(0)) \
        .sort_values(["_n", "followers"], ascending=[False, False])
    ids = list(ranked.influencer_id)
    labels = dict(zip(ranked.influencer_id, ranked["name"]))
    cur = st.session_state.get("creator_id")
    idx = ids.index(cur) if cur in ids else 0
    st.markdown("<div class='nectar-navgroup'>Signed in as</div>", unsafe_allow_html=True)
    pick = st.selectbox("Creator", ids, index=idx, label_visibility="collapsed",
                        format_func=lambda i: labels.get(i, i), key="creator_pick")
    if pick != cur:
        st.session_state["creator_id"] = pick
        st.rerun()


def _account_block() -> None:
    role = state.role()
    if role == "brand":
        camp = state.campaign()
        name, sub, initials = camp.brand_name, "Marketing", "".join(
            w[0] for w in str(camp.brand_name).split()[:2]).upper()
        switch = "Switch to Creator OS"
        target = "creator"
    else:
        me = state.signed_in_creator()
        name, sub, initials = me.name, "Creator", me.initials
        switch = "Switch to Brand OS"
        target = "brand"

    st.markdown(
        f"<div class='nectar-account'><span class='av'>{esc(initials)}</span>"
        f"<div><div class='nm'>{esc(name)}</div><div class='rl'>{esc(sub)}</div></div></div>",
        unsafe_allow_html=True,
    )
    if st.button(switch, key="role_switch", use_container_width=True):
        state.set_role(target)
        st.switch_page(st.session_state["_home_page"] if "_home_page" in st.session_state
                       else "views/brand_overview.py")


# --------------------------------------------------------------------------
# Sign-in
# --------------------------------------------------------------------------

SIGNIN_CSS = f"""
<style>
[data-testid="stSidebar"] {{ display: none !important; }}
[data-testid="stMain"] .block-container {{ max-width: 860px; padding-top: 4.5rem; }}
.signin-hero {{ text-align: center; }}
.signin-mark {{
    display: inline-flex; align-items: center; gap: 11px; margin-bottom: 26px;
}}
.signin-mark .drop {{
    width: 30px; height: 30px; border-radius: 50% 50% 50% 3px;
    transform: rotate(45deg); background: {GRADIENT};
    display: inline-flex; align-items: center; justify-content: center;
}}
.signin-mark .drop::after {{
    content: ""; width: 11px; height: 11px; border-radius: 50%;
    background: {INK}; transform: rotate(-45deg);
}}
.signin-mark .word {{ font-size: 30px; font-weight: 700; letter-spacing: -0.035em; }}
.signin-h {{ font-size: 44px; font-weight: 800; letter-spacing: -0.035em; line-height: 1.1; }}
.signin-p {{ font-size: 15.5px; color: #4A4247; margin: 12px auto 0; max-width: 380px; }}
.signin-card {{
    background: #fff; border: 1px solid {LINE}; border-bottom: none;
    border-radius: 16px 16px 0 0; padding: 24px 24px 4px 24px;
}}
/* pull the real Streamlit button up so the card and its action read as one
   object. Streamlit cannot place a widget inside an HTML string, so the seam
   is closed with geometry instead. */
[data-testid="stMain"] .stButton:has(button[key$="_role"]),
[data-testid="stMain"] div:has(> div > .stButton) {{ }}
.signin-btn-wrap .stButton > button {{ border-radius: 0 0 16px 16px !important; }}
.signin-ic {{
    width: 42px; height: 42px; border-radius: 11px; border: 1px solid {LINE};
    display: flex; align-items: center; justify-content: center;
    font-size: 17px; margin-bottom: 16px;
}}
.signin-foot {{
    text-align: center; font-family: 'JetBrains Mono', monospace; font-size: 10.5px;
    letter-spacing: 0.08em; color: #B0A8AE; margin-top: 42px;
}}
</style>
"""


_STROKE = "stroke='#4A4247' stroke-width='1.6' fill='none' stroke-linecap='round' stroke-linejoin='round'"


def _icon_briefcase() -> str:
    return ("<div class='signin-ic'><svg width='19' height='19' viewBox='0 0 24 24'>"
            f"<rect x='2.5' y='7' width='19' height='13' rx='2.5' {_STROKE}/>"
            f"<path d='M8.5 7V5.2A1.7 1.7 0 0 1 10.2 3.5h3.6A1.7 1.7 0 0 1 15.5 5.2V7' {_STROKE}/>"
            "</svg></div>")


def _icon_person() -> str:
    return ("<div class='signin-ic'><svg width='19' height='19' viewBox='0 0 24 24'>"
            f"<circle cx='12' cy='8' r='3.8' {_STROKE}/>"
            f"<path d='M4.5 20.5c0-4 3.4-6.2 7.5-6.2s7.5 2.2 7.5 6.2' {_STROKE}/>"
            "</svg></div>")


def render_signin() -> None:
    st.markdown(SIGNIN_CSS, unsafe_allow_html=True)
    st.markdown(
        "<div class='signin-hero'>"
        f"<div class='signin-mark'>{logo_svg(32)}<span class='word'>nectar</span></div>"
        "<div class='signin-h'>Influence, without the middleman.</div>"
        "<div class='signin-p'>Find the right creators. Understand why. Run the campaign.</div>"
        "</div><div style='height:34px'></div>",
        unsafe_allow_html=True,
    )
    _, c1, c2, _ = st.columns([0.6, 1, 1, 0.6])
    with c1:
        st.markdown(
            f"<div class='signin-card'>{_icon_briefcase()}"
            "<div style='font-size:17px;font-weight:700;letter-spacing:-0.01em'>Brand / Agency</div>"
            "<div style='font-size:13.5px;color:#4A4247;margin:6px 0 18px 0'>"
            "Find creators and run campaigns.</div></div>",
            unsafe_allow_html=True)
        if st.button("Continue as Brand", type="primary", use_container_width=True,
                     key="signin_brand"):
            state.set_role("brand")
            st.rerun()
    with c2:
        st.markdown(
            f"<div class='signin-card'>{_icon_person()}"
            "<div style='font-size:17px;font-weight:700;letter-spacing:-0.01em'>Creator</div>"
            "<div style='font-size:13.5px;color:#4A4247;margin:6px 0 18px 0'>"
            "Get discovered and manage partnerships.</div></div>",
            unsafe_allow_html=True)
        if st.button("Continue as Creator", type="primary", use_container_width=True,
                     key="signin_creator"):
            state.set_role("creator")
            st.rerun()

    st.markdown(
        "<div class='signin-foot'>SYNTHETIC DATA · MODEL-DRIVEN SCORING · "
        "NLP VALIDATED ON REAL CORPORA</div>",
        unsafe_allow_html=True)
