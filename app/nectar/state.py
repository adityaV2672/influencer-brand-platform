"""Session state: who you are signed in as, and what you have selected.

Kept in one place because three different pages can change the same value
(picking a campaign on Campaigns, on Discover, or on Reporting) and a
scattered set of st.session_state pokes is how those get out of step.
"""
from __future__ import annotations

from types import SimpleNamespace

import streamlit as st

from . import data

DEFAULTS = {
    "role": None,              # None -> the sign-in screen
    "campaign_id": None,
    "creator_id": None,
    "shortlist": None,         # set of influencer_id
    "open_deal": None,         # request_id shown in the Deals thread pane
    "toast": None,
}


def init() -> None:
    """Idempotent. Home.py calls this on every run, and the accessors below call
    it defensively so a page is still renderable on its own - which is what the
    page-render tests do, and what a deep link into a fresh session does."""
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if st.session_state["shortlist"] is None:
        st.session_state["shortlist"] = set()
    if st.session_state["campaign_id"] is None:
        live = data.campaigns()
        active = live[live.status == "Live"]
        st.session_state["campaign_id"] = (
            active.campaign_id.iloc[0] if len(active) else live.campaign_id.iloc[0]
        )
    if st.session_state["creator_id"] is None:
        st.session_state["creator_id"] = data.meta().get("default_creator_id")


def _ensure() -> None:
    if "shortlist" not in st.session_state or st.session_state.get("campaign_id") is None:
        init()


def role() -> str | None:
    return st.session_state.get("role")


def set_role(r: str | None) -> None:
    st.session_state["role"] = r


def _row(df, mask):
    """Return one row as an attribute bag rather than a pandas Series.

    A Series already owns the attribute `.name` (it means the Series' own
    label), so `campaign.name` silently returned the row's integer index
    instead of the campaign's name and the page rendered "Ranked for 0."
    A SimpleNamespace has no reserved attributes, so column access by dot
    means exactly what it looks like.
    """
    sub = df[mask]
    row = sub.iloc[0] if len(sub) else df.iloc[0]
    return SimpleNamespace(**row.to_dict())


def campaign():
    """The campaign every brand-side page is scoped to."""
    _ensure()
    c = data.campaigns()
    return _row(c, c.campaign_id == st.session_state.get("campaign_id"))


def signed_in_creator():
    _ensure()
    c = data.creators()
    return _row(c, c.influencer_id == st.session_state.get("creator_id"))


def shortlist() -> set:
    _ensure()
    return st.session_state["shortlist"]


def toggle_shortlist(influencer_id: str) -> bool:
    _ensure()
    s = st.session_state["shortlist"]
    if influencer_id in s:
        s.discard(influencer_id)
        return False
    s.add(influencer_id)
    return True


def flash(message: str) -> None:
    st.session_state["toast"] = message


def drain_toast() -> str | None:
    msg = st.session_state.get("toast")
    st.session_state["toast"] = None
    return msg
