"""Shared context for the Creator OS: who am I, and what is addressed to me."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from . import data, state


def me():
    return state.signed_in_creator()


def my_requests(include_history: bool = True) -> pd.DataFrame:
    """Everything addressed to this creator.

    Two sources, deliberately combined: requests from the six showcase
    campaigns, and the creator's own brand history. A creator's inbox is not
    limited to whatever campaigns one brand happens to be running.
    """
    iid = me().influencer_id
    live = data.requests()
    live = live[live.influencer_id == iid].copy()
    live["source"] = "campaign"
    if not include_history:
        return live.sort_values("stage_index", ascending=False)

    hist = data.load("nectar_creator_history.parquet")
    if hist is None or hist.empty:
        return live.sort_values("stage_index", ascending=False)
    hist = hist[hist.influencer_id == iid].copy()
    hist["source"] = "history"
    for col in set(live.columns) - set(hist.columns):
        hist[col] = None
    for col in set(hist.columns) - set(live.columns):
        live[col] = None
    both = pd.concat([live, hist[live.columns]], ignore_index=True)
    return both.sort_values("stage_index", ascending=False)


def my_fit() -> pd.DataFrame:
    """Every brief this creator has been scored against, whether or not the
    brand has approached them. This is what the creator-side Discover page
    shows: briefs you match, not brands you already know."""
    f = data.fit()
    return f[f.influencer_id == me().influencer_id].sort_values("rank_best")


def my_category_fit() -> pd.DataFrame:
    cf = data.load("nectar_category_fit.parquet")
    if cf is None:
        return pd.DataFrame(columns=["category", "fit_pct", "brands"])
    return cf[cf.influencer_id == me().influencer_id].sort_values(
        "fit_pct", ascending=False)


def my_earnings() -> pd.DataFrame:
    e = data.earnings()
    return e[e.influencer_id == me().influencer_id]


def peers() -> tuple[pd.DataFrame, str]:
    """Same niche AND same follower tier. Comparing a nano food creator with a
    macro tech creator is meaningless, which is what most public 'benchmarks'
    do."""
    c = data.creators()
    m = me()
    p = c[(c.primary_niche == m.primary_niche) & (c.follower_tier == m.follower_tier)]
    if len(p) >= 12:
        return p, f"{m.follower_tier}-tier {m.primary_niche} creators"
    p = c[c.primary_niche == m.primary_niche]
    return p, f"{m.primary_niche} creators (all sizes)"


def percentile(col: str) -> float:
    p, _ = peers()
    m = me()
    v = getattr(m, col, None)
    if col not in p.columns or v is None or pd.isna(v):
        return float("nan")
    return float((p[col] < v).mean() * 100)


def creator_picker(key: str = "who") -> None:
    """Sign in as any creator. The default is whoever has the most live
    conversations, so the demo never opens on an empty inbox."""
    c = data.creators()
    r = data.requests()
    busy = r.influencer_id.value_counts()
    ranked = c.assign(_n=c.influencer_id.map(busy).fillna(0)).sort_values(
        ["_n", "followers"], ascending=[False, False])
    ids = list(ranked.influencer_id)
    labels = dict(zip(ranked.influencer_id,
                      ranked.name + "  ·  " + ranked.primary_niche))
    cur = me().influencer_id
    idx = ids.index(cur) if cur in ids else 0
    with st.sidebar:
        pick = st.selectbox("Signed in as", ids, index=idx,
                            format_func=lambda i: labels.get(i, i), key=key)
    if pick != cur:
        st.session_state["creator_id"] = pick
        st.rerun()
