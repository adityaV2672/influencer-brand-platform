"""Shared context for the Creator OS: who am I, and what is addressed to me."""
from __future__ import annotations

import numpy as np
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
    _align_missing(hist, live)
    _align_missing(live, hist)
    # Drop empty sides before concatenating. A creator with no live requests
    # (or no history) would otherwise hand pandas an all-NA frame, and pandas
    # warns that it will stop inferring dtypes from the non-empty side.
    frames = [d for d in (live, hist[live.columns]) if not d.empty]
    if not frames:
        return live
    both = frames[0].copy() if len(frames) == 1 else pd.concat(frames, ignore_index=True)
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


def _align_missing(target: pd.DataFrame, other: pd.DataFrame) -> None:
    """Give `target` the columns it lacks, typed like `other`'s.

    Filling with a bare None makes an all-object column of nulls, and pandas
    then warns that a future version will stop inferring the concatenated
    dtype from the non-empty side - a live `followers` of int64 would silently
    become object. Numeric columns are filled with NaN as float instead, which
    is a dtype that can actually hold the missing value.
    """
    for col in [c for c in other.columns if c not in target.columns]:
        if pd.api.types.is_numeric_dtype(other[col]) and not pd.api.types.is_bool_dtype(other[col]):
            target[col] = np.full(len(target), np.nan, dtype="float64")
        else:
            target[col] = pd.Series([None] * len(target), index=target.index, dtype="object")
