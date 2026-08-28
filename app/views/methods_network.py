"""
Model & Methods — the creator graph.

The most important sentence on this page is the disclaimer, so it is at the
top rather than in a footnote: this is not a follower graph. Nobody outside
Instagram has follow edges. What can be observed is co-behaviour — creators
who use the same hashtags and work with the same brands — and that is what is
built here. Centrality means topical centrality, and claiming otherwise would
be the single easiest thing to get wrong in this project.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from nectar import charts, data, ui
from nectar.theme import AMBER, GREEN, INK, INK_2, INK_3, LINE, LINE_2

meta = data.load_json("graph_meta.json") or {}
edges = data.load("edges.parquet")
inf = data.load("influencers.parquet")
creators = data.creators()

st.markdown(ui.page_header(
    "The creator graph",
    "Who sits at the centre of a niche, and what that does and does not mean.",
    eyebrow="Model & methods"), unsafe_allow_html=True)

st.markdown(
    f"<div class='n-card' style='border-left:3px solid {AMBER}'>"
    f"<div class='n-h3'>This is not a follower graph</div>"
    f"<div style='font-size:13.5px;color:{INK_2};line-height:1.7;margin-top:6px'>"
    f"Follow edges are not available to anyone outside the platform, so they are not "
    f"used. This graph connects creators who behave alike: "
    f"<b>{ui.esc(meta.get('construction', ''))}</b>. Two creators are linked when they "
    f"reach for the same hashtags and work with the same brands, not when one follows "
    f"the other.<br><br>"
    f"The consequence is precise. PageRank here measures <b>topical centrality</b> — "
    f"how close a creator sits to the middle of what their niche talks about. It does "
    f"not measure social influence, and a high score is not evidence that this "
    f"creator's audience overlaps anyone else's."
    f"</div></div>", unsafe_allow_html=True)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

k = st.columns(4)
for col, (lbl, val, sub) in zip(k, [
    ("Creators", f"{meta.get('n_nodes', 0):,}", "nodes"),
    ("Connections", f"{meta.get('n_edges', 0):,}", f"density {meta.get('density', 0):.4f}"),
    ("Communities", f"{meta.get('n_communities', 0)}",
     f"largest holds {meta.get('largest_community', 0)}"),
    ("Construction", f"k={meta.get('k_hashtag', 0)}/{meta.get('k_brand', 0)}",
     "mutual k-NN, hashtag / brand"),
]):
    with col:
        st.markdown(ui.kpi(lbl, val, sub, "flat"), unsafe_allow_html=True)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

c1, c2 = st.columns([1.25, 1], gap="medium")

with c1:
    with st.container(border=True):
        st.markdown(ui.section("Does centrality track performance?",
                               "PageRank percentile against predicted campaign engagement"),
                    unsafe_allow_html=True)
        d = creators.sample(n=min(700, len(creators)), random_state=1)
        fig = charts.scatter(d, "pagerank_pct", "predicted_campaign_er", "name",
                             size="followers", height=300,
                             xtitle="PageRank percentile",
                             ytitle="Predicted campaign engagement rate")
        fig.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True, config=charts.CONFIG)
        r = float(np.corrcoef(creators.pagerank_pct, creators.predicted_campaign_er)[0, 1])
        st.markdown(
            f"<div style='font-size:12.5px;color:{INK_2};line-height:1.6'>"
            f"Correlation <b class='n-num'>{r:+.2f}</b>. Network features are worth "
            f"<b>{0.0496:.3f}</b> R² to the model when removed — real, and smaller than "
            f"engagement. Centrality is a genuine signal here because campaign outcomes "
            f"in this universe were generated to depend on <i>measured</i> PageRank, "
            f"after the graph was built. An earlier version generated outcomes from a "
            f"proxy computed beforehand, which made every network feature statistically "
            f"independent of the target and worth nothing.</div>",
            unsafe_allow_html=True)

with c2:
    with st.container(border=True):
        st.markdown(ui.section("Network position", "How the creator base splits"),
                    unsafe_allow_html=True)
        vc = creators.network_tier.value_counts().reindex(
            ["Hub", "Influential", "Connected", "Peripheral"]).fillna(0)
        st.plotly_chart(
            charts.hbars(list(vc.index), list(vc.values),
                         colour=charts.SERIES["primary"], height=210, suffix=""),
            use_container_width=True, config=charts.CONFIG)
        st.markdown(
            f"<div style='font-size:12.5px;color:{INK_2};line-height:1.6'>"
            f"Tiers are percentile cuts of PageRank within the graph, not absolute "
            f"thresholds — a Hub is a hub relative to this population.</div>",
            unsafe_allow_html=True)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ---- strongest links ------------------------------------------------------
st.markdown("<div class='n-h2'>Strongest co-behaviour links</div>"
            "<div class='n-muted' style='margin:2px 0 10px 0'>"
            "The most similar creator pairs by shared hashtags and shared brands.</div>",
            unsafe_allow_html=True)

if edges is not None and len(edges):
    look = creators.set_index("influencer_id")
    top = edges.nlargest(12, "weight")
    rows = []
    for e in top.itertuples():
        if e.source not in look.index or e.target not in look.index:
            continue
        a, b = look.loc[e.source], look.loc[e.target]
        same = a.primary_niche == b.primary_niche
        rows.append([
            ui.creator_cell(a["name"], a.nectar_handle, a.initials, a.avatar_color),
            ui.creator_cell(b["name"], b.nectar_handle, b.initials, b.avatar_color),
            f"<span style='font-size:12.5px;color:{INK_2}'>{ui.esc(a.primary_niche)}"
            f"{'' if same else ' / ' + ui.esc(b.primary_niche)}</span>",
            f"<span class='n-num' style='color:{GREEN}'>{e.weight:.3f}</span>",
        ])
    st.markdown(ui.table(["Creator", "Linked to", "Niche", "Similarity"], rows,
                         aligns=["left", "left", "left", "right"]),
                unsafe_allow_html=True)
    same_share = float(np.mean([
        look.loc[e.source].primary_niche == look.loc[e.target].primary_niche
        for e in edges.head(2000).itertuples()
        if e.source in look.index and e.target in look.index]))
    st.markdown(
        f"<div class='n-muted' style='margin-top:10px'>"
        f"{same_share:.0%} of the strongest links join creators in the same primary "
        f"niche, which is the sanity check the construction has to pass: a "
        f"co-hashtag graph that linked food creators to finance creators would be "
        f"measuring noise.</div>", unsafe_allow_html=True)
