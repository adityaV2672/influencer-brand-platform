"""Interactive network map of the creator graph."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from theme import (
    GRID, INK, NETWORK_TIER_COLOR, SERIES,
    fmt_count, load, locked, page_header, plotly_layout, require, sidebar_tier,
)

st.set_page_config(page_title="Network map", page_icon="◎", layout="wide")
cfg = sidebar_tier()

page_header("Network map", "Who sits at the centre of a category's shared vocabulary.")

if not cfg["network"]:
    locked("The network map, community structure and centrality view are paid features.",
           "Network intelligence")
    st.stop()

inf = require("influencers.parquet", "Creator database")
edges = load("edges.parquet")

if edges is None or "pagerank" not in inf.columns:
    st.warning("Network artifacts not found. Run `python run_pipeline.py --only sna export`.")
    st.stop()

c1, c2, c3 = st.columns([1.4, 1, 1])
with c1:
    niche = st.selectbox("Focus on a niche", ["All niches"] + sorted(inf["primary_niche"].unique()))
with c2:
    max_nodes = st.slider("Creators shown", 80, 600, 260, 20)
with c3:
    color_by = st.radio("Colour by", ["Network position", "Niche"], horizontal=True)

# ---- subgraph --------------------------------------------------------------
sub = inf if niche == "All niches" else inf[inf["primary_niche"] == niche]
sub = sub.nlargest(min(max_nodes, len(sub)), "pagerank")
ids = set(sub["influencer_id"])
e = edges[edges["source"].isin(ids) & edges["target"].isin(ids)]

if len(e) < 3:
    st.warning("Too few connections among the selected creators. Widen the selection.")
    st.stop()

# ---- layout ----------------------------------------------------------------
@st.cache_data(show_spinner="Computing layout…")
def layout(nodes: tuple, src: tuple, tgt: tuple, wts: tuple, seed: int = 7) -> dict:
    import networkx as nx

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(zip(src, tgt, wts))
    return nx.spring_layout(G, seed=seed, k=1.6 / np.sqrt(max(len(nodes), 2)), iterations=60, weight="weight")


pos = layout(tuple(sub["influencer_id"]), tuple(e["source"]), tuple(e["target"]), tuple(e["weight"]))

# ---- edges -----------------------------------------------------------------
ex, ey = [], []
for s, t in zip(e["source"], e["target"]):
    if s in pos and t in pos:
        ex += [pos[s][0], pos[t][0], None]
        ey += [pos[s][1], pos[t][1], None]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=ex, y=ey, mode="lines",
    line=dict(width=0.6, color="rgba(11,11,11,0.13)"),
    hoverinfo="skip", showlegend=False,
))

# ---- nodes -----------------------------------------------------------------
sub = sub[sub["influencer_id"].isin(pos)].copy()
sub["x"] = [pos[i][0] for i in sub["influencer_id"]]
sub["y"] = [pos[i][1] for i in sub["influencer_id"]]
sub["size"] = np.clip(sub["pagerank"] / sub["pagerank"].max() * 26, 7, 30)

if color_by == "Network position":
    groups, cmap = "network_tier", NETWORK_TIER_COLOR
    order = ["Hub", "Influential", "Connected", "Peripheral"]
else:
    groups = "primary_niche"
    uniq = sorted(sub["primary_niche"].unique())
    # Fixed slot order, never cycled: past 8 categories, remainder folds to "Other".
    cmap = {n: SERIES[i] for i, n in enumerate(uniq[:8])}
    for n in uniq[8:]:
        cmap[n] = INK["muted"]
    order = uniq

for g in order:
    s = sub[sub[groups] == g]
    if not len(s):
        continue
    fig.add_trace(go.Scatter(
        x=s["x"], y=s["y"], mode="markers", name=str(g),
        marker=dict(size=s["size"], color=cmap.get(g, INK["muted"]),
                    line=dict(width=1.4, color="#fcfcfb"), opacity=0.9),
        customdata=np.stack([s["handle"], s["primary_niche"], s["followers"],
                             s["engagement_rate"], s["community"]], axis=-1),
        hovertemplate=("<b>%{customdata[0]}</b><br>%{customdata[1]}<br>"
                       "%{customdata[2]:,} followers<br>"
                       "engagement %{customdata[3]:.2%}<br>"
                       "community #%{customdata[4]}<extra></extra>"),
    ))

fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
plotly_layout(fig, height=620)
fig.update_layout(hovermode="closest")
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

st.caption(
    f"{len(sub):,} creators · {len(e):,} connections · marker size = PageRank. "
    "Edges are shared rare hashtags and shared brand collaborations, **not** follower relationships."
)

# ---- communities -----------------------------------------------------------
st.divider()
a, b = st.columns([1, 1])
with a:
    st.markdown("**Largest communities in view**")
    comm = (
        sub.groupby("community")
        .agg(creators=("influencer_id", "count"),
             median_followers=("followers", "median"),
             median_engagement=("engagement_rate", "median"),
             top_niche=("primary_niche", lambda s: s.mode().iloc[0] if len(s) else "—"))
        .sort_values("creators", ascending=False).head(10).reset_index()
    )
    st.dataframe(
        comm.rename(columns={"community": "Community", "creators": "Creators",
                             "median_followers": "Median followers",
                             "median_engagement": "Median engagement", "top_niche": "Dominant niche"}),
        hide_index=True, width="stretch",
        column_config={
            "Median followers": st.column_config.NumberColumn(format="compact"),
            "Median engagement": st.column_config.NumberColumn(format="percent"),
        },
    )
with b:
    st.markdown("**Most central creators in view**")
    top = sub.nlargest(10, "pagerank")[["handle", "primary_niche", "followers", "network_tier", "pagerank_pct"]]
    st.dataframe(
        top.rename(columns={"handle": "Creator", "primary_niche": "Niche", "followers": "Followers",
                            "network_tier": "Position", "pagerank_pct": "PageRank pctile"}),
        hide_index=True, width="stretch",
        column_config={
            "Followers": st.column_config.NumberColumn(format="compact"),
            "PageRank pctile": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
        },
    )
