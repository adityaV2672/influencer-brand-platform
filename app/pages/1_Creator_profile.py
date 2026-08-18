"""Single-creator deep dive: score breakdown, content intelligence, network, price."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from theme import (
    DIVERGING, GRID, INK, NETWORK_TIER_COLOR, SERIES, SEQ_BLUE, STATUS,
    fmt_count, fmt_inr, fmt_pct, load, locked, metric_row, page_header,
    plotly_layout, require, sidebar_tier,
)

st.set_page_config(page_title="Creator profile", page_icon="◎", layout="wide")
cfg = sidebar_tier()

inf = require("influencers.parquet", "Creator database")
posts = load("posts_sample.parquet")

page_header("Creator profile", "Every number here traces back to a feature the model actually used.")

# ---- selection -------------------------------------------------------------
c1, c2 = st.columns([2, 1])
with c1:
    options = inf.sort_values("performance_score", ascending=False)
    labels = {
        r["influencer_id"]: f"{r['handle']}  ·  {r['primary_niche']}  ·  {fmt_count(r['followers'])} followers"
        for _, r in options.iterrows()
    }
    sel = st.selectbox("Creator", list(labels), format_func=lambda k: labels[k], label_visibility="collapsed")
with c2:
    st.caption(f"{len(inf):,} creators indexed")

r = inf[inf["influencer_id"] == sel].iloc[0]

# ==========================================================================
# Headline
# ==========================================================================
st.markdown(
    f"<div style='font-size:1.35rem;font-weight:650;color:{INK['primary']}'>{r['handle']}</div>"
    f"<div style='color:{INK['secondary']}'>{r['primary_niche']}"
    + (f" · also {r['secondary_niche']}" if pd.notna(r.get("secondary_niche")) else "")
    + f" · {r['follower_tier']} tier · audience {r['audience_geo']}, {r['audience_age_band']}</div>",
    unsafe_allow_html=True,
)
st.write("")

score_display = f"{r['performance_score']:.0f} / 100" if cfg["numeric_score"] else r["performance_band"]
metric_row([
    ("Performance score", score_display,
     "percentile vs all creators" if cfg["numeric_score"] else "🔒 numeric score is paid"),
    ("Followers", fmt_count(r["followers"]), f"{r['follower_growth_rate'] * 100:+.1f}%/mo growth"),
    ("Engagement rate", fmt_pct(r["engagement_rate"]),
     f"{r['er_vs_benchmark']:.2f}× benchmark" if pd.notna(r.get("er_vs_benchmark")) else ""),
    ("Est. fee / post",
     f"{fmt_inr(r['price_low_inr'])} – {fmt_inr(r['price_high_inr'])}" if cfg["price_band"] else "🔒",
     "80% prediction interval" if cfg["price_band"] else "paid feature"),
])
st.divider()

tab_perf, tab_content, tab_net, tab_posts = st.tabs(
    ["Performance", "Content intelligence", "Network position", "Recent posts"]
)

# ==========================================================================
# Performance
# ==========================================================================
with tab_perf:
    a, b = st.columns([1, 1])

    with a:
        st.markdown("**How this creator compares on each pillar**")
        if not cfg["numeric_score"]:
            locked("The feature-level breakdown of the score is available on the paid plan.",
                   "Score breakdown")
        else:
            def pct_of(col: str) -> float:
                if col not in inf.columns or pd.isna(r.get(col)):
                    return np.nan
                return float((inf[col] < r[col]).mean() * 100)

            rows = [
                ("Reach", pct_of("followers")),
                ("Engagement rate", pct_of("engagement_rate")),
                ("Engagement quality", pct_of("comments_to_likes")),
                ("Growth momentum", pct_of("follower_growth_rate")),
                ("Audience retention", pct_of("views_to_followers")),
                ("Network centrality", pct_of("pagerank")),
                ("Posting consistency", pct_of("posting_frequency_month")),
            ]
            rows = [(k, v) for k, v in rows if not np.isnan(v)]
            names = [k for k, _ in rows][::-1]
            vals = [v for _, v in rows][::-1]

            fig = go.Figure(go.Bar(
                x=vals, y=names, orientation="h",
                marker=dict(color=SERIES[0], line=dict(width=0)),
                text=[f"{v:.0f}" for v in vals], textposition="outside",
                hovertemplate="%{y}: %{x:.0f}th percentile<extra></extra>",
            ))
            fig.add_vline(x=50, line_width=1, line_dash="dot", line_color=INK["muted"])
            fig.update_xaxes(range=[0, 108])
            plotly_layout(fig, height=310, showlegend=False, xtitle="Percentile vs all creators")
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            st.caption("Dotted line = median creator. Bars right of it are above average.")

    with b:
        st.markdown("**Engagement vs the published benchmark**")
        vsb = r.get("er_vs_benchmark", np.nan)
        if pd.notna(vsb):
            g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=float(vsb),
                number={"suffix": "×", "font": {"size": 34, "color": INK["primary"]}},
                gauge={
                    "axis": {"range": [0, 3], "tickwidth": 1, "tickcolor": INK["muted"]},
                    "bar": {"color": SERIES[0], "thickness": 0.7},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 0.75], "color": "#f4f3f0"},
                        {"range": [0.75, 1.25], "color": "#e9e8e4"},
                        {"range": [1.25, 3], "color": "#dcdbd6"},
                    ],
                    "threshold": {"line": {"color": INK["primary"], "width": 2},
                                  "thickness": 0.85, "value": 1.0},
                },
            ))
            plotly_layout(g, height=230, showlegend=False)
            st.plotly_chart(g, width="stretch", config={"displayModeBar": False})
            st.caption(
                "1.0× is exactly the published 2026 average for this follower tier and niche. "
                "This normalises away the fact that small accounts always look better on raw engagement."
            )

        st.markdown("**Reach funnel**")
        stages = ["Followers", "Avg views", "Avg reach", "Avg engagements"]
        vals = [r["followers"], r["avg_views"], r["avg_reach"], r["avg_likes"] + r["avg_comments"]]
        f = go.Figure(go.Bar(
            x=stages, y=vals,
            marker=dict(color=[SEQ_BLUE[i] for i in (2, 3, 4, 5)]),
            text=[fmt_count(v) for v in vals], textposition="outside",
        ))
        f.update_yaxes(type="log")
        plotly_layout(f, height=250, showlegend=False, ytitle="Count (log)")
        st.plotly_chart(f, width="stretch", config={"displayModeBar": False})

# ==========================================================================
# Content intelligence
# ==========================================================================
with tab_content:
    has_nlp = any(c.startswith("content_") for c in inf.columns)
    if not has_nlp:
        st.warning("Content features are not built yet. Run `python run_pipeline.py --only nlp features export`.")
    else:
        a, b = st.columns([1, 1])

        with a:
            st.markdown("**Tone mix across recent posts**")
            shares = {k: r.get(f"content_share_{k}", np.nan) for k in ("positive", "neutral", "negative")}
            shares = {k: v for k, v in shares.items() if pd.notna(v)}
            if shares:
                order = ["positive", "neutral", "negative"]
                cmap = {"positive": SERIES[2], "neutral": INK["muted"], "negative": SERIES[7]}
                keys = [k for k in order if k in shares]
                fig = go.Figure()
                for k in keys:
                    fig.add_bar(
                        x=[shares[k] * 100], y=["Tone"], orientation="h", name=k.title(),
                        marker=dict(color=cmap[k], line=dict(width=2, color="#fcfcfb")),
                        hovertemplate=f"{k.title()}: %{{x:.1f}}%<extra></extra>",
                    )
                fig.update_layout(barmode="stack")
                fig.update_xaxes(range=[0, 100], ticksuffix="%")
                plotly_layout(fig, height=150)
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            else:
                st.caption("Sentiment shares unavailable.")

            irony = r.get("content_irony_rate", np.nan)
            if pd.notna(irony):
                st.markdown("**Irony / sarcasm**")
                st.progress(float(np.clip(irony, 0, 1)),
                            text=f"{irony:.0%} of posts flagged ironic by the transformer detector")
                st.caption(
                    "Detected with a RoBERTa model fine-tuned on SemEval irony data. "
                    "Lexicon methods score near chance on this task — see *Model & Methods*."
                )

            st.markdown("**Commercial signals**")
            sig = [
                ("Ad load", r.get("content_promo_rate", np.nan), "share of posts that are promotional"),
                ("Formal disclosure", r.get("content_disclosure_rate", np.nan), "#ad / paid-partnership tags"),
                ("Call-to-action", r.get("content_cta_rate", np.nan), "link in bio, swipe, comment below"),
                ("Questions", r.get("content_question_rate", np.nan), "community-engagement style"),
            ]
            for label, val, help_ in sig:
                if pd.notna(val):
                    st.progress(float(np.clip(val, 0, 1)), text=f"{label} — {val:.0%}")
                    st.caption(help_)

        with b:
            st.markdown("**Emotional profile (NRC lexicon)**")
            emos = ["joy", "trust", "anticipation", "surprise", "sadness", "fear", "anger", "disgust"]
            vals = [r.get(f"content_nrc_{e}", np.nan) for e in emos]
            if any(pd.notna(v) for v in vals):
                vals = [0 if pd.isna(v) else float(v) for v in vals]
                fig = go.Figure(go.Scatterpolar(
                    r=vals + [vals[0]], theta=[e.title() for e in emos] + [emos[0].title()],
                    fill="toself", fillcolor="rgba(42,120,214,0.18)",
                    line=dict(color=SERIES[0], width=2),
                    hovertemplate="%{theta}: %{r:.3f}<extra></extra>",
                ))
                fig.update_layout(
                    polar=dict(
                        bgcolor="rgba(0,0,0,0)",
                        radialaxis=dict(visible=True, gridcolor=GRID, tickfont=dict(size=9)),
                        angularaxis=dict(gridcolor=GRID, tickfont=dict(size=10)),
                    ),
                )
                plotly_layout(fig, height=330, showlegend=False)
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
                st.caption(
                    "Mohammad & Turney (2013) NRC Word-Emotion Association Lexicon — "
                    "the 8-emotion model the supervisor recommended over binary polarity."
                )

            kws = str(r.get("top_keywords", "") or "")
            if kws:
                st.markdown("**What they actually talk about**")
                st.markdown(
                    " ".join(
                        f"<span style='background:#eef3fb;color:{INK['primary']};padding:3px 9px;"
                        f"border-radius:11px;font-size:0.83rem;display:inline-block;margin:2px'>{k}</span>"
                        for k in kws.split("|")[:14] if k
                    ),
                    unsafe_allow_html=True,
                )
            tags = str(r.get("top_hashtags", "") or "")
            if tags:
                st.markdown("**Most-used hashtags**")
                st.markdown(
                    " ".join(
                        f"<span style='background:#f4f3f0;color:{INK['secondary']};padding:3px 9px;"
                        f"border-radius:11px;font-size:0.83rem;display:inline-block;margin:2px'>#{t}</span>"
                        for t in tags.split("|")[:14] if t
                    ),
                    unsafe_allow_html=True,
                )

# ==========================================================================
# Network
# ==========================================================================
with tab_net:
    if not cfg["network"]:
        locked("Network position, community membership and the centrality breakdown "
               "are available on the paid plan.", "Network intelligence")
    elif "pagerank" not in inf.columns:
        st.warning("Network features are not built yet. Run `python run_pipeline.py --only sna`.")
    else:
        a, b = st.columns([1, 1])
        with a:
            tier = r.get("network_tier", "—")
            st.markdown(
                f"<div style='font-size:0.78rem;color:{INK['muted']};text-transform:uppercase;"
                f"letter-spacing:0.04em'>Network position</div>"
                f"<div style='font-size:1.6rem;font-weight:650;"
                f"color:{NETWORK_TIER_COLOR.get(tier, INK['primary'])}'>{tier}</div>"
                f"<div style='color:{INK['secondary']};font-size:0.9rem'>"
                f"community #{int(r['community'])} · {int(r['community_size'])} creators</div>",
                unsafe_allow_html=True,
            )
            st.write("")
            metrics = [
                ("PageRank", "pagerank"),
                ("Degree centrality", "degree_centrality"),
                ("Eigenvector centrality", "eigenvector_centrality"),
                ("Betweenness centrality", "betweenness_centrality"),
                ("Closeness centrality", "closeness_centrality"),
            ]
            names, vals = [], []
            for label, col in metrics:
                if col in inf.columns and pd.notna(r.get(col)):
                    names.append(label)
                    vals.append(float((inf[col] < r[col]).mean() * 100))
            fig = go.Figure(go.Bar(
                x=vals[::-1], y=names[::-1], orientation="h",
                marker=dict(color=SERIES[6]),
                text=[f"{v:.0f}" for v in vals[::-1]], textposition="outside",
                hovertemplate="%{y}: %{x:.0f}th percentile<extra></extra>",
            ))
            fig.update_xaxes(range=[0, 108])
            plotly_layout(fig, height=260, showlegend=False, xtitle="Percentile")
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        with b:
            st.info(
                "**What these numbers mean here.** The graph is built from *co-behaviour* — "
                "creators who share rare hashtags and work with the same brands — because "
                "no platform exposes follower edges to third parties.\n\n"
                "So PageRank measures **topical centrality**: how embedded a creator is in a "
                "category's shared vocabulary. It is a genuine matching signal, but it is not "
                "a claim about who follows whom."
            )
            peers = inf[(inf["community"] == r["community"]) & (inf["influencer_id"] != sel)]
            if len(peers):
                st.markdown(f"**Closest peers in community #{int(r['community'])}**")
                st.dataframe(
                    peers.nlargest(8, "pagerank")[["handle", "primary_niche", "followers", "engagement_rate"]]
                    .rename(columns={"handle": "Creator", "primary_niche": "Niche",
                                     "followers": "Followers", "engagement_rate": "Engagement"}),
                    hide_index=True, width="stretch",
                    column_config={
                        "Followers": st.column_config.NumberColumn(format="compact"),
                        "Engagement": st.column_config.NumberColumn(format="percent"),
                    },
                )

# ==========================================================================
# Posts
# ==========================================================================
with tab_posts:
    if posts is None:
        st.warning("Post samples are not exported. Run `python run_pipeline.py --only export`.")
    else:
        p = posts[posts["influencer_id"] == sel]
        if not len(p):
            st.caption("No sampled posts for this creator.")
        else:
            st.caption(f"{len(p)} highest-engagement recent posts, with per-post model output.")
            for _, row in p.iterrows():
                sent = row.get("roberta_sentiment") or row.get("vader_label") or "—"
                iro = row.get("roberta_p_irony", np.nan)
                chips = []
                cmap = {"positive": SERIES[2], "negative": SERIES[7], "neutral": INK["muted"]}
                chips.append((str(sent).title(), cmap.get(str(sent), INK["muted"])))
                if pd.notna(iro) and float(iro) >= 0.5:
                    chips.append((f"Ironic {float(iro):.0%}", SERIES[6]))
                if row.get("has_promo"):
                    chips.append(("Promotional", SERIES[1]))
                if row.get("has_disclosure"):
                    chips.append(("Disclosed #ad", SERIES[3]))
                if row.get("topic_label") and str(row.get("topic_label")) != "outlier":
                    chips.append((str(row["topic_label"])[:34], SERIES[0]))

                with st.container(border=True):
                    st.markdown(
                        " ".join(
                            f"<span style='background:{c}1f;color:{c};padding:2px 9px;border-radius:10px;"
                            f"font-size:0.75rem;font-weight:560;margin-right:4px'>{t}</span>"
                            for t, c in chips
                        ),
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div style='color:{INK['primary']};margin:6px 0 4px 0'>{row['caption']}</div>"
                        f"<div style='color:{INK['muted']};font-size:0.8rem'>"
                        f"{fmt_count(row['likes'])} likes · {fmt_count(row['comments'])} comments · "
                        f"{int(row['days_ago'])} days ago</div>",
                        unsafe_allow_html=True,
                    )
