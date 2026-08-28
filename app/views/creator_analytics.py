"""
Creator OS — Analytics. How this creator compares with the people they are
actually competing against.

The peer set is same niche AND same follower tier. Most public "creator
benchmarks" compare a nano food creator with a macro tech creator and call the
difference performance.
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from nectar import charts, creator_ctx as ctx
from nectar import data, ui
from nectar.theme import AMBER, GREEN, INK, INK_2, INK_3, LINE, LINE_2

me = ctx.me()
peers, peer_label = ctx.peers()
cat_fit = ctx.my_category_fit()

st.markdown(ui.page_header("Analytics",
                           f"Benchmarked against {len(peers):,} {peer_label}."),
            unsafe_allow_html=True)

bench = float(me.er_vs_benchmark) if me.er_vs_benchmark == me.er_vs_benchmark else 1.0
pr_er = ctx.percentile("engagement_rate")

k = st.columns(4)
for col, (lbl, val, sub, tone) in zip(k, [
    ("Engagement rate", f"{me.engagement_rate * 100:.1f}%",
     f"{pr_er:.0f}th percentile in peer group", "good" if pr_er >= 50 else "warn"),
    ("Vs benchmark", f"{bench:.2f}×", "published rate for this size", 
     "good" if bench >= 1 else "warn"),
    ("Monthly growth", f"{me.follower_growth_rate * 100:+.1f}%",
     f"{ctx.percentile('follower_growth_rate'):.0f}th percentile", "good"),
    ("Comment ratio", f"{me.comments_to_likes:.3f}",
     f"{ctx.percentile('comments_to_likes'):.0f}th percentile", "good"),
]):
    with col:
        st.markdown(ui.kpi(lbl, val, sub, tone), unsafe_allow_html=True)

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

with st.container(border=True):
    a, b = st.columns([1, 3.4])
    with a:
        st.markdown(
            f"<div class='n-eyebrow'>Your benchmark</div>"
            f"<div class='n-num' style='font-size:40px;color:{GREEN};line-height:1.2'>"
            f"{bench:.2f}×</div>", unsafe_allow_html=True)
    with b:
        verdict = ("more attractive to brands targeting quality over reach"
                   if bench >= 1 else "a signal to work on engagement before rate")
        st.markdown(
            f"<div class='n-h2'>Your engagement is {bench:.2f}× the benchmark "
            f"for your niche.</div>"
            f"<div style='font-size:13.5px;color:{INK_2};margin-top:6px;line-height:1.55'>"
            f"Among {ui.esc(peer_label)}, your engagement puts you in the "
            f"top {max(1, 100 - pr_er):.0f}%. That makes you {verdict}.</div>",
            unsafe_allow_html=True)

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

c1, c2 = st.columns(2, gap="medium")

with c1:
    with st.container(border=True):
        st.markdown(ui.section("Where you sit", "Percentile within your peer group"),
                    unsafe_allow_html=True)
        metrics = [("Engagement rate", "engagement_rate"),
                   ("Comment ratio", "comments_to_likes"),
                   ("View-through rate", "views_to_followers"),
                   ("Growth rate", "follower_growth_rate"),
                   ("Posting frequency", "posting_frequency_month")]
        names, vals = [], []
        for label, col in metrics:
            v = ctx.percentile(col)
            if not np.isnan(v):
                names.append(label)
                vals.append(v)
        colours = [GREEN if v >= 60 else AMBER if v >= 35 else "#C2413F" for v in vals]
        fig = charts.hbars(names, vals, height=250, suffix="")
        fig.data[0].marker.color = colours[::-1]
        fig.add_vline(x=50, line_width=1, line_dash="dot", line_color=INK_3)
        st.plotly_chart(fig, use_container_width=True, config=charts.CONFIG)
        st.markdown(f"<div style='font-size:11.5px;color:{INK_3}'>"
                    "Green ≥ 60th percentile · amber 35–60 · red below 35.</div>",
                    unsafe_allow_html=True)

with c2:
    with st.container(border=True):
        st.markdown(ui.section("Audience age", "Distribution across age groups"),
                    unsafe_allow_html=True)
        for bucket in me.audience_age:
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:12px;margin-bottom:8px'>"
                f"<span class='n-num' style='width:52px;font-size:12px;color:{INK_2}'>"
                f"{ui.esc(bucket['range'])}</span>"
                f"{ui.bar(bucket['pct'] / 100, INK, width='100%')}"
                f"<span class='n-num' style='width:38px;text-align:right;font-size:12px'>"
                f"{bucket['pct']}%</span></div>", unsafe_allow_html=True)
        g = me.audience_gender
        st.markdown(
            f"<div style='background:{LINE_2};border-radius:11px;padding:11px 13px;"
            f"margin-top:10px'><div style='font-size:12px;color:{INK_2};"
            f"margin-bottom:4px'>Gender split</div>"
            + " ".join(f"<span style='font-size:12.5px;margin-right:16px'>"
                       f"<b class='n-num'>{x['pct']}%</b> {ui.esc(x['label'])}</span>"
                       for x in g)
            + "</div>", unsafe_allow_html=True)

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
st.markdown("<div class='n-h2'>Where you fit best</div>"
            "<div class='n-muted' style='margin:2px 0 12px 0'>"
            "Your brand-fit score across brand categories — the same score brands "
            "see when they search.</div>", unsafe_allow_html=True)

top = cat_fit.head(6)
cols = st.columns(len(top) if len(top) else 1)
for col, r in zip(cols, top.itertuples()):
    fg, bg = (GREEN, "#E8F4F0") if r.fit_pct >= 70 else (AMBER, "#FBF3E0")
    with col:
        st.markdown(
            f"<div style='background:{bg};border-radius:12px;padding:14px 14px;"
            f"text-align:center'>"
            f"<div class='n-num' style='font-size:26px;color:{fg}'>{r.fit_pct:.0f}%</div>"
            f"<div style='font-size:12.5px;color:{INK_2};font-weight:600;margin-top:2px'>"
            f"{ui.esc(r.category)}</div>"
            f"<div style='font-size:11px;color:{INK_3}'>{int(r.brands)} brands</div></div>",
            unsafe_allow_html=True)
