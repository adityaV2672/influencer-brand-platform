"""Creator OS — Profile. What a brand sees, and what the creator controls."""
from __future__ import annotations

import streamlit as st

from nectar import creator_ctx as ctx
from nectar import data, ui
from nectar.theme import AMBER, GREEN, INK, INK_2, INK_3, LINE, LINE_2

me = ctx.me()
peers, peer_label = ctx.peers()

hdr, act = st.columns([3.3, 1])
with hdr:
    tick = (f"<span style='color:{GREEN};font-size:17px;margin-left:7px'>✓</span>"
            if bool(me.verified) else "")
    st.markdown(
        f"<div style='display:flex;align-items:flex-start;gap:16px'>"
        f"{ui.avatar(me.initials, me.avatar_color, 66)}"
        f"<div><div class='n-h1' style='font-size:27px'>{ui.esc(me.name)}{tick}</div>"
        f"<div class='n-sub'>{ui.esc(me.nectar_handle)} · {ui.esc(me.city)}</div>"
        f"<div style='font-size:13.5px;color:{INK_2};margin-top:6px'>{ui.esc(me.bio)}</div>"
        f"<div style='margin-top:10px'>"
        + "".join(ui.tag(c) for c in list(me.categories))
        + "".join(ui.tag(p) for p in list(me.platform_names))
        + "</div></div></div>", unsafe_allow_html=True)
with act:
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.button("Edit profile", use_container_width=True, key="edit_profile")
    st.markdown(
        f"<div style='text-align:center;font-size:12px;color:{GREEN};margin-top:6px'>"
        f"{'✓ Verified' if me.verified else 'Not verified'}</div>",
        unsafe_allow_html=True)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

plat = ui.platforms_of(me)
tiles = [(k, ui.count(v), "followers") for k, v in list(plat.items())[:2]]
tiles += [("Engagement", f"{me.engagement_rate * 100:.1f}%", "avg rate"),
          ("Growth", f"{me.follower_growth_rate * 100:+.1f}%", "per month")]
cols = st.columns(len(tiles))
for col, (lbl, val, sub) in zip(cols, tiles):
    with col:
        st.markdown(
            f"<div class='n-card' style='text-align:center;padding:14px'>"
            f"<div style='font-size:12.5px;color:{INK_3}'>{ui.esc(lbl)}</div>"
            f"<div class='n-num' style='font-size:23px;margin:2px 0'>{ui.esc(val)}</div>"
            f"<div style='font-size:11.5px;color:{INK_3}'>{ui.esc(sub)}</div></div>",
            unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

tabs = st.tabs(["Performance", "Audience", "Content", "Rates", "Availability"])


def row(label, value, benchmark=None, good=True):
    b = (f"<span style='font-size:12px;color:{INK_3};margin-right:14px'>"
         f"Benchmark: {benchmark}</span>" if benchmark else "")
    return (f"<div style='display:flex;justify-content:space-between;align-items:baseline;"
            f"padding:13px 2px;border-bottom:1px solid {LINE_2}'>"
            f"<span style='font-size:13.5px'>{ui.esc(label)}</span>"
            f"<span>{b}<b class='n-num' style='color:{GREEN if good else AMBER}'>"
            f"{value}</b></span></div>")


with tabs[0]:
    pm = peers.median(numeric_only=True)
    html = [
        row("Avg engagement", f"{me.engagement_rate * 100:.1f}%",
            f"{pm.engagement_rate * 100:.1f}%", me.engagement_rate >= pm.engagement_rate),
        row("Avg reach / post", ui.count(me.avg_reach), ui.count(pm.avg_reach),
            me.avg_reach >= pm.avg_reach),
        row("Avg views / video", ui.count(me.avg_views), ui.count(pm.avg_views),
            me.avg_views >= pm.avg_views),
        row("Monthly growth", f"{me.follower_growth_rate * 100:+.1f}%",
            f"{pm.follower_growth_rate * 100:+.1f}%",
            me.follower_growth_rate >= pm.follower_growth_rate),
        row("Posts / month", f"{me.posting_frequency_month:.0f}",
            f"{pm.posting_frequency_month:.0f}",
            me.posting_frequency_month >= pm.posting_frequency_month),
    ]
    st.markdown("".join(html), unsafe_allow_html=True)
    st.markdown(f"<div class='n-muted' style='margin-top:10px'>"
                f"Benchmarks are the median of {len(peers):,} {ui.esc(peer_label)}.</div>",
                unsafe_allow_html=True)

with tabs[1]:
    a, b = st.columns(2)
    with a:
        st.markdown("<div class='n-h3' style='margin-bottom:10px'>Age</div>",
                    unsafe_allow_html=True)
        for x in me.audience_age:
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:12px;margin-bottom:8px'>"
                f"<span class='n-num' style='width:52px;font-size:12px'>{ui.esc(x['range'])}</span>"
                f"{ui.bar(x['pct'] / 100, INK, width='100%')}"
                f"<span class='n-num' style='width:36px;text-align:right;font-size:12px'>"
                f"{x['pct']}%</span></div>", unsafe_allow_html=True)
    with b:
        st.markdown("<div class='n-h3' style='margin-bottom:10px'>Top locations</div>",
                    unsafe_allow_html=True)
        for x in me.audience_locations:
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:12px;margin-bottom:8px'>"
                f"<span style='width:96px;font-size:12.5px'>{ui.esc(x['city'])}</span>"
                f"{ui.bar(x['pct'] / 100, GREEN, width='100%')}"
                f"<span class='n-num' style='width:36px;text-align:right;font-size:12px'>"
                f"{x['pct']}%</span></div>", unsafe_allow_html=True)

with tabs[2]:
    posts = data.load("posts_sample.parquet")
    mine = posts[posts.influencer_id == me.influencer_id] if posts is not None else None
    st.markdown(
        f"<div class='n-muted' style='margin-bottom:12px'>"
        f"Recent captions with the labels the NLP pipeline assigned. Sentiment is "
        f"RoBERTa; the irony probability is the CardiffNLP irony model — the signal "
        f"the lexicon methods could not produce.</div>", unsafe_allow_html=True)
    if mine is None or mine.empty:
        st.markdown(ui.empty_state("✎", "No captions on file.",
                                   "Post-level NLP output is not available for this creator."),
                    unsafe_allow_html=True)
    else:
        rows = []
        for p in mine.head(8).itertuples():
            irony = float(p.roberta_p_irony or 0)
            rows.append([
                f"<div style='font-size:13px;max-width:460px'>{ui.esc(p.caption)[:170]}</div>",
                ui.chip(str(p.roberta_sentiment).title()),
                f"<span class='n-num' style='color:{AMBER if irony > 0.5 else INK_3}'>"
                f"{irony:.2f}</span>",
                f"<span style='font-size:12.5px;color:{INK_2}'>{ui.esc(p.topic_label)}</span>",
                f"<span class='n-num'>{ui.count(p.likes)}</span>",
            ])
        st.markdown(ui.table(["Caption", "Sentiment", "P(irony)", "Topic", "Likes"], rows,
                             aligns=["left", "left", "right", "left", "right"]),
                    unsafe_allow_html=True)

with tabs[3]:
    st.markdown(
        row("Reel", ui.inr(me.rate_reel)) +
        row("Story", ui.inr(me.rate_story)) +
        row("Carousel", ui.inr(me.rate_carousel)) +
        row("Suggested range",
            f"{ui.inr(me.price_low_inr)} – {ui.inr(me.price_high_inr)}"),
        unsafe_allow_html=True)
    st.markdown(
        f"<div class='n-muted' style='margin-top:12px'>These are the fee model's "
        f"predictions for a creator with your reach, engagement and niche — the same "
        f"numbers a brand sees. Story and Carousel are priced off the Reel rate at "
        f"0.35× and 0.70×.</div>", unsafe_allow_html=True)

with tabs[4]:
    fg, bg = ((GREEN, "#E8F4F0") if me.availability == "Available"
              else (AMBER, "#FBF3E0"))
    st.markdown(
        f"<div style='background:{bg};border-radius:12px;padding:16px 18px'>"
        f"<div style='font-size:12.5px;color:{INK_2}'>Current status</div>"
        f"<div style='font-size:20px;font-weight:700;color:{fg}'>"
        f"{ui.esc(me.availability)}</div>"
        f"<div style='font-size:13px;color:{INK_2};margin-top:4px'>"
        f"{ui.esc(me.available_window)}</div></div>",
        unsafe_allow_html=True)
    st.markdown(
        f"<div class='n-muted' style='margin-top:14px'>Availability is a hard filter "
        f"on the brand side. A creator marked unavailable is excluded from search "
        f"results regardless of how well they score.</div>", unsafe_allow_html=True)
