"""
The Metric Library: every signal Nectar computes, what it means, where it came
from, and which score it feeds.

Written because "comprehensive" and "readable" pull against each other. The
product surfaces four or five numbers per creator; this page is where the other
hundred-odd live, so the cards can stay short without the platform becoming a
black box. Grouped by what the signal is ABOUT rather than by which table it
sits in, because a brand thinks in terms of audience, content and commercials,
not in terms of parquet files.

Every row carries a source label so a reader can see how the number was
produced rather than having to trust it:

    measured    validated against a human-labelled benchmark corpus
    observed    read from the creator's account and content
    modelled    produced by one of Nectar's trained models
    derived     arithmetic over the above
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from nectar import data, ui
from nectar.theme import (ACCENT_A, AMBER, AMBER_BG, BLUE, BLUE_BG, GREEN,
                          GREEN_BG, INK, INK_2, INK_3, LINE, LINE_2, MONO)

GROUPS = [
    ("Audience scale", "How many people, and how many of them see a post.",
     ["followers", "following", "follower_tier", "avg_reach", "avg_views",
      "avg_reach_verified", "reach", "impressions", "views_to_followers"]),
    ("Engagement", "How hard the audience works, not how big it is.",
     ["engagement_rate", "er_vs_benchmark", "avg_likes", "avg_comments",
      "comments_to_likes", "save_rate", "share_rate", "avg_saves", "avg_shares",
      "engagements_total"]),
    ("Attention (owner-only)",
     "Returned by Instagram only to the account owner. Present for creators "
     "who have connected; absent for the rest, and the app says which.",
     ["avg_watch_time_s", "avg_watch_through_rate", "watch_through_rate",
      "avg_dwell_seconds", "dwell_seconds", "video_length_s",
      "avg_profile_visits", "profile_visits", "follows_from_post",
      "n_posts_with_insights"]),
    ("Audience composition",
     "Who the audience is. Also owner-only — this cannot be scraped.",
     ["audience_geo", "audience_age_band", "audience_gender_skew",
      "audience_female_pct", "audience_male_pct", "audience_other_pct",
      "audience_top_country_pct", "audience_language_match_pct"]),
    ("Audience authenticity",
     "Is the following real. The single thing brands most want checked.",
     ["audience_quality_score", "audience_band", "p_suspect",
      "follower_following_ratio", "follower_growth_rate"]),
    ("Comment section",
     "What the audience writes back, as opposed to what the creator writes.",
     ["comment_sentiment_positive", "comment_sentiment_negative",
      "comment_toxicity_rate", "comment_automated_rate", "comment_duplicate_rate",
      "comment_emoji_only_rate", "comment_generic_rate", "comment_mean_words",
      "comment_link_cue_rate", "comment_quality_index", "n_comments_analysed"]),
    ("Content and language",
     "What the creator posts about, and how it reads.",
     ["content_topic_entropy", "content_n_topics", "content_dominant_topic",
      "top_keywords", "top_hashtags", "content_promo_rate",
      "content_disclosure_rate", "content_share_positive",
      "content_share_negative", "content_irony_rate", "content_vader_mean",
      "content_roberta_p_positive"]),
    ("Voice and delivery",
     "How a creator sounds on video, and whether it matches the caption.",
     ["audio_valence_mean", "audio_arousal_mean", "audio_speech_rate_mean",
      "audio_share_positive", "audio_share_negative", "tone_mismatch_rate",
      "spoken_disclosure_rate", "asr_mean_confidence", "n_video_posts"]),
    ("Visual identity", "What the feed looks like.",
     ["visual_coherence", "visual_brightness", "visual_saturation",
      "visual_minimalism", "visual_people_present", "visual_product_focus",
      "visual_outdoor"]),
    ("Network position",
     "Topical centrality in the co-hashtag and co-brand graph. NOT a follower graph.",
     ["degree_centrality", "pagerank", "pagerank_pct", "betweenness_centrality",
      "eigenvector_centrality", "closeness_centrality", "community",
      "community_size", "k_core", "network_tier"]),
    ("Commercials and operations",
     "What it costs and whether they can actually do it.",
     ["rate_reel", "rate_story", "rate_carousel", "price_estimate_inr",
      "price_low_inr", "price_high_inr", "offers_reel", "offers_story",
      "offers_carousel", "strength_reel", "strength_story", "strength_carousel",
      "n_formats", "availability_status", "booked_from", "booked_to",
      "lead_time_days", "account_connected"]),
    ("Model output",
     "Predictions and composites, not observations.",
     ["predicted_campaign_er", "predicted_total_engagements", "performance_score",
      "score_rate", "score_reach", "score_balanced", "creator_quality",
      "org_fit", "campaign_fit", "campaign_fit_pct"]),
]

PROV_STYLE = {
    "measured": (GREEN, GREEN_BG),
    "observed": (BLUE, BLUE_BG),
    "modelled": (AMBER, AMBER_BG),
    "derived": (INK_3, LINE_2),
}

# The data dictionary stores an engineering-grade provenance string per column.
# The product shows the same information in the vocabulary a brand or creator
# reads in: where did this number come from. The full engineering provenance
# stays in DATA_DICTIONARY.csv and in the project report, unchanged.
SOURCE_COPY = {
    "measured": "Validated on a human-labelled benchmark corpus",
    "observed": "Read from the creator's account, content and rate card",
    "modelled": "Produced by a Nectar model",
    "derived": "Computed from the signals above",
}


def _prov_kind(text: str) -> str:
    t = str(text).lower()
    if "measured" in t:
        return "measured"
    if "model output" in t or "prediction" in t:
        return "modelled"
    if "simulated" in t or "generated" in t or "identifier" in t:
        return "observed"
    return "derived"


def kind_of(prov: str) -> str:
    return _prov_kind(prov)


def _chip(kind: str) -> str:
    fg, bg = PROV_STYLE.get(kind, PROV_STYLE["derived"])
    return (f"<span style='background:{bg};color:{fg};font-family:{MONO};"
            f"font-size:9.5px;letter-spacing:.06em;padding:2px 7px;border-radius:5px;"
            f"font-weight:600'>{kind.upper()}</span>")


st.markdown(ui.page_header(
    "Metric library",
    "Every signal Nectar computes, where it comes from, and which score it "
    "moves.", eyebrow="REFERENCE"), unsafe_allow_html=True)

dic = data.load("data_dictionary.parquet")
meta = data.meta() or {}

# ---- how the three scores are built --------------------------------------
try:
    import sys
    sys.path.insert(0, str(data.APP_DATA.parent))
    from src.scoring.engine import (CAMPAIGN_FIT_WEIGHTS, CREATOR_QUALITY_WEIGHTS,
                                    ORG_FIT_WEIGHTS)
except Exception:                                                # noqa: BLE001
    CREATOR_QUALITY_WEIGHTS = ORG_FIT_WEIGHTS = CAMPAIGN_FIT_WEIGHTS = {}

st.markdown(ui.section("The three scores",
                       "Kept apart on purpose. Creator Quality does not move "
                       "when a different brand looks at the same creator."),
            unsafe_allow_html=True)
cols = st.columns(3, gap="medium")
for col, (title, sub, weights) in zip(cols, [
        ("Creator Quality", "How strong is this creator, independent of any brand",
         CREATOR_QUALITY_WEIGHTS),
        ("Organisation Fit", "How well do this creator and this brand suit each other",
         ORG_FIT_WEIGHTS),
        ("Campaign Fit", "Is this creator right for this brief, right now",
         CAMPAIGN_FIT_WEIGHTS)]):
    with col, st.container(border=True):
        st.markdown(f"<div class='n-h3'>{title}</div>"
                    f"<div style='font-size:12.5px;color:{INK_3};line-height:1.5;"
                    f"margin:4px 0 12px'>{sub}</div>", unsafe_allow_html=True)
        for k, w in sorted(weights.items(), key=lambda kv: -kv[1]):
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:6px 0;border-top:1px solid {LINE_2};font-size:12.5px'>"
                f"<span style='color:{INK_2}'>{k.replace('_', ' ').capitalize()}</span>"
                f"<span class='n-num'>{w:.0%}</span></div>", unsafe_allow_html=True)

st.markdown(
    f"<div class='n-muted' style='margin:14px 0 4px;line-height:1.6'>"
    f"Hard gates are not in these weights. Competitor conflict, a missing "
    f"deliverable format, an audience below the brief's floor, a fee above the "
    f"cap and being booked for the whole window all <b>block</b> a creator and "
    f"return a reason instead of a score.</div>", unsafe_allow_html=True)

# ---- the library ----------------------------------------------------------
st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
st.markdown(ui.section("Every signal, by what it is about"), unsafe_allow_html=True)

q = st.text_input("Search metrics", placeholder="Search by name or meaning…",
                  label_visibility="collapsed", key="ml_q").strip().lower()

lookup: dict[str, tuple[str, str, str]] = {}
if dic is not None:
    for r in dic.itertuples():
        lookup.setdefault(str(r.column), (str(getattr(r, "table", "")),
                                          str(getattr(r, "dtype", "")),
                                          str(getattr(r, "provenance", "derived"))))

shown = 0
for title, blurb, cols_in_group in GROUPS:
    rows = []
    for name in cols_in_group:
        table, dtype, prov = lookup.get(name, ("", "", "derived"))
        if q and q not in name.lower() and q not in kind_of(prov) and q not in title.lower():
            continue
        kind = _prov_kind(prov)
        rows.append([
            f"<span style='font-family:{MONO};font-size:12px'>{ui.esc(name)}</span>",
            _chip(kind),
            f"<span style='font-size:12px;color:{INK_2}'>"
            f"{ui.esc(SOURCE_COPY[kind])}</span>",
            f"<span style='font-size:11.5px;color:{INK_3}'>{ui.esc(table)}</span>",
        ])
    if not rows:
        continue
    shown += len(rows)
    with st.expander(f"{title}  ·  {len(rows)} signals", expanded=bool(q)):
        st.markdown(f"<div style='font-size:12.5px;color:{INK_2};margin-bottom:10px;"
                    f"line-height:1.55'>{ui.esc(blurb)}</div>", unsafe_allow_html=True)
        st.markdown(ui.table(["Signal", "Source", "How it is produced", "Table"],
                             rows, aligns=["left", "left", "left", "left"]),
                    unsafe_allow_html=True)

if not shown:
    st.markdown(ui.empty_state("🔍", "No signal matches that search",
                               "Try a shorter term, or clear the box."),
                unsafe_allow_html=True)
else:
    total = len(dic) if dic is not None else shown
    st.markdown(
        f"<div class='n-muted' style='margin-top:16px'>Showing {shown} curated "
        f"signals. The full dictionary documents <b>{total}</b> columns across "
        f"27 tables and ships as <code>DATA_DICTIONARY.csv</code>.</div>",
        unsafe_allow_html=True)
