"""Help. What this is, what is real, and how to run it."""
from __future__ import annotations

import streamlit as st

from nectar import data, ui
from nectar.theme import AMBER, GREEN, INK_2, INK_3, LINE

meta = data.meta()
prov = meta.get("provenance", {})

st.markdown(ui.page_header("Help", "What this is, and what each number means."),
            unsafe_allow_html=True)

with st.container(border=True):
    st.markdown(
        f"<div class='n-h3'>What Nectar is</div>"
        f"<div style='font-size:13.5px;color:{INK_2};line-height:1.7;margin-top:6px'>"
        f"A two-sided influencer marketing platform. Brands describe a campaign and "
        f"get a ranked, safety-screened shortlist of creators with a reason attached "
        f"to every recommendation. Creators see the same scores about themselves, "
        f"benchmarked against the people they actually compete with, and negotiate "
        f"directly. The ranking is a trained model, not a follower count."
        f"</div>", unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
st.markdown("<div class='n-h2'>Where each number comes from</div>"
            "<div style='height:10px'></div>", unsafe_allow_html=True)

for label, key in [
    ("Campaign Fit", "campaign_fit"),
    ("Organisation Fit", "org_fit"),
    ("Fees and rate cards", "fees"),
    ("Predicted vs actual", "predicted_vs_actual"),
]:
    if key not in prov:
        continue
    with st.container(border=True):
        st.markdown(
            f"<div class='n-h3'>{ui.esc(label)}</div>"
            f"<div style='font-size:13px;color:{INK_2};line-height:1.65;margin-top:4px'>"
            f"{ui.esc(prov[key])}</div>", unsafe_allow_html=True)

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

FAQ = [
    ("Where does a creator's data come from?",
     "Public profile signals are read for every creator. Saves, shares, watch time "
     "and audience demographics are returned by Instagram only to the account "
     "owner, so those arrive when a creator connects their account. Creators who "
     "have connected are marked as verified; the rest are shown with inferred "
     "figures and labelled as such."),
    ("How accurate is the language analysis?",
     "The sentiment and toxicity models are trained and scored on human-annotated "
     "corpora — TweetEval and the Misra sarcasm headlines — before they are "
     "applied to anything on the platform. Their held-out accuracy is reported on "
     "the Metric library page."),
    ("Why does a creator get blocked instead of a low score?",
     "Because some constraints are contractual rather than preferences. A "
     "competitor exclusivity clause, a format the creator does not produce, or a "
     "campaign window they are booked through cannot be outweighed by a good fit "
     "score, so the creator is removed with a reason instead of ranked lower."),
    ("Is the network a follower graph?",
     "No, and this matters. Follow edges are not available outside the platform. The "
     "graph links creators who use the same hashtags and work with the same brands, "
     "so centrality means topical centrality, not social influence."),
    ("Why does Reporting show actuals above predictions on every campaign?",
     "The performance model is trained on log engagement rate. Back-transforming a "
     "prediction from log space systematically understates the mean, which is most of "
     "the gap. It is a property of the target transform, not a lucky quarter."),
    ("Why is the shortlist so small on some briefs?",
     "Two gates run before fit is considered: the brief's per-creator fee cap and its "
     "minimum audience size. Find creators shows the eligible pool updating "
     "live as those are set, so the consequence is visible before you commit."),
]
st.markdown("<div class='n-h2'>Questions</div><div style='height:8px'></div>",
            unsafe_allow_html=True)
for q, a in FAQ:
    with st.expander(q):
        st.markdown(f"<div style='font-size:13.5px;color:{INK_2};line-height:1.7'>"
                    f"{ui.esc(a)}</div>", unsafe_allow_html=True)

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
with st.container(border=True):
    st.markdown(
        f"<div class='n-h3'>Rebuilding it</div>"
        f"<div style='font-size:13px;color:{INK_2};line-height:1.8;margin-top:6px'>"
        f"<code>python run_pipeline.py</code> — regenerate the universe, run the NLP, "
        f"build the graph, train the models<br>"
        f"<code>python -m src.features.export_app</code> — write the dashboard payload<br>"
        f"<code>python -m src.features.export_nectar</code> — build the product layer<br>"
        f"<code>python -m src.features.export_csv</code> — write every table to "
        f"<code>data/csv/</code><br>"
        f"<code>python -m pytest tests/</code> — integrity checks and page renders"
        f"</div>", unsafe_allow_html=True)
