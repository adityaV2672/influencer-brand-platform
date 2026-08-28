"""
Model & Methods — the performance and price models.

This page exists because the product surface deliberately hides its own
machinery. Every number a brand sees on Discover comes from here, and a
scoring product that cannot show its working is a black box with a nice font.
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from nectar import charts, data, ui
from nectar.theme import AMBER, GREEN, INK, INK_2, INK_3, LINE, LINE_2

res = data.load_json("model_results.json") or {}
perf = res.get("performance", {})
price = res.get("price", {})
imp = data.load("feature_importance.parquet")
shap = data.load("feature_shap.parquet")

st.markdown(ui.page_header(
    "The model", "What predicts campaign performance, how well, and how we know.",
    eyebrow="Model & methods"), unsafe_allow_html=True)

r2 = perf.get("r2_log", 0)
ceiling = perf.get("theoretical_r2_log_ceiling", 0)
frac = perf.get("fraction_of_ceiling", 0)

struct = perf.get("baseline_structural", {}) or {}
struct_r2 = struct.get("r2_log")
learnable = perf.get("share_of_learnable_signal")

k = st.columns(4)
for col, (lbl, val, sub, tone) in zip(k, [
    ("R² (log target)", f"{r2:.3f}", f"{perf.get('n_rows', 0):,} campaigns, nested CV", "good"),
    ("Structural baseline", f"{struct_r2:.3f}" if struct_r2 is not None else "—",
     "arithmetic, no learning", "flat"),
    ("Share of learnable signal",
     f"{learnable:.0%}" if learnable is not None else "—",
     "lift over structure, vs ceiling", "good"),
    ("Rank quality", f"{perf.get('ndcg@10_within_brief', 0):.2f}",
     "NDCG@10 within a brief", "good"),
]):
    with col:
        st.markdown(ui.kpi(lbl, val, sub, tone), unsafe_allow_html=True)

st.markdown(
    f"<div class='n-card' style='margin-top:16px'>"
    f"<div class='n-h3'>Two different ceilings, and why the honest one is lower</div>"
    f"<div style='font-size:13.5px;color:{INK_2};line-height:1.65;margin-top:6px'>"
    f"Campaign outcomes carry a deliberate lognormal(0, 0.32) shock that nothing "
    f"observable can explain, which caps R² at <b>{ceiling:.3f}</b>. Against that, "
    f"this model reaches {frac:.0%}. But that ceiling belongs to an <i>oracle</i> "
    f"that knows the hidden creator traits, and this model does not — so quoting "
    f"{frac:.0%} flatters it.<br><br>"
    f"The fair reference is the <b>structural baseline</b>: a five-term regression "
    f"on the parts of the generator's own formula that are visible in the feature "
    f"table (the published engagement curve for this size and category, category "
    f"match, geography match, age match, and the measured share of amplification). "
    f"That is arithmetic a reader could do with a calculator, and it scores "
    f"<b>{struct_r2:.3f}</b>. The model's genuine contribution — inferring the "
    f"hidden traits from noisy observable behaviour — is the "
    f"<b>{perf.get('learned_lift_over_structure', 0):+.3f}</b> on top, which is "
    f"<b>{learnable:.0%}</b> of the headroom that was actually available."
    f"</div></div>"
    if struct_r2 is not None else
    f"<div class='n-card' style='margin-top:16px'><div class='n-h3'>Ceiling</div>"
    f"<div style='font-size:13.5px;color:{INK_2};margin-top:6px'>"
    f"Ceiling {ceiling:.3f}; model at {frac:.0%} of it.</div></div>",
    unsafe_allow_html=True)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ---- baselines ------------------------------------------------------------
st.markdown("<div class='n-h2'>Against what baseline?</div>"
            "<div class='n-muted' style='margin:2px 0 12px 0'>"
            "A model is only as good as the thing it beats.</div>",
            unsafe_allow_html=True)

bb = perf.get("baseline_benchmark_curve", {})
bc = perf.get("baseline_composite_index", {})
rows = [
    ["<b>Published benchmark curve</b><div style='font-size:12px;color:#8A8289'>"
     "engagement rate predicted from follower count alone</div>",
     f"<span class='n-num'>{bb.get('r2_log', 0):+.3f}</span>",
     f"<span class='n-num'>{bb.get('spearman', 0):.2f}</span>",
     f"<span class='n-num'>{bb.get('ndcg@10_global', 0):.2f}</span>"],
    ["<b>Composite index</b><div style='font-size:12px;color:#8A8289'>"
     "hand-weighted pillar score, isotonically calibrated and fitted on the full set</div>",
     f"<span class='n-num'>{bc.get('r2_log', 0):.3f}</span>",
     f"<span class='n-num'>{bc.get('spearman', 0):.2f}</span>",
     f"<span class='n-num'>{bc.get('ndcg@10_global', 0):.2f}</span>"],
    ["<b>Structural baseline</b><div style='font-size:12px;color:#8A8289'>"
     "ridge on the generator's five observable terms — arithmetic, not learning</div>",
     f"<span class='n-num'>{struct_r2:.3f}</span>" if struct_r2 is not None else "—",
     f"<span class='n-num'>{struct.get('spearman', 0):.2f}</span>", "—"],
    ["<b>LightGBM (this model)</b><div style='font-size:12px;color:#8A8289'>"
     "GroupKFold out-of-fold, early stopping on an inner split</div>",
     f"<span class='n-num' style='color:{GREEN}'>{r2:.3f}</span>",
     f"<span class='n-num' style='color:{GREEN}'>{perf.get('spearman', 0):.2f}</span>",
     f"<span class='n-num' style='color:{GREEN}'>{perf.get('ndcg@10_global', 0):.2f}</span>"],
]
st.markdown(ui.table(["Approach", "R² (log)", "Spearman", "NDCG@10 (global)"], rows,
                     aligns=["left", "right", "right", "right"]),
            unsafe_allow_html=True)
st.markdown(
    f"<div class='n-muted' style='margin-top:8px'>"
    f"The NDCG column is the <b>global</b> one and is shown only because the "
    f"baselines are quoted on it. It is decided by ten rows out of "
    f"{perf.get('n_rows', 0):,}, so it swings for reasons unrelated to model "
    f"quality. The number that answers the interface is NDCG within one brief: "
    f"<b>{perf.get('ndcg@10_within_brief', float('nan')):.3f}</b> at k=10 and "
    f"<b>{perf.get('ndcg@5_within_brief', float('nan')):.3f}</b> at k=5, averaged "
    f"over {perf.get('ndcg@10_n_briefs', 0)} briefs — because Discover ranks "
    f"creators inside a brief, not across the whole database.</div>",
    unsafe_allow_html=True)
st.markdown(
    f"<div class='n-muted' style='margin-top:10px;line-height:1.6'>"
    f"The composite baseline is <b>deliberately flattered</b> — it is calibrated and "
    f"fitted on the whole dataset, while the model is scored strictly out-of-fold. "
    f"The structural baseline is the one that matters: it says "
    f"<b>{perf.get('structural_share_of_r2', 0):.0%}</b> of this model's R² is "
    f"recoverable by arithmetic on columns the model was handed. That is a property "
    f"of any simulation — the generator has to compute the outcome from something — "
    f"and it is reported here rather than left for an examiner to find."
    f"</div>", unsafe_allow_html=True)

st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)

# ---- ablation -------------------------------------------------------------
abl = perf.get("ablation", {})
drops = abl.get("drops", {})
only = abl.get("only", {})
if drops:
    c1, c2 = st.columns(2, gap="medium")
    LABELS = {"reach": "Reach", "engagement": "Engagement", "content": "Content / NLP",
              "network": "Network", "brandfit": "Brand fit"}
    with c1:
        with st.container(border=True):
            st.markdown(ui.section("What each pillar is worth",
                                   "R² lost when the pillar is removed"),
                        unsafe_allow_html=True)
            items = sorted(drops.items(), key=lambda kv: -kv[1]["delta"])
            st.plotly_chart(
                charts.hbars([LABELS.get(k, k) for k, _ in items],
                             [v["delta"] for _, v in items],
                             colour=charts.SERIES["primary"], height=230, suffix=""),
                use_container_width=True, config=charts.CONFIG)
            st.markdown(
                f"<div style='font-size:12px;color:{INK_3}'>Every pillar contributes. "
                f"Content is the smallest at {drops.get('content', {}).get('delta', 0):.4f} "
                f"— worth keeping, and worth being honest about the size of.</div>",
                unsafe_allow_html=True)
    with c2:
        with st.container(border=True):
            st.markdown(ui.section("What each pillar can do alone",
                                   "R² from that pillar only"),
                        unsafe_allow_html=True)
            items = sorted(only.items(), key=lambda kv: -kv[1])
            st.plotly_chart(
                charts.hbars([LABELS.get(k, k) for k, _ in items],
                             [v for _, v in items],
                             colour=charts.SERIES["reach"], height=230, suffix=""),
                use_container_width=True, config=charts.CONFIG)
            st.markdown(
                f"<div style='font-size:12px;color:{INK_3}'>Alone-scores sum to far more "
                f"than the drop-scores: the pillars are correlated, so most of what any "
                f"one carries is recoverable from the others.</div>",
                unsafe_allow_html=True)

st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)

# ---- features -------------------------------------------------------------
f1, f2 = st.columns(2, gap="medium")
with f1:
    with st.container(border=True):
        st.markdown(ui.section("Feature importance", "LightGBM split gain"),
                    unsafe_allow_html=True)
        if imp is not None:
            top = imp.head(10)
            st.plotly_chart(
                charts.hbars(top.feature, top.gain_pct, colour=charts.SERIES["primary"],
                             height=290),
                use_container_width=True, config=charts.CONFIG)
with f2:
    with st.container(border=True):
        st.markdown(ui.section("SHAP", "Mean absolute contribution to a prediction"),
                    unsafe_allow_html=True)
        if shap is not None:
            col = "mean_abs_shap" if "mean_abs_shap" in shap.columns else shap.columns[1]
            top = shap.nlargest(10, col)
            st.plotly_chart(
                charts.hbars(top.feature, top[col] / top[col].max() * 100,
                             colour=charts.SERIES["engagement"], height=290),
                use_container_width=True, config=charts.CONFIG)

st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)

# ---- validation design ----------------------------------------------------
st.markdown("<div class='n-h2'>How the evaluation is protected</div>"
            "<div style='height:10px'></div>", unsafe_allow_html=True)
for title, body in [
    ("GroupKFold on creator, not random folds",
     "Each creator contributes up to three campaigns. A random split would put the "
     "same creator in train and test, and the model would score itself on people it "
     "had memorised. Folds are split by creator id, so a creator is never in both."),
    ("Latent traits are generated and never exported",
     "The universe is built from latent creator traits — content quality, "
     "authenticity, consistency, promo saturation, niche focus. They are written to "
     "disk and never joined into any feature table. They are the ground truth the "
     "model is not allowed to see."),
    ("Leakage is gated in code, not by inspection",
     "A banned-substring list plus an assertion runs before every fit. Any column "
     "carrying campaign outcome, latent trait or target-derived information fails "
     "the build rather than quietly inflating the score."),
    ("The price model's R² is not evidence, and is not quoted as such",
     f"Fees are predicted by their own LightGBM. Its R² in log space is "
     f"{price.get('r2_log', 0):.3f}, and that number should be ignored: the "
     f"generator prices a campaign from the published rate card and an engagement "
     f"premium, and both are model features, so evaluating that closed form with "
     f"its noise removed already explains "
     f"{price.get('closed_form_r2_log', float('nan')):.3f}. The model is recovering "
     f"an algebraic identity. What IS worth quoting is calibration: a MAPE of "
     f"{price.get('mape', 0):.0%} and an 80% prediction band that covers "
     f"{price.get('band_coverage_p10_p90', 0):.0%} of held-out fees. Rate cards on "
     f"the creator side and budget caps on the brand side read from it."),
    ("Predictions are back-transformed with a smearing correction",
     f"The model is trained on log engagement rate, and exp() of a log-space "
     f"prediction estimates the median rather than the mean — which biased every "
     f"forecast low by about 10%. Duan's (1983) smearing estimator, measured "
     f"out-of-fold at ×{perf.get('smearing_factor', 1.0):.4f}, is applied at both "
     f"evaluation and serving time so the Reporting page and this page cannot "
     f"disagree."),
]:
    with st.container(border=True):
        st.markdown(
            f"<div class='n-h3'>{ui.esc(title)}</div>"
            f"<div style='font-size:13.5px;color:{INK_2};line-height:1.6;margin-top:5px'>"
            f"{ui.esc(body)}</div>", unsafe_allow_html=True)


# ==========================================================================
# Are the brand-fit weights load-bearing?
# ==========================================================================
sens = data.load("nectar_weight_sensitivity.parquet")
if sens is not None and len(sens):
    st.markdown("<div style='height:22px'></div>"
                "<div class='n-h2'>Do the brand-fit weights matter?</div>"
                "<div class='n-muted' style='margin:2px 0 12px 0'>"
                "The five composite weights are asserted, not learned. This is the "
                "test of whether that is defensible.</div>",
                unsafe_allow_html=True)

    worst = sens.loc[sens.mean_overlap.idxmin()]
    best = sens.loc[sens.mean_overlap.idxmax()]
    stable = float(sens.mean_overlap.mean())

    with st.container(border=True):
        st.markdown(ui.section(
            "Top-20 shortlist survival",
            "Each weight scaled to 50%, 75%, 125% and 150% of its value, "
            "renormalised, shortlist rebuilt"), unsafe_allow_html=True)
        agg = (sens.groupby("component", as_index=False)
               .agg(mean_overlap=("mean_overlap", "mean"))
               .sort_values("mean_overlap"))
        LABEL = {"semantic_similarity": "Semantic similarity",
                 "category_match": "Category match", "audience_match": "Audience match",
                 "content_safety": "Content safety", "consistency": "Consistency"}
        st.plotly_chart(
            charts.hbars([LABEL.get(c, c) for c in agg.component],
                         agg.mean_overlap * 100,
                         colour=charts.SERIES["primary"], height=230),
            use_container_width=True, config=charts.CONFIG)
        st.markdown(
            f"<div style='font-size:12.5px;color:{INK_2};line-height:1.6'>"
            f"Across every perturbation the shortlist keeps "
            f"<b>{stable:.0%}</b> of its members on average. The most sensitive "
            f"weight is <b>{LABEL.get(worst.component, worst.component)}</b> — at "
            f"{worst.multiplier:.0%} of its value the top 20 still shares "
            f"<b>{worst.mean_overlap:.0%}</b> of its names, and never falls below "
            f"{worst.min_overlap:.0%} on any single campaign. "
            + ("The ranking is therefore driven by the components' agreement rather "
               "than by the exact weights, which is what makes an unvalidated "
               "weighting defensible here."
               if stable >= 0.75 else
               "That is low enough that the weights ARE doing the ranking, and the "
               "choice of 34/28/18/12/8 would need justifying before anyone relied "
               "on this shortlist.")
            + "</div>", unsafe_allow_html=True)
