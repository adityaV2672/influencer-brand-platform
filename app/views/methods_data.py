"""
Model & Methods — Data.

Every table the platform uses, where each number came from, and a download for
each one as CSV. The provenance column is the point: this project mixes a
generated creator universe, measured NLP output, model predictions and a
simulated transaction layer, and a reader who cannot tell which is which cannot
judge any of it.
"""
from __future__ import annotations

import io
import json
import zipfile

import pandas as pd
import streamlit as st

from nectar import data, ui
from nectar.theme import AMBER, GREEN, INK, INK_2, INK_3, LINE, LINE_2

st.markdown(ui.page_header(
    "Data", "Every table, its provenance, and a CSV of it.",
    eyebrow="Model & methods"), unsafe_allow_html=True)

TABLES = [
    ("Creator universe", [
        ("nectar_creators.parquet", "creators",
         "One row per creator: engineered features plus the presentation fields "
         "the product shows.", "generated + model output"),
        ("influencers.parquet", "influencer_features",
         "The engineered feature table before the product layer.", "generated + measured"),
        ("posts_sample.parquet", "posts_sample",
         "Captions with their per-post NLP labels.", "generated + measured"),
        ("brands.parquet", "brands",
         "The synthetic brand universe.", "generated"),
    ]),
    ("Matching and campaigns", [
        ("nectar_campaigns.parquet", "campaigns",
         "The six showcase campaigns and their eligibility policy.", "simulated"),
        ("nectar_fit.parquet", "campaign_fit",
         "Every (campaign, creator) pair with fit components and safety gates.",
         "model output"),
        ("brand_fit.parquet", "brand_fit_sbert",
         "The pipeline's SBERT-scored matrix, top 60 creators per brand.",
         "model output"),
        ("nectar_category_fit.parquet", "category_fit",
         "Creator fit against a representative brand in each category.", "model output"),
    ]),
    ("Transactions", [
        ("nectar_requests.parquet", "requests",
         "Who was approached, at what fee, and how far they got.", "simulated"),
        ("nectar_messages.parquet", "messages", "Negotiation threads.", "simulated"),
        ("nectar_creator_history.parquet", "creator_history",
         "Past brand approaches per creator.", "simulated"),
        ("nectar_earnings.parquet", "earnings",
         "Monthly realised earnings.", "simulated"),
    ]),
    ("Results and evidence", [
        ("nectar_campaign_summary.parquet", "campaign_summary",
         "Campaign results including predicted-vs-actual.", "measured + model output"),
        ("nectar_calibration.parquet", "model_calibration",
         "Out-of-fold predicted vs actual, by brand category.", "measured"),
        ("benchmark_results.parquet", "nlp_benchmarks",
         "NLP accuracy on real, human-labelled corpora.", "measured"),
        ("feature_importance.parquet", "feature_importance",
         "LightGBM split gain per feature.", "model output"),
        ("feature_shap.parquet", "feature_shap",
         "Mean absolute SHAP per feature.", "model output"),
        ("edges.parquet", "graph_edges",
         "Strongest co-behaviour links between creators.", "measured"),
        ("topics.parquet", "topics", "BERTopic topics and their top words.", "measured"),
    ]),
]

PROV_COLOUR = {
    "generated": ("#2E6FB7", "#EAF1F9"),
    "measured": (GREEN, "#E8F4F0"),
    "model output": ("#7C4DA0", "#F1EAF7"),
    "simulated": (AMBER, "#FBF3E0"),
}


def prov_chip(label: str) -> str:
    parts = [p.strip() for p in label.split("+")]
    out = []
    for p in parts:
        fg, bg = PROV_COLOUR.get(p, (INK_3, LINE_2))
        out.append(f"<span class='n-chip' style='color:{fg};background:{bg}'>{p}</span>")
    return " ".join(out)


def to_csv(df: pd.DataFrame) -> bytes:
    """List and dict columns are JSON-encoded so the CSV round-trips."""
    out = df.copy()
    for col in out.columns:
        s = out[col].dropna()
        if s.empty:
            continue
        first = s.iloc[0]
        if isinstance(first, (list, dict, tuple)) or hasattr(first, "tolist"):
            out[col] = out[col].map(
                lambda v: json.dumps(v.tolist() if hasattr(v, "tolist") else v,
                                     default=str) if v is not None else "")
    return out.to_csv(index=False).encode("utf-8")


# ---- legend ---------------------------------------------------------------
st.markdown(
    f"<div class='n-card'><div class='n-h3'>What the labels mean</div>"
    f"<div style='font-size:13.5px;color:{INK_2};line-height:1.85;margin-top:8px'>"
    f"{prov_chip('generated')} &nbsp;drawn by the synthetic universe generator from "
    f"latent creator traits, calibrated so engagement rates and fees reproduce "
    f"published 2026 benchmarks by follower tier.<br>"
    f"{prov_chip('measured')} &nbsp;computed from data — NLP over captions, centrality "
    f"over the co-behaviour graph, accuracy against human labels.<br>"
    f"{prov_chip('model output')} &nbsp;predicted by a trained model: LightGBM for "
    f"performance and price, the brand-fit composite for matching.<br>"
    f"{prov_chip('simulated')} &nbsp;the transaction layer — funnel stages, "
    f"negotiations, payment status. There is no counterpart in the modelling data, "
    f"so it is invented, with a fixed seed, and labelled as such."
    f"</div></div>", unsafe_allow_html=True)

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

# ---- download everything --------------------------------------------------
loaded = {}
total_rows = 0
for _, group in TABLES:
    for fname, key, _, _ in group:
        df = data.load(fname)
        if df is not None:
            loaded[key] = df
            total_rows += len(df)

a, b, c = st.columns([1.1, 1.1, 2])
with a:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for key, df in loaded.items():
            z.writestr(f"{key}.csv", to_csv(df))
        z.writestr("MANIFEST.csv", pd.DataFrame([
            {"file": f"{k}.csv", "rows": len(v), "columns": v.shape[1]}
            for k, v in loaded.items()]).to_csv(index=False))
    st.download_button("⤓  Download all as CSV (ZIP)", buf.getvalue(),
                       file_name="nectar_data.zip", mime="application/zip",
                       type="primary", use_container_width=True)
with b:
    dd = data.load("data_dictionary.parquet")
    if dd is not None:
        st.download_button("⤓  Data dictionary", to_csv(dd),
                           file_name="DATA_DICTIONARY.csv", mime="text/csv",
                           use_container_width=True)
with c:
    st.markdown(
        f"<div style='padding-top:9px;font-size:12.5px;color:{INK_3}'>"
        f"{len(loaded)} tables · {total_rows:,} rows in total. "
        f"Run <code>python -m src.features.export_csv</code> to write the same "
        f"files to <code>data/csv/</code> on disk.</div>",
        unsafe_allow_html=True)

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

# ---- per-table ------------------------------------------------------------
for section, group in TABLES:
    st.markdown(f"<div class='n-h2'>{section}</div><div style='height:8px'></div>",
                unsafe_allow_html=True)
    for fname, key, blurb, prov in group:
        df = loaded.get(key)
        if df is None:
            continue
        with st.container(border=True):
            head, dl = st.columns([3.4, 1])
            with head:
                st.markdown(
                    f"<div style='display:flex;align-items:baseline;gap:10px;flex-wrap:wrap'>"
                    f"<span class='n-h3'>{key}.csv</span>"
                    f"<span class='n-num' style='font-size:12px;color:{INK_3}'>"
                    f"{len(df):,} rows × {df.shape[1]} cols</span>"
                    f"{prov_chip(prov)}</div>"
                    f"<div style='font-size:13px;color:{INK_2};margin-top:5px'>"
                    f"{ui.esc(blurb)}</div>", unsafe_allow_html=True)
            with dl:
                st.download_button("⤓  CSV", to_csv(df), file_name=f"{key}.csv",
                                   mime="text/csv", use_container_width=True,
                                   key=f"dl_{key}")
            with st.expander(f"Preview and columns"):
                st.dataframe(df.head(8), width="stretch", hide_index=True)
                st.markdown(
                    f"<div style='font-size:12px;color:{INK_3};margin-top:6px'>"
                    f"<b>Columns:</b> " + ", ".join(f"<code>{c}</code>" for c in df.columns)
                    + "</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

# ---- what is not here -----------------------------------------------------
st.markdown(
    f"<div class='n-card' style='border-left:3px solid {AMBER}'>"
    f"<div class='n-h3'>What is not in this list</div>"
    f"<div style='font-size:13.5px;color:{INK_2};line-height:1.7;margin-top:6px'>"
    f"Two things are deliberately absent. <b>The latent traits</b> — content quality, "
    f"authenticity, consistency, promo saturation, niche focus — are what the "
    f"generator used to build the universe. They are written to disk during "
    f"generation and never joined into any feature table, because they are the "
    f"ground truth the model is not allowed to see. <b>The full post corpus</b> "
    f"(52,089 captions) and the raw generated profiles live in <code>data/processed/</code>, "
    f"which is rebuildable with <code>python run_pipeline.py</code> and therefore not "
    f"committed; the sample above carries the same columns."
    f"</div></div>", unsafe_allow_html=True)
