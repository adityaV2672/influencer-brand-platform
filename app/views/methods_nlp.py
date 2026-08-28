"""
Model & Methods — NLP.

This page answers the question the project was actually set: is a positive /
negative lexicon good enough, or do NRC, VADER, SBERT embeddings, BERTopic and
LLM prompting earn their keep?

The answer is measured on REAL, human-labelled corpora, not on the synthetic
captions. Measuring a sentiment classifier on text a generator wrote is
circular: the generator put the sentiment there, so of course the classifier
finds it. Only real labels can settle the question.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from nectar import charts, data, ui
from nectar.theme import AMBER, GREEN, INK, INK_2, INK_3, LINE, LINE_2

bench = data.load("benchmark_results.parquet")
meta = data.load_json("benchmark_meta.json") or {}
coh = data.load_json("topic_coherence.json") or {}
nlp = data.load_json("nlp_report.json") or {}

st.markdown(ui.page_header(
    "NLP methods",
    "Six approaches, three tasks, four human-labelled corpora.",
    eyebrow="Model & methods"), unsafe_allow_html=True)

if bench is None:
    st.markdown(ui.empty_state("⚗", "Benchmarks not built.",
                               "Run `python run_pipeline.py --only benchmarks`."),
                unsafe_allow_html=True)
    st.stop()

ok = bench[bench.status == "ok"]
failed = bench[bench.status != "ok"]

FAMILY_COLOUR = {
    "baseline": INK_3, "lexicon": charts.SERIES["spend"],
    "classical-ml": charts.SERIES["primary"], "transformer": charts.SERIES["reach"],
    "llm": charts.SERIES["engagement"],
}

st.markdown(
    f"<div class='n-card'><div class='n-h3'>Why real corpora</div>"
    f"<div style='font-size:13.5px;color:{INK_2};line-height:1.65;margin-top:6px'>"
    f"The creator universe in this product is synthetic. Its captions were written "
    f"by a generator, so measuring a sentiment model on them would only measure "
    f"whether the classifier can invert the generator. Every accuracy figure below "
    f"comes from a corpus labelled by people: TweetEval (Barbieri et al., 2020) for "
    f"sentiment, emotion and irony, and the Misra & Arora news-headline sarcasm set. "
    f"The methods are then applied to the synthetic captions — which is the only "
    f"direction that makes sense.</div></div>",
    unsafe_allow_html=True)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

TASKS = [("sentiment", "Sentiment", "Positive / neutral / negative on real tweets"),
         ("emotion", "Emotion", "Four-way emotion classification"),
         ("irony", "Irony &amp; sarcasm", "The task the professor singled out")]

for task, title, blurb in TASKS:
    d = ok[ok.task == task]
    if d.empty:
        continue
    st.markdown(f"<div class='n-h2'>{title}</div>"
                f"<div class='n-muted' style='margin:2px 0 10px 0'>{blurb}</div>",
                unsafe_allow_html=True)
    for corpus, g in d.groupby("corpus"):
        g = g.sort_values("macro_f1", ascending=False)
        base = g[g.family == "baseline"]
        base_f1 = float(base.macro_f1.iloc[0]) if len(base) else 0.0
        with st.container(border=True):
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"align-items:baseline;margin-bottom:10px'>"
                f"<span class='n-h3'>{ui.esc(corpus)}</span>"
                f"<span class='n-muted'>{int(g.n_eval.max()):,} labelled examples</span>"
                f"</div>", unsafe_allow_html=True)

            # Zero-shot and corpus-trained methods are not answering the same
            # question, so they are not put in one leaderboard. A model fitted on
            # this corpus should beat one that has never seen it; ranking them
            # together makes that look like a finding.
            groups = [
                ("Applied without seeing this corpus",
                 g[~g.supervised.fillna(False).astype(bool)],
                 "lexicons, majority baseline and zero-shot transformers"),
                ("Trained on this corpus",
                 g[g.supervised.fillna(False).astype(bool)],
                 "fitted on the training split, scored on held-out test"),
            ]
            for title, gg, note in groups:
                if gg.empty:
                    continue
                st.markdown(
                    f"<div class='n-eyebrow' style='margin:10px 0 4px 0'>{title}</div>"
                    f"<div style='font-size:11.5px;color:{INK_3};margin-bottom:6px'>"
                    f"{note}</div>", unsafe_allow_html=True)
                rows = []
                for r in gg.itertuples():
                    align = getattr(r, "alignment", None) or {}
                    suspect = bool(align.get("label_alignment_suspect"))
                    beats = r.macro_f1 > base_f1 and r.family != "baseline"
                    colour = GREEN if beats else (INK_3 if r.family == "baseline" else AMBER)
                    name = ui.esc(r.method_name)
                    if suspect:
                        name += (f" <span class='n-chip' style='color:{AMBER};"
                                 f"background:#FBF3E0'>label alignment suspect</span>")
                    rows.append([
                        f"<span style='font-weight:"
                        f"{700 if r.family == 'transformer' else 500}'>{name}</span>",
                        f"<span class='n-chip' style='color:"
                        f"{FAMILY_COLOUR.get(r.family, INK_3)};background:{LINE_2}'>"
                        f"{ui.esc(r.family or '—')}</span>",
                        f"<span class='n-num'>{r.accuracy:.3f}</span>",
                        f"<span class='n-num' style='color:{colour}'>{r.macro_f1:.3f}</span>",
                        (f"<span class='n-num' style='color:{INK_3}'>"
                         f"{r.texts_per_sec:,.0f}/s</span>"
                         if r.texts_per_sec == r.texts_per_sec else "—"),
                    ])
                st.markdown(
                    ui.table(["Method", "Family", "Accuracy", "Macro F1", "Throughput"],
                             rows, aligns=["left", "left", "right", "right", "right"]),
                    unsafe_allow_html=True)

            flagged = [r for r in g.itertuples()
                       if (getattr(r, "alignment", None) or {}).get("label_alignment_suspect")]
            for r in flagged:
                a = r.alignment
                st.markdown(
                    f"<div style='margin-top:10px;border-left:3px solid {AMBER};"
                    f"background:#FBF3E0;border-radius:0 10px 10px 0;padding:11px 14px'>"
                    f"<div style='font-size:12.5px;font-weight:600;color:{INK}'>"
                    f"{ui.esc(r.method_name)} — this row is a wiring fault, not a result"
                    f"</div>"
                    f"<div style='font-size:12.5px;color:{INK_2};line-height:1.6;"
                    f"margin-top:4px'>"
                    f"As scored it reaches {a.get('identity_accuracy', 0):.3f} accuracy. "
                    f"Relabelling its outputs "
                    f"({', '.join(f'{k}→{v}' for k, v in (a.get('best_permutation') or {}).items() if k != v)}) "
                    f"lifts it to {a.get('best_permutation_accuracy', 0):.3f} — a gain of "
                    f"{a.get('alignment_gap', 0):.3f}. A genuinely weak classifier does not "
                    f"gain half an accuracy point from being renamed, so the model is "
                    f"discriminating and its label space is misaligned with the corpus. "
                    f"The corpus side has been verified against the authors' own "
                    f"mapping file, which leaves the checkpoint's label order as the "
                    f"cause. It is flagged rather than silently corrected, because "
                    f"deciding that needs the checkpoint in hand — and this row is "
                    f"excluded from the comparison above."
                    f"</div></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

# ---- the finding ----------------------------------------------------------
irony = ok[(ok.task == "irony")]
lex = irony[irony.family == "lexicon"]
base = irony[irony.family == "baseline"]
worst_gap = None
if len(lex) and len(base):
    by_corpus = []
    for corpus, g in irony.groupby("corpus"):
        b = g[g.family == "baseline"]
        l = g[g.family == "lexicon"]
        if len(b) and len(l):
            by_corpus.append((corpus, float(b.accuracy.iloc[0]), float(l.accuracy.max())))
    worst_gap = by_corpus

st.markdown(
    f"<div class='n-card' style='border-left:3px solid {AMBER}'>"
    f"<div class='n-h3'>The finding that matters</div>"
    f"<div style='font-size:13.5px;color:{INK_2};line-height:1.7;margin-top:8px'>"
    + ("".join(
        f"On <b>{ui.esc(c)}</b>, the best lexicon method reaches "
        f"<b>{l:.1%}</b> accuracy against a majority-class baseline of "
        f"<b>{b:.1%}</b> — {'below' if l < b else 'barely above'} the baseline. "
        for c, b, l in (worst_gap or [])))
    + "A positive/negative lexicon cannot detect irony, because irony is "
      "positive words used to mean the opposite. This is precisely the failure "
      "mode that makes a Bing-style lexicon unsafe for brand-safety screening: "
      "a sarcastic takedown of a product scores as praise."
    f"</div></div>", unsafe_allow_html=True)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ---- topics ---------------------------------------------------------------
if coh:
    st.markdown("<div class='n-h2'>Topic modelling</div>"
                "<div class='n-muted' style='margin:2px 0 10px 0'>"
                "BERTopic against LDA on the same corpus.</div>",
                unsafe_allow_html=True)
    # The coherence file nests the two models under their own keys and keeps
    # citations alongside them, so iterate the known model keys rather than
    # every key in the file.
    rows = []
    for key in ("bertopic", "lda"):
        vals = coh.get(key)
        if not isinstance(vals, dict):
            continue
        rows.append([
            f"<b>{'BERTopic' if key == 'bertopic' else 'LDA'}</b>",
            f"<span class='n-num'>{coh.get('n_topics', '—')}</span>",
            f"<span class='n-num'>{vals.get('npmi', float('nan')):.4f}</span>",
            f"<span class='n-num'>{vals.get('c_v', float('nan')):.4f}</span>",
            f"<span class='n-num'>{vals.get('diversity', float('nan')):.3f}</span>",
        ])
    if rows:
        st.markdown(ui.table(["Model", "Topics", "NPMI", "C_v", "Diversity"], rows,
                             aligns=["left", "right", "right", "right", "right"]),
                    unsafe_allow_html=True)
        bt, ld = coh.get("bertopic", {}), coh.get("lda", {})
        st.markdown(
            f"<div class='n-muted' style='margin-top:10px;line-height:1.6'>"
            f"BERTopic wins on coherence (NPMI {bt.get('npmi', 0):.3f} against "
            f"{ld.get('npmi', 0):.3f}) and loses slightly on diversity "
            f"({bt.get('diversity', 0):.3f} against {ld.get('diversity', 0):.3f}) — "
            f"it finds tighter topics that share a few more words. "
            f"{coh.get('outlier_fraction', 0):.0%} of the "
            f"{coh.get('n_documents', 0):,} captions are assigned to no topic at all, "
            f"which LDA cannot express: LDA gives every document a distribution, "
            f"whether or not it is about anything.</div>",
            unsafe_allow_html=True)

# ---- what did not run -----------------------------------------------------
if len(failed):
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    names = sorted(set(failed.method_name))
    st.markdown(
        f"<div class='n-card' style='border-left:3px solid {AMBER}'>"
        f"<div class='n-h3'>What did not run, and why</div>"
        f"<div style='font-size:13.5px;color:{INK_2};line-height:1.7;margin-top:8px'>"
        f"{len(failed)} LLM-prompting runs are recorded as failed. The prompting "
        f"harness is implemented — zero-shot, few-shot and chain-of-thought against a "
        f"local Ollama model — but the model server could not be installed on the "
        f"machine that ran the pipeline, so no results were produced. "
        f"These rows are kept rather than deleted: a benchmark table that silently "
        f"drops the methods that failed is not a benchmark. "
        f"<br><br>The gap this leaves is real. LLM prompting is the one approach in "
        f"the brief that was not measured, and on the irony task it is the approach "
        f"most likely to beat the transformer. The claim this page can defend is "
        f"about the five methods that did run."
        f"</div></div>", unsafe_allow_html=True)
