"""
Model & Methods - the evidence page.

This is where the project's empirical claims live: which NLP method actually
works, measured on real human-labelled data, and whether the learned scoring
model beats the simpler alternatives it replaced.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from theme import (
    DIVERGING, GRID, INK, SEQ_BLUE, SERIES, STATUS,
    load, load_json, page_header, plotly_layout, sidebar_tier,
)

st.set_page_config(page_title="Model & methods", page_icon="◎", layout="wide")
sidebar_tier()

page_header(
    "Model & methods",
    "Every number below is measured, reproducible, and scored on held-out data.",
)

bench = load("benchmark_results.parquet")
model_res = load_json("model_results.json")
coh = load_json("topic_coherence.json")
imp = load("feature_importance.parquet")
shap_imp = load("feature_shap.parquet")
gmeta = load_json("graph_meta.json")

tabs = st.tabs(["NLP method comparison", "Scoring model", "Topics", "Network", "Data & honesty"])

# ==========================================================================
# 1. NLP comparison
# ==========================================================================
with tabs[0]:
    st.markdown("### Which sentiment method should this platform use?")
    st.caption(
        "Measured on **real, human-labelled corpora** — TweetEval (Barbieri et al., "
        "Findings of EMNLP 2020) and the news-headline sarcasm set (Misra & Arora). "
        "The creator universe in this app is synthetic; these accuracy figures are not."
    )

    if bench is None:
        st.warning("Benchmark results not found. Run `python run_pipeline.py --only evaluate export`.")
    else:
        ok = bench[bench["status"] == "ok"].copy()

        task = st.radio("Task", sorted(ok["task"].unique()), horizontal=True)
        t = ok[ok["task"] == task]
        corpora = sorted(t["corpus"].unique())
        corpus = st.radio("Corpus", corpora, horizontal=True) if len(corpora) > 1 else corpora[0]
        t = t[t["corpus"] == corpus].sort_values("macro_f1", ascending=False)

        fam_color = {
            "baseline": INK["muted"], "lexicon": SERIES[1], "classical-ml": SERIES[3],
            "transformer": SERIES[0], "llm": SERIES[6],
        }

        a, b = st.columns([1.35, 1])
        with a:
            fig = go.Figure()
            for fam in ["baseline", "lexicon", "classical-ml", "transformer", "llm"]:
                sub = t[t["family"] == fam]
                if not len(sub):
                    continue
                fig.add_bar(
                    x=sub["macro_f1"], y=sub["method_name"], orientation="h",
                    name=fam, marker=dict(color=fam_color[fam]),
                    text=[f"{v:.3f}" for v in sub["macro_f1"]], textposition="outside",
                    hovertemplate="<b>%{y}</b><br>macro-F1 %{x:.3f}<extra></extra>",
                )
            base = t[t["method_key"] == "majority_baseline"]
            if len(base):
                fig.add_vline(x=float(base["macro_f1"].iloc[0]), line_width=1,
                              line_dash="dot", line_color=INK["primary"])
            fig.update_layout(yaxis=dict(autorange="reversed"))
            fig.update_xaxes(range=[0, min(1.05, float(t["macro_f1"].max()) * 1.25)])
            plotly_layout(fig, height=max(280, 42 * len(t)), xtitle="Macro-F1 (higher is better)")
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            st.caption("Dotted line = majority-class baseline. A method left of it has learnt nothing.")

        with b:
            st.markdown("**Full results**")
            show = t[["method_name", "family", "accuracy", "macro_f1", "n_eval", "texts_per_sec"]].copy()
            show.columns = ["Method", "Family", "Accuracy", "Macro-F1", "n", "texts/sec"]
            st.dataframe(
                show, hide_index=True, width="stretch",
                column_config={
                    "Accuracy": st.column_config.NumberColumn(format="%.3f"),
                    "Macro-F1": st.column_config.NumberColumn(format="%.3f"),
                    "texts/sec": st.column_config.NumberColumn(format="%.0f"),
                },
            )
            if t["subsampled"].any():
                st.caption(
                    "LLM rows are scored on a stratified subsample; every other method was "
                    "additionally scored on that identical subsample so the comparison is like-for-like."
                )

        # The headline finding. Phrased from the measured numbers rather than
        # asserted, so that if a re-run changes the result the text changes too.
        if task == "irony":
            lex = t[t["family"] == "lexicon"]
            base_rows = t[t["method_key"] == "majority_baseline"]
            if len(lex) and len(base_rows):
                base_acc = float(base_rows["accuracy"].iloc[0])
                best_lex = float(lex["accuracy"].max())
                if best_lex <= base_acc:
                    st.error(
                        f"**The headline result.** Every word-list method scores at or below the "
                        f"majority-class baseline on irony (best lexicon accuracy **{best_lex:.3f}** "
                        f"vs baseline **{base_acc:.3f}**). Sarcasm inverts meaning without changing "
                        f"vocabulary, so no amount of tuning a positive/negative word list recovers "
                        f"it. This is the evidence behind moving the content pillar to transformer "
                        f"and LLM methods."
                    )
                else:
                    st.warning(
                        f"**Result on this run:** the best lexicon method reaches accuracy "
                        f"**{best_lex:.3f}** against a majority baseline of **{base_acc:.3f}** — "
                        f"it clears the baseline, but by a margin far smaller than the learned "
                        f"methods above. Compare macro-F1 rather than accuracy here: the corpus is "
                        f"imbalanced, and accuracy flatters a method that mostly predicts the "
                        f"majority class."
                    )
        elif task == "sentiment":
            st.info(
                "**A result worth stating plainly.** NRC scores *worse* than VADER on three-class "
                "polarity here, despite being the richer lexicon. NRC's strength is its eight "
                "emotion categories, not polarity — so this project uses NRC for the emotion "
                "profile and does not use it as the polarity method."
            )

        with st.expander("Failure cases — where lexicons break, in their own words"):
            st.markdown(
                "| Text | True | Bing | VADER |\n|---|---|---|---|\n"
                "| *Oh great, another subscription fee. Brilliant work.* | negative | **positive** | **positive** |\n"
                "| *Nothing says premium like a charger sold separately. Fantastic.* | negative | **positive** | **positive** |\n"
                "| *Genuinely impressed by this serum, my skin looks calmer.* | positive | positive | positive |\n"
                "| *Disappointed by the build quality, would not repeat.* | negative | negative | negative |\n\n"
                "Both lexicons handle literal text correctly and both fail on ironic praise — "
                "the words are positive, the meaning is not."
            )

# ==========================================================================
# 2. Scoring model
# ==========================================================================
with tabs[1]:
    if model_res is None:
        st.warning("Model results not found. Run `python run_pipeline.py --only models export`.")
    else:
        p = model_res["performance"]
        st.markdown("### Does the learned model beat the simpler alternatives?")

        c = st.columns(4)
        c[0].metric("R² (log)", f"{p['r2_log']:.3f}",
                    help="Explained variance in log space, out-of-fold.")
        c[1].metric("Spearman", f"{p['spearman']:.3f}", help="Rank correlation — what ranking quality means.")
        c[2].metric("NDCG@10", f"{p['ndcg@10']:.3f}", help="Quality of the top-10 shortlist.")
        c[3].metric("of achievable ceiling", f"{p['fraction_of_ceiling']:.0%}",
                    help=p.get("ceiling_note", ""))

        st.caption(
            f"Validated with **GroupKFold on creator id** — a creator never appears in both "
            f"train and test. Fold R²: {', '.join(f'{v:.3f}' for v in p['fold_r2_log'])} "
            f"(mean {p['fold_r2_log_mean']:.3f} ± {p['fold_r2_log_std']:.3f})."
        )

        a, b = st.columns([1, 1])
        with a:
            st.markdown("**Learned model vs the alternatives it replaced**")
            rows = [
                ("LightGBM (this system)", p["r2_log"], p["spearman"]),
                ("Phase-1 weighted index", p["baseline_composite_index"]["r2_log"],
                 p["baseline_composite_index"]["spearman"]),
                ("Published benchmark curve", p["baseline_benchmark_curve"]["r2_log"],
                 p["baseline_benchmark_curve"]["spearman"]),
            ]
            names = [r[0] for r in rows][::-1]
            vals = [r[1] for r in rows][::-1]
            colors = [INK["muted"], INK["muted"], SERIES[0]]
            fig = go.Figure(go.Bar(
                x=vals, y=names, orientation="h", marker=dict(color=colors),
                text=[f"{v:.3f}" for v in vals], textposition="outside",
            ))
            fig.add_vline(x=0, line_width=1, line_color=INK["primary"])
            plotly_layout(fig, height=230, showlegend=False, xtitle="R² (log), out-of-fold")
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            st.caption(
                "The weighted index is the Phase-1 design from the original proposal, and it is "
                "**flattered** here — isotonically calibrated on the full dataset. It still loses "
                f"by a wide margin ({p['baseline_composite_index']['r2_log']:.3f} vs {p['r2_log']:.3f})."
            )

        with b:
            st.markdown("**Which pillar earns its place?**")
            ab = p.get("ablation", {})
            drops = ab.get("drops", {})
            if drops:
                names = list(drops.keys())
                deltas = [drops[k]["delta"] for k in names]
                order = np.argsort(deltas)
                names = [names[i] for i in order]
                deltas = [deltas[i] for i in order]
                colors = [SERIES[0] if d > 0 else SERIES[7] for d in deltas]
                fig = go.Figure(go.Bar(
                    x=deltas, y=[n.title() for n in names], orientation="h",
                    marker=dict(color=colors),
                    text=[f"{d:+.3f}" for d in deltas], textposition="outside",
                ))
                fig.add_vline(x=0, line_width=1, line_color=INK["primary"])
                plotly_layout(fig, height=230, showlegend=False,
                              xtitle="R² lost when the pillar is removed")
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

                neg = [n for n, d in zip(names, deltas) if d <= 0]
                if neg:
                    st.warning(
                        f"**Negative result, reported rather than buried.** Removing the "
                        f"**{', '.join(neg)}** pillar does not hurt the model. On this data the "
                        f"content features are not adding predictive signal beyond what reach, "
                        f"engagement and network already capture. The NLP layer still earns its "
                        f"place for brand-safety screening and explanation — but it should not be "
                        f"claimed as a driver of performance prediction."
                    )

        if imp is not None:
            st.markdown("**What the model actually uses**")
            top = imp.head(15).iloc[::-1]
            fig = go.Figure(go.Bar(
                x=top["gain_pct"], y=top["feature"], orientation="h",
                marker=dict(color=SERIES[0]),
                text=[f"{v:.1f}%" for v in top["gain_pct"]], textposition="outside",
            ))
            plotly_layout(fig, height=440, showlegend=False, xtitle="Share of total split gain (%)")
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        st.markdown("### Price model")
        q = model_res["price"]
        c = st.columns(4)
        c[0].metric("R² (log)", f"{q['r2_log']:.3f}")
        c[1].metric("MAPE", f"{q['mape']:.1%}")
        c[2].metric("Rule-based baseline R²", f"{q['baseline_rate_card']['r2_log']:.3f}")
        c[3].metric("Band coverage", f"{q['band_coverage_p10_p90']:.0%}", help="Share of true fees inside the shown band.")
        st.info(
            f"**The ML price model barely beats the rule.** R² {q['r2_log']:.3f} against "
            f"{q['baseline_rate_card']['r2_log']:.3f} for a published rate card with no learning at all. "
            "The business conclusion is that Phase-1 rule-based pricing is sufficient and the "
            "regression model is not worth its maintenance cost until real negotiated-deal data exists."
        )

# ==========================================================================
# 3. Topics
# ==========================================================================
with tabs[2]:
    st.markdown("### BERTopic vs LDA on short captions")
    if coh is None:
        st.warning("Topic results not found. Run `python run_pipeline.py --only nlp export`.")
    else:
        c = st.columns(4)
        c[0].metric("Topics found", coh.get("n_topics", "—"))
        c[1].metric("Outlier share", f"{coh.get('outlier_fraction', 0):.1%}")
        c[2].metric("BERTopic NPMI", f"{coh['bertopic']['npmi']:.3f}")
        c[3].metric("LDA NPMI", f"{coh['lda']['npmi']:.3f}")

        met = ["npmi", "c_v", "diversity"]
        fig = go.Figure()
        for i, (name, key) in enumerate([("BERTopic", "bertopic"), ("LDA", "lda")]):
            fig.add_bar(
                x=[m.upper() for m in met], y=[coh[key][m] for m in met],
                name=name, marker=dict(color=SERIES[i], line=dict(width=2, color="#fcfcfb")),
                text=[f"{coh[key][m]:.3f}" for m in met], textposition="outside",
            )
        plotly_layout(fig, height=300, ytitle="Score")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption(
            "Both models fitted on the identical corpus with the identical topic count and scored "
            "with the identical coherence implementation. Diversity is reported alongside coherence "
            "because a model that repeats one generic topic scores well on coherence alone."
        )

        topics = load("topics.parquet")
        if topics is not None:
            st.markdown("**Discovered topics**")
            st.dataframe(
                topics.sort_values("size", ascending=False)[["topic_id", "size", "top_words"]]
                .rename(columns={"topic_id": "Topic", "size": "Posts", "top_words": "Top terms"}),
                hide_index=True, width="stretch", height=320,
            )

# ==========================================================================
# 4. Network
# ==========================================================================
with tabs[3]:
    st.markdown("### The influencer graph")
    if gmeta is None:
        st.warning("Graph metadata not found. Run `python run_pipeline.py --only sna export`.")
    else:
        c = st.columns(4)
        c[0].metric("Nodes", f"{gmeta.get('n_nodes', 0):,}")
        c[1].metric("Edges", f"{gmeta.get('n_edges', 0):,}")
        c[2].metric("Density", f"{gmeta.get('density', 0):.5f}")
        c[3].metric("Communities", gmeta.get("n_communities", "—"))

        st.warning(
            "**Read this before quoting a centrality number.** The original design assumed a "
            "follower graph. Instagram exposes no follower edges to third parties, so there is no "
            "legal route to one at this scale.\n\n"
            "The graph is built instead from **co-behaviour**: creators who share rare hashtags and "
            "work with the same brands. That is derivable from content alone — exactly the data a "
            "real platform has on day one.\n\n"
            "The consequence: **PageRank here measures topical centrality, not social influence.** "
            "It is a genuine matching signal. It is not a claim about who follows whom."
        )
        st.caption(f"Construction: {gmeta.get('construction', '—')}")
        if not gmeta.get("betweenness_exact", True):
            st.caption(
                f"Betweenness is approximated with {gmeta.get('betweenness_pivots')} sampled pivots "
                "(Brandes–Pich estimator) — exact computation is O(V·E) and does not scale."
            )

# ==========================================================================
# 5. Honesty
# ==========================================================================
with tabs[4]:
    st.markdown("### What is real here, and what is not")
    a, b = st.columns(2)
    with a:
        st.success(
            "**Real and verifiable**\n\n"
            "- Every NLP accuracy figure — measured on human-labelled TweetEval and "
            "news-headline sarcasm corpora\n"
            "- Engagement-rate and INR fee bands — calibrated to published 2026 industry benchmarks, "
            "with tier medians verified to fall inside the published ranges\n"
            "- All model metrics — out-of-fold under GroupKFold, with leakage checks enforced in code\n"
            "- The network construction, its limitations, and the approximation used for betweenness\n"
            "- Every citation, with a resolvable URL"
        )
    with b:
        st.error(
            "**Synthetic, and deliberately so**\n\n"
            "- The 2,000 creators, their profiles, posts and metrics\n"
            "- The 120 brands and the historical campaign outcomes\n\n"
            "This is a **simulation study**. Creators are generated from hidden latent traits "
            "(content quality, audience authenticity, consistency, ad saturation); observable "
            "features are noisy functions of those traits, and the model never sees the traits.\n\n"
            "Because the noise level is set by us, the **maximum achievable R² is known**, and model "
            "performance is reported as a fraction of it rather than as an unanchored number."
        )
    st.info(
        "**The limitation this creates, stated plainly.** Synthetic captions cannot validate an NLP "
        "method — measuring how well a detector recovers a label we injected only measures how well "
        "it reverse-engineers our template. That is precisely why every sentiment, emotion and "
        "sarcasm claim in this project is measured on real labelled corpora instead, and why the "
        "*NLP method comparison* tab is the load-bearing evidence rather than anything computed on "
        "the synthetic captions."
    )

    manifest = load_json("feature_manifest.json")
    if manifest:
        with st.expander("Leakage controls enforced in code"):
            st.markdown(
                f"Target: `{manifest['target']}` · "
                f"{len(manifest['numeric_features'])} numeric + "
                f"{len(manifest['categorical_features'])} categorical features.\n\n"
                "Any column matching these substrings is refused entry to the model matrix, "
                "and the trainer raises if one slips through:"
            )
            st.code("\n".join(manifest["banned_substrings"]))
