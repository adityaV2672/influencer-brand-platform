"""
Export the slim artifact bundle the dashboard reads.

The deployed app must run inside roughly 1 GB of RAM on free hosting, which
rules out loading SBERT, RoBERTa or BERTopic at request time. The pattern used
here is the standard offline-scoring / online-serving split:

    offline (this file, run once)   heavy models, full corpus, all inference
    online  (app/)                  reads precomputed parquet, does no ML

The only exception is the on-demand LLM tone analysis, which runs solely when a
local Ollama instance is reachable and is hidden in the hosted build.

Everything written to app_data/ is deliberately small enough to commit to git,
so deployment is a push rather than a data-pipeline problem.
"""
from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd

from src.config import ARTIFACT_DIR, PROCESSED_DIR, ROOT
from src.features.build_features import FEATURE_DIR
from src.models.brandfit import BRANDFIT_DIR
from src.models.train import MODEL_DIR
from src.network.sna import GRAPH_DIR
from src.nlp.pipeline import NLP_DIR
from src.nlp.topics import TOPIC_DIR

APP_DATA = ROOT / "app_data"
APP_DATA.mkdir(parents=True, exist_ok=True)

# Columns the dashboard actually shows or filters on. Anything else is dropped
# to keep the payload small and to avoid shipping internals to the browser.
INFLUENCER_COLS = [
    "influencer_id", "handle", "primary_niche", "secondary_niche",
    "followers", "follower_tier", "following", "follower_following_ratio",
    "avg_likes", "avg_comments", "avg_views", "avg_reach",
    "engagement_rate", "er_vs_benchmark", "comments_to_likes", "views_to_followers",
    "follower_growth_rate", "posting_frequency_month",
    "audience_geo", "audience_age_band", "audience_gender_skew",
    "degree_centrality", "pagerank", "pagerank_pct", "eigenvector_centrality",
    "betweenness_centrality", "closeness_centrality", "community",
    "community_size", "network_tier", "k_core",
    "top_keywords", "top_hashtags", "n_posts_analysed",
]

CONTENT_PREFIXES = ("content_",)


def _predict_generic_performance(inf: pd.DataFrame) -> pd.DataFrame:
    """Score every creator with the trained model under a neutral brand context.

    The model was trained on (creator, brand) pairs. For the standalone
    "Influencer Performance Score" the dashboard shows, we hold the brand
    context at a neutral setting - in-category, matching geo and age, median
    budget - so the score reflects the creator, not a particular brief. The
    brand-specific number is the Brand-Fit score, which is shown separately.
    """
    import joblib

    from src.models.train import apply_categories

    bundle = joblib.load(MODEL_DIR / "performance_model.joblib")
    model, num, cat = bundle["model"], bundle["numeric"], bundle["categorical"]
    cats = bundle.get("categories", {})

    X = inf.copy()
    X["match_primary_niche"] = 1
    X["match_secondary_niche"] = 0
    X["match_geo"] = 1
    X["match_age"] = 1
    X["log_budget"] = float(np.log10(400_000))
    X["brand_category"] = X["primary_niche"]

    for c in num:
        if c not in X.columns:
            X[c] = np.nan
    for c in cat:
        if c not in X.columns:
            X[c] = "unknown"
    # Re-apply the exact training categories - see apply_categories().
    X = apply_categories(X, cats)
    for c in cat:
        if not isinstance(X[c].dtype, pd.CategoricalDtype):
            X[c] = X[c].astype("category")

    pred_log = model.predict(X[num + cat])
    pred = np.exp(pred_log)

    out = pd.DataFrame({"influencer_id": inf["influencer_id"], "predicted_campaign_er": pred})

    # THREE RANKING MODES, and the reason there are three.
    #
    # Ranking by predicted engagement RATE alone is mathematically guaranteed to
    # put the smallest accounts on top, because engagement rate falls with
    # audience size. A brand with a large budget then opens the product and sees
    # nothing but 800-follower creators - technically the correct answer to
    # "who gets the highest engagement rate", and useless as a shortlist.
    #
    # Ranking by predicted TOTAL engagements has the opposite bias: it collapses
    # to "who has the most followers", which is the vanity metric this whole
    # project exists to argue against.
    #
    # Neither is correct in isolation, because the right answer depends on the
    # campaign objective - and that is a business decision, not a technical one.
    # So all three are computed here and the choice is exposed in the UI rather
    # than hidden inside a default sort.
    out["predicted_total_engagements"] = pred * inf["followers"].to_numpy()

    out["score_rate"] = (out["predicted_campaign_er"].rank(pct=True) * 100).round(1)
    out["score_reach"] = (out["predicted_total_engagements"].rank(pct=True) * 100).round(1)
    # Balanced: mean of the two percentile ranks, re-ranked so the result is
    # itself a percentile. Averaging percentiles rather than raw values keeps
    # the blend from being dominated by the heavier-tailed of the two.
    out["score_balanced"] = (
        ((out["score_rate"] + out["score_reach"]) / 2).rank(pct=True) * 100
    ).round(1)

    # `performance_score` stays as the rate-based score for backwards
    # compatibility with the profile pages and the price model.
    out["performance_score"] = out["score_rate"]
    for col in ("score_rate", "score_reach", "score_balanced"):
        out[f"band_{col.split('_')[1]}"] = pd.cut(
            out[col], bins=[0, 40, 75, 100.01],
            labels=["Low", "Medium", "High"], right=False,
        ).astype(str)
    out["performance_band"] = out["band_rate"]
    return out


def _predict_price(inf: pd.DataFrame) -> pd.DataFrame:
    import joblib

    from src.models.train import apply_categories

    bundle = joblib.load(MODEL_DIR / "price_model.joblib")
    model, num, cat, band = bundle["model"], bundle["numeric"], bundle["categorical"], bundle["band"]
    cats = bundle.get("categories", {})

    X = inf.copy()
    X["match_primary_niche"] = 1
    X["match_secondary_niche"] = 0
    X["match_geo"] = 1
    X["match_age"] = 1
    X["log_budget"] = float(np.log10(400_000))
    X["brand_category"] = X["primary_niche"]
    for c in num:
        if c not in X.columns:
            X[c] = np.nan
    for c in cat:
        if c not in X.columns:
            X[c] = "unknown"
    X = apply_categories(X, cats)
    for c in cat:
        if not isinstance(X[c].dtype, pd.CategoricalDtype):
            X[c] = X[c].astype("category")

    fee = np.exp(model.predict(X[num + cat]))
    return pd.DataFrame(
        {
            "influencer_id": inf["influencer_id"],
            "price_estimate_inr": fee.round(0),
            "price_low_inr": (fee * band["low"]).round(0),
            "price_high_inr": (fee * band["high"]).round(0),
        }
    )


def run() -> dict:
    inf = pd.read_parquet(FEATURE_DIR / "influencer_features.parquet")

    print("  scoring creators with the trained models ...")
    perf = _predict_generic_performance(inf)
    price = _predict_price(inf)

    keep = [c for c in INFLUENCER_COLS if c in inf.columns]
    keep += [c for c in inf.columns if c.startswith(CONTENT_PREFIXES) and c not in keep]
    slim = inf[keep].merge(perf, on="influencer_id").merge(price, on="influencer_id")

    # Downcast to keep the hosted memory footprint small.
    for c in slim.select_dtypes("float64").columns:
        slim[c] = slim[c].astype("float32")
    for c in slim.select_dtypes("int64").columns:
        slim[c] = pd.to_numeric(slim[c], downcast="integer")

    slim.to_parquet(APP_DATA / "influencers.parquet", index=False)

    # --- supporting tables ---------------------------------------------------
    written = {"influencers.parquet": len(slim)}

    brands = pd.read_parquet(PROCESSED_DIR / "brands.parquet")
    brands.to_parquet(APP_DATA / "brands.parquet", index=False)
    written["brands.parquet"] = len(brands)

    copies = [
        (BRANDFIT_DIR / "brand_fit_matrix.parquet", "brand_fit.parquet"),
        (GRAPH_DIR / "edges_top.parquet", "edges.parquet"),
        (TOPIC_DIR / "bertopic_topics.parquet", "topics.parquet"),
        (ARTIFACT_DIR / "benchmarks" / "results.parquet", "benchmark_results.parquet"),
        (MODEL_DIR / "performance_importance.parquet", "feature_importance.parquet"),
        (MODEL_DIR / "performance_shap.parquet", "feature_shap.parquet"),
    ]
    for src, dst in copies:
        if src.exists():
            shutil.copy(src, APP_DATA / dst)
            try:
                written[dst] = len(pd.read_parquet(APP_DATA / dst))
            except Exception:  # noqa: BLE001
                written[dst] = "?"
        else:
            print(f"    ! missing {src.name} - dashboard section will show a notice")

    # A representative sample of posts so the dashboard can show real captions
    # with their per-post NLP labels, without shipping 50k rows.
    pf = NLP_DIR / "post_features.parquet"
    if pf.exists():
        posts_raw = pd.read_parquet(PROCESSED_DIR / "posts.parquet")[
            ["post_id", "influencer_id", "caption", "likes", "comments", "views", "days_ago"]
        ]
        pfeat = pd.read_parquet(pf)
        keep_cols = [c for c in pfeat.columns if c in {
            "post_id", "vader_compound", "vader_label", "roberta_sentiment",
            "roberta_p_irony", "roberta_is_ironic", "topic_id", "topic_label",
            "has_promo", "has_cta", "has_disclosure", "n_hashtags", "n_words",
            "brands_mentioned", "products_mentioned",
        }]
        merged = posts_raw.merge(pfeat[keep_cols], on="post_id", how="inner")
        # sort-then-head rather than groupby.apply(nlargest): the apply form
        # needs `include_groups` (pandas >= 2.2) and that flag *drops the group
        # key*, silently losing influencer_id from the output.
        sample = (
            merged.sort_values("likes", ascending=False)
            .groupby("influencer_id", as_index=False, sort=False)
            .head(8)
            .reset_index(drop=True)
        )
        assert "influencer_id" in sample.columns, "posts sample lost its group key"
        sample.to_parquet(APP_DATA / "posts_sample.parquet", index=False)
        written["posts_sample.parquet"] = len(sample)

    # --- json side-cars ------------------------------------------------------
    for src, dst in [
        (MODEL_DIR / "model_results.json", "model_results.json"),
        (TOPIC_DIR / "coherence.json", "topic_coherence.json"),
        (GRAPH_DIR / "graph_meta.json", "graph_meta.json"),
        (NLP_DIR / "nlp_report.json", "nlp_report.json"),
        (FEATURE_DIR / "feature_manifest.json", "feature_manifest.json"),
        (BRANDFIT_DIR / "brandfit_config.json", "brandfit_config.json"),
        (ARTIFACT_DIR / "benchmarks" / "run_meta.json", "benchmark_meta.json"),
    ]:
        if src.exists():
            shutil.copy(src, APP_DATA / dst)

    total_mb = sum(f.stat().st_size for f in APP_DATA.iterdir() if f.is_file()) / 1e6
    manifest = {
        "files": written,
        "total_size_mb": round(total_mb, 2),
        "note": "All heavy inference is precomputed. The dashboard loads no ML models.",
    }
    (APP_DATA / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"    exported {len(list(APP_DATA.iterdir()))} files, {total_mb:.1f} MB total")
    for k, v in written.items():
        print(f"      {k:<28} {v}")
    return manifest


if __name__ == "__main__":
    run()
