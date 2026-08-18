"""
The feature store: collapses three parallel extraction tracks into one
influencer-level table, then joins it to campaigns to form the modelling set.

Leakage rules enforced here (the part that decides whether the model results
mean anything)
--------------------------------------------------------------------------
1. Latent generative traits are NEVER joined in. They live in
   data/processed/latents.parquet and are used only to compute the theoretical
   performance ceiling reported in the evaluation.
2. `campaign_engagement_rate` and `campaign_engagements` are the target and its
   direct arithmetic parent. Neither is ever a feature.
3. `fee_inr` is excluded from the engagement model. It is an outcome of the same
   negotiation and would leak; it is the target of the *separate* price model.
4. Organic `engagement_rate` IS allowed as a feature. It is measured on the
   creator's non-sponsored history, which a real platform observes before any
   campaign is booked. It is the single strongest legitimate predictor and
   excluding it would understate what the system can actually do.
5. Aggregations are computed from the creator's post history only - never from
   the campaign rows.

FEATURE_COLUMNS below is the single source of truth; the trainer refuses to run
if any column outside it reaches the model matrix.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import ARTIFACT_DIR, PROCESSED_DIR
from src.network.sna import GRAPH_DIR
from src.nlp.pipeline import NLP_DIR

FEATURE_DIR = ARTIFACT_DIR / "features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

# Anything matching these is refused entry to the model matrix.
BANNED_SUBSTRINGS = (
    "campaign_engagement", "campaign_engagements", "fee_inr",
    "content_quality", "authenticity", "consistency", "promo_saturation", "niche_focus",
    "gen_", "category_fit_true",
)


# ==========================================================================
# Post -> influencer aggregation
# ==========================================================================


def aggregate_content(post_feats: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-post NLP features to per-influencer content features."""
    g = post_feats.groupby("influencer_id")

    agg: dict[str, tuple] = {
        "n_posts_analysed": ("post_id", "count"),
        "content_hashtags_mean": ("n_hashtags", "mean"),
        "content_hashtags_std": ("n_hashtags", "std"),
        "content_mentions_mean": ("n_mentions", "mean"),
        "content_emoji_mean": ("n_emoji", "mean"),
        "content_words_mean": ("n_words", "mean"),
        "content_words_std": ("n_words", "std"),
        "content_caps_ratio": ("caps_ratio", "mean"),
        "content_exclamations": ("exclamations", "mean"),
        "content_promo_rate": ("has_promo", "mean"),
        "content_disclosure_rate": ("has_disclosure", "mean"),
        "content_cta_rate": ("has_cta", "mean"),
        "content_question_rate": ("has_question", "mean"),
        "content_brand_mention_rate": ("n_brands_mentioned", "mean"),
        "content_product_mention_rate": ("n_products_mentioned", "mean"),
    }
    # Optional columns - present only if that NLP stage succeeded.
    optional = {
        "content_vader_mean": ("vader_compound", "mean"),
        "content_vader_std": ("vader_compound", "std"),
        "content_roberta_p_positive": ("roberta_p_positive", "mean"),
        "content_roberta_p_negative": ("roberta_p_negative", "mean"),
        "content_irony_rate": ("roberta_is_ironic", "mean"),
        "content_irony_prob_mean": ("roberta_p_irony", "mean"),
    }
    for k, (col, how) in optional.items():
        if col in post_feats.columns:
            agg[k] = (col, how)

    for emo in ("anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"):
        col = f"nrc_{emo}"
        if col in post_feats.columns:
            agg[f"content_{col}"] = (col, "mean")

    out = g.agg(**agg).reset_index()

    # Sentiment mix from whichever labeller is available (transformer preferred).
    label_col = "roberta_sentiment" if "roberta_sentiment" in post_feats.columns else "vader_label"
    if label_col in post_feats.columns:
        mix = (
            post_feats.groupby(["influencer_id", label_col]).size()
            .unstack(fill_value=0).pipe(lambda d: d.div(d.sum(axis=1), axis=0))
        )
        mix.columns = [f"content_share_{c}" for c in mix.columns]
        out = out.merge(mix.reset_index(), on="influencer_id", how="left")

    # Topic concentration: how focused is this creator's content?
    if "topic_id" in post_feats.columns:
        def _entropy(s: pd.Series) -> float:
            p = s.value_counts(normalize=True).to_numpy()
            p = p[p > 0]
            return float(-(p * np.log(p)).sum())

        topic = g["topic_id"].agg(
            content_topic_entropy=_entropy,
            content_n_topics=lambda s: s[s != -1].nunique(),
            content_dominant_topic=lambda s: (
                s[s != -1].mode().iloc[0] if (s != -1).any() else -1
            ),
            content_outlier_rate=lambda s: float((s == -1).mean()),
        ).reset_index()
        out = out.merge(topic, on="influencer_id", how="left")

    return out


# ==========================================================================
# Assembly
# ==========================================================================


def build_influencer_table() -> pd.DataFrame:
    profiles = pd.read_parquet(PROCESSED_DIR / "profiles.parquet")

    post_feats_path = NLP_DIR / "post_features.parquet"
    if post_feats_path.exists():
        content = aggregate_content(pd.read_parquet(post_feats_path))
    else:
        print("  ! no NLP post features found - content pillar will be empty")
        content = pd.DataFrame({"influencer_id": profiles["influencer_id"]})

    net_path = GRAPH_DIR / "network_features.parquet"
    network = pd.read_parquet(net_path) if net_path.exists() else pd.DataFrame(
        {"influencer_id": profiles["influencer_id"]}
    )

    kw_path = NLP_DIR / "influencer_keywords.parquet"
    keywords = pd.read_parquet(kw_path) if kw_path.exists() else pd.DataFrame(
        {"influencer_id": profiles["influencer_id"]}
    )

    df = (
        profiles
        .merge(content, on="influencer_id", how="left")
        .merge(network, on="influencer_id", how="left")
        .merge(keywords, on="influencer_id", how="left")
    )

    # Derived reach/engagement ratios that the raw columns do not express.
    df["follower_following_ratio"] = df["followers"] / df["following"].clip(lower=1)
    df["log_followers"] = np.log10(df["followers"].clip(lower=1))
    df["engagement_per_post"] = df["avg_likes"] + df["avg_comments"]
    df["reach_efficiency"] = df["avg_reach"] / df["followers"].clip(lower=1)

    # Engagement relative to the published benchmark for this size and niche -
    # this is what "good for their size" actually means, and it is far more
    # informative to a model than raw engagement rate.
    from src.data import benchmarks as bm

    bench = np.array([
        bm.expected_er(f, n) for f, n in zip(df["followers"], df["primary_niche"])
    ])
    df["er_vs_benchmark"] = df["engagement_rate"] / np.clip(bench, 1e-6, None)

    df.to_parquet(FEATURE_DIR / "influencer_features.parquet", index=False)
    return df


def build_modelling_table() -> pd.DataFrame:
    """Campaign-level rows: influencer features + brand context + target."""
    inf = pd.read_parquet(FEATURE_DIR / "influencer_features.parquet")
    campaigns = pd.read_parquet(PROCESSED_DIR / "campaigns.parquet")
    brands = pd.read_parquet(PROCESSED_DIR / "brands.parquet")

    df = (
        campaigns
        .merge(inf, on="influencer_id", how="left")
        .merge(
            brands[["brand_id", "category", "target_geo", "target_age_band", "budget_inr"]],
            on="brand_id", how="left", suffixes=("", "_brand"),
        )
    )

    # Match features: computed from observable attributes, not from the
    # generator's hidden `category_fit_true`.
    df["match_primary_niche"] = (df["primary_niche"] == df["category"]).astype(int)
    df["match_secondary_niche"] = (df["secondary_niche"] == df["category"]).astype(int)
    df["match_geo"] = (df["audience_geo"] == df["target_geo"]).astype(int)
    df["match_age"] = (df["audience_age_band"] == df["target_age_band"]).astype(int)
    df["log_budget"] = np.log10(df["budget_inr"].clip(lower=1))

    df.to_parquet(FEATURE_DIR / "modelling_table.parquet", index=False)
    return df


# ==========================================================================
# Feature selection
# ==========================================================================

CATEGORICAL = [
    "primary_niche", "secondary_niche", "follower_tier", "audience_geo",
    "audience_age_band", "network_tier", "brand_category",
]


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_features, categorical_features), leakage-checked."""
    exclude = {
        "influencer_id", "handle", "campaign_id", "brand_id",
        "top_keywords", "top_hashtags", "content_dominant_topic",
        "category", "target_geo", "target_age_band", "brand_name",
        "brand_keywords", "competitor_brands",
    }
    numeric, categorical = [], []
    for c in df.columns:
        if c in exclude:
            continue
        if any(b in c for b in BANNED_SUBSTRINGS):
            continue
        if c in CATEGORICAL:
            categorical.append(c)
        elif pd.api.types.is_numeric_dtype(df[c]):
            numeric.append(c)
    return numeric, categorical


def assert_no_leakage(cols: list[str]) -> None:
    bad = [c for c in cols if any(b in c for b in BANNED_SUBSTRINGS)]
    if bad:
        raise AssertionError(f"Leaked columns reached the model matrix: {bad}")


# ==========================================================================


def run() -> pd.DataFrame:
    print("  building influencer feature table ...")
    inf = build_influencer_table()
    print(f"    {len(inf):,} influencers x {inf.shape[1]} columns")

    print("  building campaign modelling table ...")
    model_df = build_modelling_table()
    num, cat = feature_columns(model_df)
    assert_no_leakage(num + cat)
    print(f"    {len(model_df):,} campaign rows")
    print(f"    {len(num)} numeric + {len(cat)} categorical features, leakage check passed")

    (FEATURE_DIR / "feature_manifest.json").write_text(
        json.dumps(
            {
                "n_influencers": len(inf),
                "n_campaign_rows": len(model_df),
                "numeric_features": num,
                "categorical_features": cat,
                "banned_substrings": list(BANNED_SUBSTRINGS),
                "target": "campaign_engagement_rate",
                "price_target": "fee_inr",
            },
            indent=2,
        )
    )
    return model_df


if __name__ == "__main__":
    run()
