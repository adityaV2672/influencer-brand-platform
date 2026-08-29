"""
Write every dataset the platform uses to CSV, with a data dictionary.

    python -m src.features.export_csv

Two things this is careful about:

  * CSV is a lossy container. List and dict columns (a creator's platform
    split, an audience age distribution) are JSON-encoded rather than dumped
    as Python reprs, so the file round-trips.
  * A folder of CSVs with no column definitions is not documentation. Every
    column in every table appears in DATA_DICTIONARY.csv with its type, an
    example value, and where the number came from - generated, measured,
    model output, or simulated for the product surface.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import ARTIFACT_DIR, ROOT

APP_DATA = ROOT / "app_data"
CSV_DIR = ROOT / "data" / "csv"

# Every table, where it comes from, and what it is for.
TABLES = [
    ("creators", APP_DATA / "nectar_creators.parquet",
     "One row per creator: the full feature table plus the presentation fields "
     "the product shows (name, city, platforms, rate card)."),
    ("campaigns", APP_DATA / "nectar_campaigns.parquet",
     "The six showcase campaigns, their budgets, briefs and eligibility policy."),
    ("audio_posts", APP_DATA / "nectar_audio_posts.parquet",
     "SIMULATED voice track for every video post: valence, arousal, speech "
     "rate, pause and music ratio, the audio sentiment label, and whether it "
     "disagrees with the caption model. No audio exists (src/nlp/audio_sim.py)."),
    ("audio_creators", APP_DATA / "nectar_audio_creators.parquet",
     "SIMULATED per-creator voice profile aggregated from audio_posts. Two of "
     "its columns feed the content-safety component of the fit composite."),
    ("creator_terms", APP_DATA / "nectar_creator_terms.parquet",
     "The lexical profile behind the typed-brief matcher: the forty heaviest "
     "TF-IDF terms for each creator, L2-normalised."),
    ("vocabulary", APP_DATA / "nectar_vocab.parquet",
     "Every term the typed-brief matcher can recognise, with its document "
     "frequency and inverse document frequency."),
    ("brand_mentions", APP_DATA / "nectar_brand_mentions.parquet",
     "Which creator mentioned which brand, how often, and whether any of it was "
     "a disclosed paid post - the evidence the competitor-conflict veto reads."),
    ("campaign_fit", APP_DATA / "nectar_fit.parquet",
     "Every (campaign, creator) pair with its fit composite, components, safety "
     "gates and ranking under each objective."),
    ("requests", APP_DATA / "nectar_requests.parquet",
     "The request funnel: who was approached for which campaign, at what fee, "
     "and how far they got."),
    ("messages", APP_DATA / "nectar_messages.parquet",
     "Negotiation threads attached to requests."),
    ("funnel", APP_DATA / "nectar_funnel.parquet",
     "Cumulative count of requests reaching each funnel stage, per campaign."),
    ("campaign_summary", APP_DATA / "nectar_campaign_summary.parquet",
     "Campaign-level results including predicted-vs-actual calibration."),
    ("creator_performance", APP_DATA / "nectar_creator_performance.parquet",
     "Per-creator delivered reach, engagement, cost, CPE and CPR."),
    ("creator_history", APP_DATA / "nectar_creator_history.parquet",
     "Past brand approaches per creator, outside the six showcase campaigns."),
    ("earnings", APP_DATA / "nectar_earnings.parquet",
     "Monthly realised earnings per creator."),
    ("category_fit", APP_DATA / "nectar_category_fit.parquet",
     "Each creator's brand-fit score against a representative brand in every category."),
    ("model_calibration", APP_DATA / "nectar_calibration.parquet",
     "Out-of-fold predicted vs actual campaign engagement rate, by brand category."),
    ("brands", APP_DATA / "brands.parquet",
     "The synthetic brand universe: category, budget, target audience, competitors."),
    ("brand_fit_sbert", APP_DATA / "brand_fit.parquet",
     "The pipeline's SBERT-scored brand-fit matrix (top 60 creators per brand)."),
    ("posts_sample", APP_DATA / "posts_sample.parquet",
     "A sample of posts with captions and their per-post NLP labels."),
    ("modelling_table", ARTIFACT_DIR / "features" / "modelling_table.parquet",
     "The table the performance model is trained on: one row per campaign, with "
     "the creator's features and the realised campaign outcome."),
    ("influencer_features", ARTIFACT_DIR / "features" / "influencer_features.parquet",
     "The full engineered feature table, before the product layer is applied."),
    ("nlp_benchmarks", APP_DATA / "benchmark_results.parquet",
     "Accuracy and macro-F1 for every NLP method on every human-labelled corpus."),
    ("feature_importance", APP_DATA / "feature_importance.parquet",
     "LightGBM split gain per feature."),
    ("feature_shap", APP_DATA / "feature_shap.parquet",
     "Mean absolute SHAP contribution per feature."),
    ("graph_edges", APP_DATA / "edges.parquet",
     "The strongest co-behaviour edges between creators."),
    ("topics", APP_DATA / "topics.parquet",
     "BERTopic topics with their top words."),
]

# Provenance, by column-name pattern. Checked in order; first match wins.
PROVENANCE = [
    (("influencer_id", "post_id", "brand_id", "campaign_id", "request_id"),
     "identifier"),
    (("name", "nectar_handle", "initials", "avatar_color", "city", "bio",
      "available_window", "availability"),
     "presentation — generated deterministically from the creator id, never a model input"),
    (("campaign_fit", "org_fit", "brand_fit", "fit_", "semantic_"),
     "model output — brand-fit composite (src/models/brandfit.py)"),
    (("predicted_", "score_", "performance_score", "price_", "rate_", "brief_fee"),
     "model output — LightGBM performance or price model"),
    (("content_",), "measured — NLP pipeline over the creator's captions"),
    (("pagerank", "degree_", "eigenvector", "betweenness", "closeness", "community",
      "k_core", "network_tier"),
     "measured — co-behaviour graph (src/network/sna.py)"),
    (("campaign_engagement", "campaign_engagements", "fee_inr", "actual_"),
     "generated — the synthetic universe's realised outcome"),
    (("stage_index", "status", "sent_at", "viewed_at", "responded_at", "counter_",
      "payment", "usage_rights", "exclusivity", "month", "seq", "sender", "body",
      "offer_"),
     "simulated — the transactional layer (src/nectar/build_pipeline.py)"),
    (("followers", "following", "avg_", "engagement_rate", "er_vs_benchmark",
      "comments_to_likes", "views_to_followers", "follower_", "posting_frequency",
      "audience_"),
     "generated — the synthetic creator universe (src/data/generate_synthetic.py)"),
    (("audio_", "speech_rate", "pause_ratio", "pitch_variation", "music_ratio",
      "tone_mismatch", "n_video_posts"),
     "simulated - generated voice track, no audio was ever recorded "
     "(src/nlp/audio_sim.py)"),
    (("term", "idf", "df", "weight"),
     "derived - TF-IDF over creator captions and keywords (src/nectar/build_terms.py)"),
    (("n_mentions", "n_paid", "days_ago_min"),
     "generated - brand mentions in the synthetic post corpus"),
    (("accuracy", "macro_f1", "weighted_f1", "n_eval", "texts_per_sec"),
     "measured — evaluation on a real, human-labelled corpus"),
]


def provenance_for(column: str) -> str:
    for patterns, label in PROVENANCE:
        for p in patterns:
            if column == p or column.startswith(p):
                return label
    return "derived"


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """JSON-encode any column holding lists or dicts.

    Writing them straight to CSV produces Python reprs with single quotes,
    which no CSV reader can parse back into a structure.
    """
    out = df.copy()
    for col in out.columns:
        sample = out[col].dropna()
        if sample.empty:
            continue
        first = sample.iloc[0]
        if isinstance(first, (list, dict, tuple)) or hasattr(first, "tolist"):
            out[col] = out[col].map(
                lambda v: json.dumps(v.tolist() if hasattr(v, "tolist") else v,
                                     default=str) if v is not None else "")
    return out


def _n_unique(series: pd.Series) -> int:
    """-1 where uniqueness is not defined.

    Columns holding lists or arrays (a creator's platform names, an audience
    age distribution) are unhashable, and pandas raises rather than returning
    a value - so ask the flattened text instead of the raw object.
    """
    try:
        return int(series.nunique(dropna=True))
    except TypeError:
        try:
            return int(series.map(lambda v: json.dumps(
                v.tolist() if hasattr(v, "tolist") else v, default=str)).nunique())
        except Exception:      # noqa: BLE001
            return -1


def _example(series: pd.Series) -> str:
    s = series.dropna()
    if s.empty:
        return ""
    return str(s.iloc[0])[:60]


def run(out_dir: Path | None = None) -> dict:
    out_dir = Path(out_dir or CSV_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest, dictionary = [], []
    for name, path, description in TABLES:
        if not path.exists():
            print(f"    ! missing {path.name} — skipped")
            continue
        df = pd.read_parquet(path)
        flat = _flatten(df)
        target = out_dir / f"{name}.csv"
        flat.to_csv(target, index=False)
        manifest.append({
            "file": f"{name}.csv",
            "rows": len(df),
            "columns": df.shape[1],
            "size_kb": round(target.stat().st_size / 1024, 1),
            "source_parquet": str(path.relative_to(ROOT)),
            "description": description,
        })
        for col in df.columns:
            dictionary.append({
                "table": name,
                "column": col,
                "dtype": str(df[col].dtype),
                "non_null": int(df[col].notna().sum()),
                "n_unique": _n_unique(df[col]),
                "example": _example(flat[col]),
                "provenance": provenance_for(col),
            })

    man = pd.DataFrame(manifest)
    man.to_csv(out_dir / "MANIFEST.csv", index=False)
    dic = pd.DataFrame(dictionary)
    dic.to_csv(out_dir / "DATA_DICTIONARY.csv", index=False)

    total = sum(m["size_kb"] for m in manifest)
    print(f"    wrote {len(manifest)} CSVs ({total / 1024:.1f} MB) to {out_dir}")
    print(f"    data dictionary: {len(dictionary)} columns documented")
    for m in manifest:
        print(f"      {m['file']:<28} {m['rows']:>7,} rows x {m['columns']:>3} cols")
    return {"files": manifest, "columns_documented": len(dictionary)}


if __name__ == "__main__":
    run()
