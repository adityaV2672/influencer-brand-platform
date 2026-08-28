"""
Build the Nectar product layer and write it to app_data/.

Runs after the modelling pipeline. Reads the scored feature table, the
brand-fit matrix, the trained models' out-of-fold predictions and the campaign
outcomes, and writes the small parquet files the dashboard reads at request
time. The hosted app loads no models and does no inference - see the note in
export_app.py for why.

    python -m src.features.export_nectar
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ARTIFACT_DIR, ROOT
from src.nectar.build_campaigns import (
    add_reasons, build_campaigns, build_fit, weight_sensitivity,
)
from src.nectar.build_creators import build as build_creators
from src.nectar.build_terms import build as build_terms, build_brand_mentions
from src.nectar.build_pipeline import (
    build_creator_history, build_funnel, build_messages, build_monthly,
    build_reporting, build_requests,
)
from src.nectar.semantic_impute import build_full_fit

APP_DATA = ROOT / "app_data"
APP_DATA.mkdir(parents=True, exist_ok=True)


def _creator_earnings(requests: pd.DataFrame, creators: pd.DataFrame,
                      history: pd.DataFrame) -> pd.DataFrame:
    """Monthly earnings per creator: money actually realised.

    Built by summing the fees on deals that reached payment, by the month they
    were paid - the showcase campaigns plus the creator's own brand history.
    An earlier version spread a single total across six months with a made-up
    growth curve, which meant the Earnings chart was a shape, not a record.
    """
    months = ["Mar", "Apr", "May", "Jun", "Jul", "Aug"]
    paid_hist = history[history.status.isin(["Paid", "Approved"])][
        ["influencer_id", "month", "fee_inr"]].rename(columns={"fee_inr": "amount"})
    camp_paid = requests[requests.stage_index >= 7][["influencer_id", "fee_inr"]].copy()
    camp_paid["month"] = "Aug"          # the showcase campaigns settle this month
    camp_paid = camp_paid.rename(columns={"fee_inr": "amount"})

    both = pd.concat([paid_hist, camp_paid], ignore_index=True)
    if both.empty:
        return pd.DataFrame(columns=["influencer_id", "month", "amount"])
    out = both.groupby(["influencer_id", "month"], as_index=False)["amount"].sum()
    # Fill the gaps so every creator has a complete six-month series.
    full = pd.MultiIndex.from_product(
        [out.influencer_id.unique(), months], names=["influencer_id", "month"])
    out = out.set_index(["influencer_id", "month"]).reindex(full, fill_value=0.0)
    return out.reset_index()


def run() -> dict:
    print("  building the Nectar product layer ...")

    inf = pd.read_parquet(APP_DATA / "influencers.parquet")
    brands = pd.read_parquet(APP_DATA / "brands.parquet")
    fit_matrix = pd.read_parquet(APP_DATA / "brand_fit.parquet")
    modelling = pd.read_parquet(ARTIFACT_DIR / "features" / "modelling_table.parquet")
    oof = np.load(ARTIFACT_DIR / "models" / "performance_oof.npy")

    creators = build_creators(inf)
    campaigns = build_campaigns(brands)

    # The shipped brand-fit matrix holds the top 60 creators per brand. That is
    # too thin for a product surface - every creator appeared in exactly one
    # campaign - so it is extended to the whole creator base for the brands that
    # matter here. See src/nectar/semantic_impute.py for what is exact and what
    # is bounded.
    camp_brands = brands[brands.brand_id.isin(campaigns.brand_id)]
    full_fit, semantic_stats = build_full_fit(inf, camp_brands, fit_matrix)

    # "Which brand categories want you?" on the creator side: one representative
    # brand per category (the best-funded one), scored against every creator.
    rep = (brands.sort_values("budget_inr", ascending=False)
           .groupby("category", as_index=False).head(1))
    cat_pairs, _ = build_full_fit(inf, rep, fit_matrix)
    cat_fit = cat_pairs.merge(rep[["brand_id", "category"]], on="brand_id")
    cat_fit = cat_fit[["influencer_id", "category", "brand_fit_ungated"]].rename(
        columns={"brand_fit_ungated": "fit"})
    cat_fit["fit_pct"] = (cat_fit["fit"] * 100).round(0)
    cat_fit["brands"] = cat_fit.category.map(
        brands.groupby("category").size().to_dict())
    cat_fit = cat_fit.sort_values(["influencer_id", "fit_pct"], ascending=[True, False])
    fit = add_reasons(build_fit(full_fit, creators, campaigns))

    # How much does the shortlist depend on weights nobody validated?
    from src.models.brandfit import COMPONENT_WEIGHTS
    sensitivity = weight_sensitivity(fit, COMPONENT_WEIGHTS)
    requests = build_requests(fit, campaigns)
    messages = build_messages(requests)
    funnel = build_funnel(requests)
    camp_summary, per_creator, calibration = build_reporting(
        requests, campaigns, creators, modelling, oof)
    monthly = build_monthly(camp_summary)
    history = build_creator_history(creators, cat_fit, brands)
    earnings = _creator_earnings(requests, creators, history)

    # How many creators each campaign actually signed - shown on the Campaigns
    # table, so it has to agree with the request pipeline rather than be an
    # independent invention.
    signed = requests[requests.stage_index >= 4].groupby("campaign_id").size()
    campaigns["creators_count"] = campaigns.campaign_id.map(signed).fillna(0).astype(int)

    # Spend and progress come from the pipeline, not from a random draw, so the
    # Campaigns table, the Overview tiles and the Reporting page all quote the
    # same number.
    committed = camp_summary.set_index("campaign_id")["spend"]
    campaigns["spent_inr"] = campaigns.campaign_id.map(committed).fillna(0.0)
    campaigns["progress"] = (campaigns.spent_inr / campaigns.budget_inr).round(3)

    # The creator whose inbox the Creator OS opens on. Scored on live campaign
    # conversations first and brand history second, so the demo opens on someone
    # with something to act on rather than on an empty state.
    live_score = requests[requests.stage_index >= 2].groupby("influencer_id").size() * 3
    hist_score = history.groupby("influencer_id").size()
    total = live_score.add(hist_score, fill_value=0).sort_values(ascending=False)
    default_creator = total.index[0] if len(total) else creators.influencer_id.iloc[0]

    # ---- write ------------------------------------------------------------
    # Lexical layer for the brand intake page. A brand typing a fresh brief has
    # no precomputed SBERT row, and the hosted app carries no model, so the
    # typed text is matched against TF-IDF profiles instead. See
    # src/nectar/build_terms.py for what that costs.
    posts_path = ROOT / "data" / "processed" / "posts.parquet"
    posts = pd.read_parquet(
        posts_path,
        columns=["influencer_id", "caption", "gen_brand", "gen_has_promo", "days_ago"],
    )
    creator_terms, vocab = build_terms(creators, posts)
    brand_mentions = build_brand_mentions(posts)
    print(f"    lexical profiles: {len(vocab):,} vocabulary terms, "
          f"{len(creator_terms):,} creator-term weights")

    tables = {
        "nectar_creators.parquet": creators,
        "nectar_campaigns.parquet": campaigns,
        "nectar_fit.parquet": fit,
        "nectar_requests.parquet": requests,
        "nectar_messages.parquet": messages,
        "nectar_funnel.parquet": funnel,
        "nectar_campaign_summary.parquet": camp_summary,
        "nectar_creator_performance.parquet": per_creator,
        "nectar_calibration.parquet": calibration,
        "nectar_monthly.parquet": monthly,
        "nectar_earnings.parquet": earnings,
        "nectar_category_fit.parquet": cat_fit,
        "nectar_creator_history.parquet": history,
        "nectar_weight_sensitivity.parquet": sensitivity,
        "nectar_creator_terms.parquet": creator_terms,
        "nectar_vocab.parquet": vocab,
        "nectar_brand_mentions.parquet": brand_mentions,
    }
    written = {}
    for fname, df in tables.items():
        df.to_parquet(APP_DATA / fname, index=False)
        written[fname] = len(df)

    meta = {
        "default_creator_id": str(default_creator),
        "n_creators": int(len(creators)),
        "n_campaigns": int(len(campaigns)),
        "n_requests": int(len(requests)),
        "files": written,
        "semantic_extension": semantic_stats,
        "provenance": {
            "campaign_fit": "brand_fit_ungated from src/models/brandfit.py (SBERT semantic "
                            "similarity, category affinity, audience overlap, content safety, "
                            "consistency)",
            "typed_brief_fit": "same components and weights as campaign_fit, except the semantic term, which is TF-IDF cosine over creator captions and keywords rather than SBERT - the hosted app loads no model, so a brief typed at request time cannot be embedded",
            "org_fit": "0.45*content_safety + 0.35*consistency + 0.20*audience_match",
            "fees": "price_model.joblib (LightGBM), Reel = point estimate, "
                    "Story = 0.35x, Carousel = 0.70x",
            "predicted_vs_actual": "performance_oof.npy (GroupKFold out-of-fold predictions) "
                                   "vs campaign_engagement_rate in modelling_table.parquet",
            "simulated_here": "request funnel stages, negotiation messages, payment status, "
                              "creator names, cities, bios, avatars, availability windows",
        },
    }
    (APP_DATA / "nectar_meta.json").write_text(json.dumps(meta, indent=2))

    # The data dictionary ships with the app so the Data page can serve it
    # without the 37 MB of CSVs that live only on disk.
    try:
        from src.features.export_csv import run as export_csv
        csv_meta = export_csv()
        dic = pd.read_csv(ROOT / "data" / "csv" / "DATA_DICTIONARY.csv")
        dic.to_parquet(APP_DATA / "data_dictionary.parquet", index=False)
        meta["csv_export"] = {"tables": len(csv_meta["files"]),
                              "columns_documented": csv_meta["columns_documented"]}
        (APP_DATA / "nectar_meta.json").write_text(json.dumps(meta, indent=2))
    except Exception as exc:                                  # noqa: BLE001
        print(f"    ! CSV export skipped: {exc}")

    print(f"    default creator: {default_creator}")
    for k, v in written.items():
        print(f"      {k:<38} {v}")
    return meta


if __name__ == "__main__":
    run()
