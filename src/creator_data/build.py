"""
Build the creator-supplied data layer and everything derived from it.

    python -m src.creator_data.build --stage comments
    python -m src.creator_data.build --stage nlp
    python -m src.creator_data.build --stage quality
    python -m src.creator_data.build --stage supplied
    python -m src.creator_data.build --stage all

Staged because the comment corpus is expensive to generate and cheap to reuse,
and because a change to the NLP models should not force a regeneration of the
text they are applied to.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.config import ARTIFACT_DIR, ROOT
from src.creator_data import audience_quality as AQ
from src.creator_data import comment_nlp as CN
from src.creator_data import comments as CM
from src.creator_data import supplied as SUP

ART = ARTIFACT_DIR / "creator_data"
ART.mkdir(parents=True, exist_ok=True)
APP = ROOT / "app_data"

COMMENTS = ART / "comments.parquet"
SCORED = ART / "comments_scored.parquet"

# What ships to the app. The full corpus is hundreds of thousands of rows; the
# dashboard needs the aggregates plus enough raw comments to show a real
# comment section on a profile page.
SAMPLE_COMMENTS = 24_000


def _posts() -> pd.DataFrame:
    p = pd.read_parquet(ROOT / "data" / "processed" / "posts.parquet",
                        columns=["post_id", "influencer_id", "comments",
                                 "likes", "views", "post_niche", "days_ago"])
    p["influencer_id"] = p.influencer_id.astype(str)
    return p


def _latents() -> pd.DataFrame:
    l = pd.read_parquet(ROOT / "data" / "processed" / "latents.parquet")
    l["influencer_id"] = l.influencer_id.astype(str)
    return l


def stage_comments() -> None:
    c = CM.generate(_posts(), _latents())
    c.to_parquet(COMMENTS, index=False)
    print(f"    {len(c):,} comments from {c.post_id.nunique():,} posts, "
          f"{c.influencer_id.nunique():,} creators")
    print("    archetype mix: "
          + ", ".join(f"{k} {v:.1%}" for k, v in
                      c.archetype.value_counts(normalize=True).items()))


def stage_nlp() -> None:
    models, res = CN.train()
    c = pd.read_parquet(COMMENTS)
    scored = CN.score_comments(c, models)
    scored.to_parquet(SCORED, index=False)
    agg = CN.aggregate(scored)
    agg.to_parquet(APP / "nectar_comment_profile.parquet", index=False)
    scored.sample(min(SAMPLE_COMMENTS, len(scored)), random_state=7).to_parquet(
        APP / "nectar_comments_sample.parquet", index=False)
    print(f"    scored {len(scored):,} comments; "
          f"{agg.comment_automated_rate.mean():.1%} flagged automated on average")


def stage_quality() -> None:
    scored = pd.read_parquet(SCORED)
    agg = pd.read_parquet(APP / "nectar_comment_profile.parquet")
    inf = pd.read_parquet(APP / "influencers.parquet")
    t = AQ.truth(scored)
    feats = AQ.build_features(inf, agg)
    out = AQ.train(feats, t)
    out["scored"].to_parquet(APP / "nectar_audience_quality.parquet", index=False)
    for a in out["results"]["arms"]:
        print(f"    {a['arm']:<34} feats {a['features']:>2}  "
              f"macroF1 {a['macro_f1']:.4f}  acc {a['accuracy']:.4f}")


def stage_supplied() -> None:
    inf = pd.read_parquet(APP / "influencers.parquet")
    inf["influencer_id"] = inf.influencer_id.astype(str)
    conn = SUP.connection_status(inf.influencer_id)
    priv = SUP.post_private_metrics(_posts(), _latents(), conn)
    demo = SUP.audience_demographics(inf.influencer_id, inf, conn)

    # Creator-level rollup of the private post metrics - what a brand sees on
    # a connected creator's profile.
    g = priv.groupby("influencer_id")
    roll = pd.DataFrame({
        "avg_saves": g.saves.mean().round(1),
        "avg_shares": g.shares.mean().round(1),
        "avg_reach_verified": g.reach.mean().round(0),
        "avg_watch_time_s": g.avg_watch_time_s.mean().round(2),
        "avg_watch_through_rate": g.watch_through_rate.mean().round(4),
        "avg_dwell_seconds": g.dwell_seconds.mean().round(2),
        "avg_profile_visits": g.profile_visits.mean().round(1),
        "avg_follows_from_post": g.follows_from_post.mean().round(2),
        "save_rate": g.save_rate.mean().round(5),
        "share_rate": g.share_rate.mean().round(5),
        "n_posts_with_insights": g.size(),
    }).reset_index()

    conn.to_parquet(APP / "nectar_connections.parquet", index=False)
    roll.merge(demo, on="influencer_id", how="outer").to_parquet(
        APP / "nectar_creator_insights.parquet", index=False)
    priv.sample(min(20_000, len(priv)), random_state=11).to_parquet(
        APP / "nectar_post_insights_sample.parquet", index=False)

    print(f"    {int(conn.account_connected.sum()):,} of {len(conn):,} creators "
          f"connected ({conn.account_connected.mean():.1%})")
    print(f"    private metrics on {len(priv):,} posts; "
          f"mean save rate {priv.save_rate.mean():.3%}, "
          f"mean watch-through {priv.watch_through_rate.mean():.1%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["comments", "nlp", "quality", "supplied", "all"])
    a = ap.parse_args()
    if a.stage in ("comments", "all"):
        print("  generating the comment corpus ..."); stage_comments()
    if a.stage in ("nlp", "all"):
        print("  training comment models on real labelled tweets ..."); stage_nlp()
    if a.stage in ("quality", "all"):
        print("  audience quality ..."); stage_quality()
    if a.stage in ("supplied", "all"):
        print("  creator-supplied insights ..."); stage_supplied()


if __name__ == "__main__":
    main()
