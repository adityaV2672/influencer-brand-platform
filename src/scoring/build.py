"""
Build the three-tier scores and write what the app reads.

    python -m src.scoring.build
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import ARTIFACT_DIR, ROOT
from src.scoring import capability as CAP
from src.scoring import engine as E

APP = ROOT / "app_data"
ART = ARTIFACT_DIR / "scoring"
ART.mkdir(parents=True, exist_ok=True)


def main() -> None:
    creators = pd.read_parquet(APP / "nectar_creators.parquet")
    creators["influencer_id"] = creators.influencer_id.astype(str)
    latents = pd.read_parquet(ROOT / "data" / "processed" / "latents.parquet")
    aq = pd.read_parquet(APP / "nectar_audience_quality.parquet")
    comm = pd.read_parquet(APP / "nectar_comment_profile.parquet")
    vis = pd.read_parquet(APP / "nectar_visual.parquet")
    insights = pd.read_parquet(APP / "nectar_creator_insights.parquet")
    fit = pd.read_parquet(APP / "nectar_fit.parquet")
    campaigns = pd.read_parquet(APP / "nectar_campaigns.parquet")

    print("  capability and availability ...")
    cap = CAP.build(creators, latents)
    cap.to_parquet(APP / "nectar_capability.parquet", index=False)
    print(f"    {cap.n_formats.mean():.2f} formats offered on average; "
          f"{(cap.availability_status == 'Available').mean():.0%} free now")

    print("  creator quality ...")
    q = E.creator_quality(creators, aq, comm, vis, insights)
    q.to_parquet(APP / "nectar_creator_quality.parquet", index=False)
    print(f"    mean {q.creator_quality.mean():.1f}, "
          f"{(q.verified_metrics).mean():.0%} with verified metrics; "
          + ", ".join(f"{k} {v}" for k, v in
                      q.creator_quality_band.value_counts().items()))

    print("  organisation fit ...")
    emb = np.load(ARTIFACT_DIR / "reco" / "visual_embeddings.npy")
    order = {k: i for i, k in enumerate(vis.influencer_id.astype(str))}
    brands = pd.read_parquet(APP / "brands.parquet")
    camp_brands = brands[brands.brand_id.isin(campaigns.brand_id)]
    bvec = __import__("src.reco.visual", fromlist=["x"]).brand_visual_profile(
        camp_brands, creators, emb, vis)

    pairs = fit[fit.brand_id.isin(camp_brands.brand_id)].copy()
    pairs["influencer_id"] = pairs.influencer_id.astype(str)
    bidx = {b: i for i, b in enumerate(camp_brands.brand_id)}
    rows = pairs.influencer_id.map(order).fillna(0).astype(int).to_numpy()
    cols = pairs.brand_id.map(bidx).fillna(0).astype(int).to_numpy()
    vsim = pd.Series(np.clip((emb[rows] * bvec[cols]).sum(axis=1), 0, 1),
                     index=pairs.index)

    org = E.organisation_fit(pairs, q, creators, comm, aq, visual_sim=vsim)
    org.to_parquet(APP / "nectar_org_fit.parquet", index=False)
    print(f"    {len(org):,} brand-creator pairs; mean org fit "
          f"{org.org_fit.mean():.1f}; {int(org.org_blocked.sum()):,} blocked")

    print("  campaign fit ...")
    mentions = pd.read_parquet(APP / "nectar_brand_mentions.parquet")
    frames, n_conflict = [], 0
    for camp in campaigns.itertuples():
        o = org[org.brand_id == camp.brand_id]
        conflicted = E.competitor_conflict(camp, mentions)
        n_conflict += len(conflicted)
        frames.append(E.campaign_fit(camp, creators, o, cap, CAP, conflicted))
    print(f"    competitor veto fires on {n_conflict:,} campaign-creator pairs")
    cf = pd.concat(frames, ignore_index=True)
    cf.to_parquet(APP / "nectar_campaign_fit.parquet", index=False)
    elig = ~cf.blocked
    print(f"    {len(cf):,} campaign-creator rows; {int(elig.sum()):,} eligible, "
          f"{int((~elig).sum()):,} blocked")
    print(f"    mean campaign fit among eligible: {cf.loc[elig,'campaign_fit'].mean():.1f}")

    summary = {
        "weights": {"creator_quality": E.CREATOR_QUALITY_WEIGHTS,
                    "organisation_fit": E.ORG_FIT_WEIGHTS,
                    "campaign_fit": E.CAMPAIGN_FIT_WEIGHTS},
        "creator_quality": {"mean": round(float(q.creator_quality.mean()), 2),
                            "verified_share": round(float(q.verified_metrics.mean()), 4),
                            "bands": q.creator_quality_band.value_counts().to_dict()},
        "org_fit": {"pairs": int(len(org)), "mean": round(float(org.org_fit.mean()), 2),
                    "blocked": int(org.org_blocked.sum())},
        "campaign_fit": {"rows": int(len(cf)), "eligible": int(elig.sum()),
                         "blocked": int((~elig).sum())},
        "note": "Hard gates block rather than deduct. A blocked creator carries "
                "no campaign fit score at all, only a reason.",
    }
    (ART / "scoring_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  wrote {ART / 'scoring_summary.json'}")


if __name__ == "__main__":
    main()
