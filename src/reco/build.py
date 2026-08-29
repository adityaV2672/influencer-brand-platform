"""
Build the recommendation layer and write what the app reads.

    python -m src.reco.build
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import ARTIFACT_DIR, ROOT
from src.reco import cf as CF
from src.reco import interactions as IX
from src.reco import ranker as RK
from src.reco import visual as V

APP = ROOT / "app_data"
ART = ARTIFACT_DIR / "reco"
ART.mkdir(parents=True, exist_ok=True)


def main() -> None:
    creators = pd.read_parquet(APP / "nectar_creators.parquet")
    latents = pd.read_parquet(ROOT / "data" / "processed" / "latents.parquet")
    aq = pd.read_parquet(APP / "nectar_audience_quality.parquet")
    comm = pd.read_parquet(APP / "nectar_comment_profile.parquet")
    brands = pd.read_parquet(APP / "brands.parquet")

    print("  visual embeddings ...")
    emb, vmeta = V.creator_embeddings(creators, latents)
    vres = V.evaluate(emb, creators)
    np.save(ART / "visual_embeddings.npy", emb)
    vmeta.to_parquet(APP / "nectar_visual.parquet", index=False)
    print(f"    niche recovered from the visual centroid: macro F1 "
          f"{vres['macro_f1']:.4f} (majority {vres['majority_accuracy']:.4f})")

    print("  behavioural log ...")
    attrs = IX.creator_attributes(creators, aq, vmeta, comm)
    tastes = IX.brand_tastes(brands.brand_id)
    log = IX.simulate(brands, attrs, tastes, creators)
    attrs.to_parquet(APP / "nectar_creator_attributes.parquet", index=False)
    tastes.to_parquet(APP / "nectar_brand_taste.parquet", index=False)
    log.to_parquet(APP / "nectar_interactions.parquet", index=False)
    print(f"    {len(log):,} events, {log.brand_id.nunique()} brands, "
          f"{int(log.completed.sum()):,} completed deals")

    print("  learned ranking ...")
    rk = RK.train(log, attrs)
    rk["scored"].to_parquet(APP / "nectar_ranker_scores.parquet", index=False)
    for a in rk["results"]["arms"]:
        print(f"    {a['arm']:<30} NDCG@10 {a['ndcg@10']:.4f}")

    print("  collaborative filtering ...")
    M, blist, clist = CF.matrix(log)
    cres = CF.evaluate(M)
    recon = CF.factors(M)
    cf_long = pd.DataFrame({
        "brand_id": np.repeat(blist, len(clist)),
        "influencer_id": np.tile(clist, len(blist)),
        "cf_score": recon.ravel().round(4),
    })
    # Only the top 60 per brand ship; the full 120 x 1,452 matrix is not
    # something the dashboard ever needs to read.
    cf_long = (cf_long.sort_values(["brand_id", "cf_score"], ascending=[True, False])
               .groupby("brand_id").head(60).reset_index(drop=True))
    cf_long.to_parquet(APP / "nectar_cf_scores.parquet", index=False)
    print(f"    hit@10 {cres['cf_hit@10']:.4f} vs popularity "
          f"{cres['pop_hit@10']:.4f}; median rank {cres['median_rank_cf']} "
          f"vs {cres['median_rank_popularity']}")

    summary = {"visual": vres, "ranker": rk["results"], "cf": cres,
               "interaction_log": json.loads(
                   (ART / "interaction_log_meta.json").read_text())}
    (ART / "reco_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  wrote {ART / 'reco_summary.json'}")


if __name__ == "__main__":
    main()
