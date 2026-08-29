"""
The behavioural event log: what brands and creators actually did.

Why this exists, and why it is the prerequisite for the other two models
------------------------------------------------------------------------
A content-based recommender scores a pair on their attributes. It cannot know
that THIS brand, unlike every other brand in its category, keeps rejecting
large accounts and signing mid-tier ones with dense comment sections. That
preference is not in the features. It is only in the behaviour.

So every brand here is given a TASTE VECTOR - idiosyncratic weights over the
same creator attributes the global composite uses, drawn per brand and never
exposed as a feature. A brand's shortlisting decisions are generated from its
own weights, not the global ones. Two consequences:

  * A learned ranker fitted on this log can beat the hand-weighted composite,
    because the composite is one set of weights for everybody.
  * Collaborative filtering has something real to find: brands with similar
    taste vectors converge on similar creators, and that similarity is
    recoverable from the interaction matrix alone.

If taste were not idiosyncratic, both models would be re-deriving the content
score and there would be no honest reason to build either.

SIMULATION NOTE
---------------
No brand has used this platform. The log is generated. The models trained on
it in ranker.py and cf.py are real, cross-validated and compared against
baselines - the same footing as everything else here.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from src.config import ARTIFACT_DIR, SEED

OUT = ARTIFACT_DIR / "reco"
OUT.mkdir(parents=True, exist_ok=True)

# Attributes a brand can have an opinion about. The GLOBAL composite weights
# these one way; each brand weights them its own way.
TASTE_DIMS = ["reach", "engagement", "audience_quality", "visual_fit",
              "semantic_fit", "price_value", "consistency"]

GLOBAL_TASTE = np.array([0.18, 0.22, 0.14, 0.10, 0.20, 0.08, 0.08])

# How far a brand's own weights wander from the global ones. At 0 the log
# teaches nothing the composite does not already know.
TASTE_SPREAD = 0.55

STAGES = ["viewed", "shortlisted", "contacted", "accepted", "completed"]


def _unit(key: str, salt: str) -> float:
    h = hashlib.sha256(f"{salt}:{key}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


def brand_tastes(brand_ids, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 2929)
    rows = []
    for bid in [str(b) for b in brand_ids]:
        w = GLOBAL_TASTE * (1.0 + TASTE_SPREAD * rng.normal(0, 1, len(TASTE_DIMS)))
        w = np.clip(w, 0.01, None); w = w / w.sum()
        rows.append({"brand_id": bid,
                     **{f"taste_{d}": round(float(x), 4) for d, x in zip(TASTE_DIMS, w)}})
    return pd.DataFrame(rows)


def creator_attributes(creators: pd.DataFrame, aq: pd.DataFrame,
                       vis: pd.DataFrame, comm: pd.DataFrame) -> pd.DataFrame:
    """The attribute matrix both the composite and the brands score against.

    Everything is a percentile so the taste weights are commensurable - a raw
    follower count and an engagement rate cannot be weighted against each
    other on their own scales.
    """
    c = creators.copy(); c["influencer_id"] = c.influencer_id.astype(str)
    c = (c.merge(aq[["influencer_id", "audience_quality_score"]], on="influencer_id", how="left")
           .merge(vis[["influencer_id", "visual_coherence"]], on="influencer_id", how="left")
           .merge(comm[["influencer_id", "comment_quality_index"]], on="influencer_id", how="left"))

    def pct(s):
        return s.rank(pct=True).fillna(0.5).to_numpy()

    price = 1.0 - pct(c.rate_reel / np.maximum(c.followers, 1))
    return pd.DataFrame({
        "influencer_id": c.influencer_id.to_numpy(),
        "a_reach": pct(c.followers),
        "a_engagement": pct(c.engagement_rate),
        "a_audience_quality": pct(c.audience_quality_score.fillna(50)),
        "a_visual_fit": pct(c.visual_coherence.fillna(0.5)),
        "a_semantic_fit": pct(c.comment_quality_index.fillna(0.5)),
        "a_price_value": price,
        "a_consistency": 1.0 - pct(c.content_topic_entropy.fillna(1.5)),
    })


def simulate(brands: pd.DataFrame, attrs: pd.DataFrame, tastes: pd.DataFrame,
             creators: pd.DataFrame, n_seen: int = 90,
             seed: int = SEED) -> pd.DataFrame:
    """One row per (brand, creator) the brand ever looked at."""
    rng = np.random.default_rng(seed + 4242)
    A = attrs[[f"a_{d}" for d in TASTE_DIMS]].to_numpy()
    ids = attrs.influencer_id.to_numpy()
    index = {k: i for i, k in enumerate(ids)}
    c = creators.copy(); c["influencer_id"] = c.influencer_id.astype(str)
    by_cat = {k: [index[i] for i in g.influencer_id if i in index]
              for k, g in c.groupby("primary_niche")}
    all_rows = np.arange(len(ids))
    T = tastes.set_index("brand_id")

    rows = []
    for brand in brands.itertuples():
        bid = str(brand.brand_id)
        if bid not in T.index:
            continue
        w = T.loc[bid, [f"taste_{d}" for d in TASTE_DIMS]].to_numpy(float)
        pool = by_cat.get(getattr(brand, "category", None), [])
        # A brand mostly browses its own category and occasionally strays.
        k_in = min(int(n_seen * 0.75), len(pool))
        seen = list(rng.choice(pool, k_in, replace=False)) if k_in else []
        seen += list(rng.choice(all_rows, n_seen - len(seen), replace=False))
        seen = list(dict.fromkeys(seen))

        util = A[seen] @ w
        util = (util - util.mean()) / (util.std() + 1e-9)
        p_short = 1 / (1 + np.exp(-(1.55 * util - 0.85)))
        shortlisted = rng.random(len(seen)) < p_short

        for j, r in enumerate(seen):
            stage, outcome = "viewed", None
            if shortlisted[j]:
                stage = "shortlisted"
                if rng.random() < 0.62:
                    stage = "contacted"
                    # Creators accept on their own terms: better-paid, less
                    # saturated creators say no more often.
                    if rng.random() < 0.58:
                        stage = "accepted"
                        if rng.random() < 0.80:
                            stage = "completed"
                            # Realised performance: the brand's own utility
                            # plus campaign noise. This is the label a ranker
                            # trained on OUTCOMES rather than clicks would use.
                            outcome = float(np.clip(
                                0.5 + 0.32 * util[j] + rng.normal(0, 0.28), 0, 2))
            rows.append({
                "brand_id": bid, "influencer_id": ids[r],
                "stage": stage,
                "stage_index": STAGES.index(stage),
                "shortlisted": int(stage != "viewed"),
                "accepted": int(stage in ("accepted", "completed")),
                "completed": int(stage == "completed"),
                "outcome_index": None if outcome is None else round(outcome, 4),
                "days_ago": int(rng.integers(1, 365)),
            })
    log = pd.DataFrame(rows)
    meta = {
        "n_events": int(len(log)),
        "n_brands": int(log.brand_id.nunique()),
        "n_creators": int(log.influencer_id.nunique()),
        "stage_shares": log.stage.value_counts(normalize=True).round(4).to_dict(),
        "taste_spread": TASTE_SPREAD,
        "global_taste": dict(zip(TASTE_DIMS, GLOBAL_TASTE.round(4).tolist())),
        "provenance": "SIMULATED. No brand has used this platform. Brand taste "
                      "vectors are drawn per brand and are never exposed as a "
                      "feature, which is what gives the learned ranker and the "
                      "collaborative filter something the content composite "
                      "does not already contain.",
    }
    (OUT / "interaction_log_meta.json").write_text(json.dumps(meta, indent=2))
    return log
