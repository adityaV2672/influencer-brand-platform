"""
Extend the brand-fit matrix beyond the 60 creators per brand that SBERT scored.

Why this exists
---------------
`brandfit.build_matrix` scores the top 60 creators per brand. That is the right
economics for the pipeline (O(brands x k) as the creator base grows) but it
leaves the product with a thin slice: every creator appeared in exactly one
campaign, so a creator's inbox held at most one request and the "which brand
categories want you" panel had a single bar.

Four of the five fit components - category match, audience match, content
safety, consistency - are deterministic functions of columns already in the
feature table, and are recomputed exactly here using brandfit's own
`score_pair`. Together they carry 0.66 of the composite's weight.

The fifth, semantic similarity, is an SBERT cosine, and the encoder cannot be
downloaded in this environment.

What was tried first, and why it was thrown away
------------------------------------------------
The obvious move is to treat the 7,200 SBERT-scored pairs as labelled data and
regress the cosine on cheap features (a TF-IDF cosine between the same two
profile texts, category match, audience size). That was built and measured, and
it does not work: held-out RMSE 0.0529 against a predict-the-mean baseline of
0.0530. No improvement at all.

The reason is selection, not weak features. Those 7,200 rows are the TOP 60 per
brand *by cosine*, so the labelled sample is range-restricted at the high end.
A model fitted there learns nothing about the 1,940 creators below the cut.
`evaluate_regression_alternative()` reproduces the measurement, because a
negative result worth acting on is worth being able to re-run.

What is done instead
--------------------
A bound, not an estimate. SBERT returned the top 60 by cosine for each brand,
so every creator it did NOT return has a cosine no greater than the lowest of
those 60. Unscored pairs are assigned that per-brand floor.

This is conservative by construction and it preserves the one ordering fact we
actually know: creators SBERT ranked highly keep their real, higher score and
stay above the extended pool. Every extended row is flagged
`semantic_imputed = True`; the flag reaches the CSV export and the data
dictionary, and pairs SBERT genuinely scored always keep their real value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.brandfit import (
    COMPONENT_WEIGHTS, brand_profile_text, influencer_profile_text, score_pair,
)

SEED = 20260904


def _category_match(inf_row, category) -> float:
    """brandfit.score_pair's rule, kept in one place so the two cannot drift."""
    if inf_row["primary_niche"] == category:
        return 1.0
    if inf_row["secondary_niche"] == category:
        return 0.55
    return 0.15


# --------------------------------------------------------------------------
# The rejected alternative, kept runnable
# --------------------------------------------------------------------------

def evaluate_regression_alternative(influencers: pd.DataFrame, brands: pd.DataFrame,
                                    scored: pd.DataFrame) -> dict:
    """Measure whether a fitted imputer beats predicting the mean. It does not."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split

    inf = influencers.reset_index(drop=True)
    brd = brands.reset_index(drop=True)
    inf_text = [influencer_profile_text(r) for _, r in inf.iterrows()]
    brand_text = [brand_profile_text(r) for _, r in brd.iterrows()]
    vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2), sublinear_tf=True)
    vec.fit(inf_text + brand_text)
    tf = np.asarray((vec.transform(inf_text) @ vec.transform(brand_text).T).todense())

    iidx = {i: k for k, i in enumerate(inf.influencer_id)}
    bidx = {b: j for j, b in enumerate(brd.brand_id)}
    lab = scored[scored.brand_id.isin(bidx) & scored.influencer_id.isin(iidx)].merge(
        inf[["influencer_id", "primary_niche", "secondary_niche", "followers"]],
        on="influencer_id").merge(brd[["brand_id", "category"]], on="brand_id")
    if len(lab) < 60:
        return {"skipped": "not enough labelled pairs for these brands"}

    lab["tfidf_cos"] = [tf[iidx[i], bidx[b]] for i, b in zip(lab.influencer_id, lab.brand_id)]
    lab["cm"] = [_category_match(r, c) for (_, r), c in zip(lab.iterrows(), lab.category)]
    X = np.column_stack([lab.tfidf_cos, lab.tfidf_cos ** 2, lab.cm,
                         np.log10(lab.followers.clip(lower=1))])
    y = lab.semantic_cosine.to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    rmse = float(np.sqrt(np.mean((Ridge(alpha=1.0).fit(Xtr, ytr).predict(Xte) - yte) ** 2)))
    base = float(np.sqrt(np.mean((ytr.mean() - yte) ** 2)))
    return {
        "fitted_rmse": round(rmse, 5),
        "predict_the_mean_rmse": round(base, 5),
        "improvement": round(base - rmse, 5),
        "verdict": "no improvement over the mean; rejected in favour of a per-brand bound",
        "cause": "labelled pairs are the top-60 per brand by cosine, so the sample is "
                 "range-restricted at the high end",
    }


# --------------------------------------------------------------------------
# The bound that is actually used
# --------------------------------------------------------------------------

def build_full_fit(influencers: pd.DataFrame, brands: pd.DataFrame,
                   scored: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Score every (creator, brand) pair for the brands passed in."""
    inf = influencers.reset_index(drop=True)
    brd = brands.reset_index(drop=True)

    known = {(r.influencer_id, r.brand_id): r.semantic_cosine for r in scored.itertuples()}
    # Per-brand floor: the lowest cosine SBERT returned for that brand. Anyone
    # it did not return sits at or below this by construction.
    floors = scored.groupby("brand_id").semantic_cosine.min().to_dict()
    global_floor = float(scored.semantic_cosine.min())

    rows = []
    for _, brand in brd.iterrows():
        floor = float(floors.get(brand.brand_id, global_floor))
        for _, infrow in inf.iterrows():
            real = known.get((infrow.influencer_id, brand.brand_id))
            cos = float(real) if real is not None else floor
            rec = score_pair(infrow, brand, cos)
            rec.update({
                "brand_id": brand.brand_id,
                "influencer_id": infrow.influencer_id,
                "semantic_cosine": round(cos, 4),
                "semantic_imputed": real is None,
            })
            rows.append(rec)

    out = pd.DataFrame(rows)
    stats = {
        "method": "per-brand cosine floor for pairs SBERT did not score",
        "n_pairs": int(len(out)),
        "n_sbert_scored": int((~out.semantic_imputed).sum()),
        "n_bounded": int(out.semantic_imputed.sum()),
        "semantic_weight": COMPONENT_WEIGHTS["semantic_similarity"],
        "exact_component_weight": round(
            1 - COMPONENT_WEIGHTS["semantic_similarity"], 3),
        "rejected_alternative": evaluate_regression_alternative(inf, brd, scored),
    }
    return out, stats
