"""
Brand-Fit scoring.

Why this is NOT a learned model, deliberately
---------------------------------------------
Every other scoring surface in this system is learned. Brand-Fit is not, and
that is an argued choice rather than a shortcut:

1. There is no label for it. "Did this creator fit this brand?" is not recorded
   anywhere. Training on campaign engagement would just re-learn the performance
   model and call it fit, which is circular.
2. Half of fit is a hard constraint, not a preference. If a creator has
   promoted a direct competitor, no similarity score should be able to
   outweigh that - it is a veto. Learned models blend; brand safety needs gates.
3. It is the number a brand manager will argue with. A composite they can
   decompose into "semantic 0.71, category match yes, geo match no" is
   defensible in a meeting. A gradient-boosted score is not.

Structure:

    eligibility gates  ->  hard pass/fail (competitor conflict, geo, age)
    similarity score   ->  SBERT cosine between creator content and brand profile
    category signals   ->  niche match, category consistency
    safety adjustments ->  ad saturation, negative-tone penalty, vocal delivery

    fit = gate * weighted_sum(components)

Every component is exposed in the dashboard so the score can be taken apart.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import ARTIFACT_DIR, PROCESSED_DIR, SBERT_MODEL

BRANDFIT_DIR = ARTIFACT_DIR / "brandfit"
BRANDFIT_DIR.mkdir(parents=True, exist_ok=True)

COMPONENT_WEIGHTS = {
    "semantic_similarity": 0.34,
    "category_match": 0.28,
    "audience_match": 0.18,
    "content_safety": 0.12,
    "consistency": 0.08,
}


# ==========================================================================
# Text profiles
# ==========================================================================


def influencer_profile_text(row: pd.Series) -> str:
    """One short document summarising what a creator is about."""
    parts = [
        f"{row.get('primary_niche', '')} content creator",
        f"also covers {row.get('secondary_niche', '')}" if row.get("secondary_niche") else "",
        str(row.get("top_keywords", "") or "").replace("|", ", "),
        str(row.get("top_hashtags", "") or "").replace("|", ", "),
    ]
    return ". ".join(p for p in parts if p).strip()


def brand_profile_text(row: pd.Series) -> str:
    parts = [
        f"{row.get('category', '')} brand",
        str(row.get("brand_keywords", "") or "").replace("|", ", "),
        f"targeting {row.get('target_age_band', '')} audience in {row.get('target_geo', '')}",
    ]
    return ". ".join(p for p in parts if p).strip()


# ==========================================================================
# Scoring
# ==========================================================================


# A creator is only in conflict if the competitor relationship looks like an
# actual commercial one. The first version of this gate blocked on ANY mention
# ever, and blocked 85% of all pairs - because a passing mention of a category's
# best-known brand is near-universal within that category. That is not brand
# safety, it is a broken filter, and it made the matching page useless.
#
# Real exclusivity clauses cover recent PAID collaborations. So the gate now
# requires one of:
#   * a disclosed paid post featuring the competitor, within the window, or
#   * repeated mentions (>= MIN_REPEAT) within the window
# A single unpaid mention eight months ago is downgraded to a soft warning.
CONFLICT_WINDOW_DAYS = 180
CONFLICT_MIN_REPEAT = 2


def _parse_conflict_field(raw) -> dict[str, dict]:
    """`brand:count:paid|brand:count:paid` -> {brand: {count, paid}}."""
    out: dict[str, dict] = {}
    for part in str(raw or "").split("|"):
        if not part.strip():
            continue
        bits = part.split(":")
        name = bits[0].strip().lower()
        if not name:
            continue
        out[name] = {
            "count": int(bits[1]) if len(bits) > 1 and bits[1].isdigit() else 1,
            "paid": bool(int(bits[2])) if len(bits) > 2 and bits[2].isdigit() else False,
        }
    return out


def eligibility(inf: pd.Series, brand: pd.Series) -> tuple[float, list[str]]:
    """Hard gates. Returns (multiplier, list of reasons)."""
    reasons: list[str] = []
    mult = 1.0

    competitors = {c.strip().lower() for c in str(brand.get("competitor_brands", "") or "").split("|") if c.strip()}
    recent = _parse_conflict_field(inf.get("competitor_activity", ""))

    hard, soft = [], []
    for name in sorted(competitors & set(recent)):
        info = recent[name]
        if info["paid"] or info["count"] >= CONFLICT_MIN_REPEAT:
            kind = "paid partnership" if info["paid"] else f"{info['count']} posts"
            hard.append(f"{name} ({kind})")
        else:
            soft.append(name)

    if hard:
        mult = 0.0
        reasons.append(
            f"BLOCKED: recent commercial relationship with competitor(s) {', '.join(hard)}"
        )
    elif soft:
        mult *= 0.80
        reasons.append(
            f"mentioned competitor(s) {', '.join(soft)} once, unpaid — worth checking, not disqualifying"
        )

    # Ad saturation is a soft penalty, not a veto.
    promo = float(inf.get("content_promo_rate", 0) or 0)
    if promo > 0.55:
        mult *= 0.65
        reasons.append(f"heavily ad-saturated feed ({promo:.0%} of posts promotional)")
    elif promo > 0.38:
        mult *= 0.85
        reasons.append(f"elevated ad load ({promo:.0%} of posts promotional)")

    return mult, reasons


def score_pair(
    inf: pd.Series,
    brand: pd.Series,
    semantic: float,
) -> dict:
    """Score one influencer-brand pair. `semantic` is a precomputed cosine."""
    category_match = (
        1.0 if inf.get("primary_niche") == brand.get("category")
        else 0.55 if inf.get("secondary_niche") == brand.get("category")
        else 0.15
    )
    geo = 1.0 if inf.get("audience_geo") == brand.get("target_geo") else 0.45
    age = 1.0 if inf.get("audience_age_band") == brand.get("target_age_band") else 0.55
    audience_match = 0.55 * geo + 0.45 * age

    # Safety: penalise consistently negative tone and heavy irony, both of which
    # make branded content read badly.
    #
    # Two of the four terms are read from the voice track. A brand reads the
    # caption; an audience watches the Reel, and a creator whose captions are
    # cheerful can deliver a flat or contemptuous voice-over. `tone_mismatch`
    # is the sign disagreement between the caption model and the audio model -
    # the signal a single-modality pipeline cannot produce at all.
    #
    # THE AUDIO TERMS ARE SIMULATED. src/nlp/audio_sim.py generates them; no
    # waveform exists in this project. They are weighted well below the text
    # terms for that reason, and both default to zero so the function still
    # scores correctly against a feature table built before audio existed.
    neg = float(inf.get("content_share_negative", 0) or 0)
    irony = float(inf.get("content_irony_rate", 0) or 0)
    audio_neg = float(inf.get("audio_share_negative", 0) or 0)
    mismatch = float(inf.get("tone_mismatch_rate", 0) or 0)
    content_safety = float(np.clip(
        1.0 - 0.8 * neg - 0.5 * irony - 0.30 * audio_neg - 0.20 * mismatch,
        0.0, 1.0))

    # Consistency: low topic entropy = focused creator = predictable for a brand.
    ent = inf.get("content_topic_entropy", np.nan)
    consistency = float(np.clip(1.0 - (ent / 3.0), 0.0, 1.0)) if pd.notna(ent) else 0.5

    components = {
        "semantic_similarity": float(np.clip((semantic + 1) / 2, 0, 1)),
        "category_match": category_match,
        "audience_match": audience_match,
        "content_safety": content_safety,
        "consistency": consistency,
    }
    base = sum(COMPONENT_WEIGHTS[k] * v for k, v in components.items())
    gate, reasons = eligibility(inf, brand)

    return {
        "brand_fit": round(float(base * gate), 4),
        "brand_fit_ungated": round(float(base), 4),
        "gate_multiplier": round(float(gate), 3),
        "gate_reasons": "; ".join(reasons),
        **{f"fit_{k}": round(float(v), 4) for k, v in components.items()},
    }


# ==========================================================================
# Batch build
# ==========================================================================


def build_matrix(
    influencers: pd.DataFrame,
    brands: pd.DataFrame,
    top_k: int = 60,
    model_name: str = SBERT_MODEL,
) -> pd.DataFrame:
    """Score every brand against its top-k influencers by semantic similarity.

    Scoring all 2,000 x 120 pairs is cheap here, but the k-limit is what keeps
    this O(brands x k) as the creator base grows, and it is what the deployed
    dashboard reads.
    """
    from src.nlp.embeddings import cosine_matrix, embed

    inf_text = [influencer_profile_text(r) for _, r in influencers.iterrows()]
    brand_text = [brand_profile_text(r) for _, r in brands.iterrows()]

    print(f"    embedding {len(inf_text):,} creator profiles and {len(brand_text)} brand profiles")
    inf_emb = embed(inf_text, model_name, show_progress=False)
    brand_emb = embed(brand_text, model_name, show_progress=False)
    sim = cosine_matrix(brand_emb, inf_emb)          # (n_brands, n_influencers)

    rows = []
    for bi, (_, brand) in enumerate(brands.iterrows()):
        order = np.argsort(-sim[bi])[:top_k]
        for ii in order:
            inf = influencers.iloc[ii]
            rec = score_pair(inf, brand, float(sim[bi, ii]))
            rec.update(
                {
                    "brand_id": brand["brand_id"],
                    "influencer_id": inf["influencer_id"],
                    "semantic_cosine": round(float(sim[bi, ii]), 4),
                }
            )
            rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_parquet(BRANDFIT_DIR / "brand_fit_matrix.parquet", index=False)
    (BRANDFIT_DIR / "brandfit_config.json").write_text(
        json.dumps(
            {
                "component_weights": COMPONENT_WEIGHTS,
                "top_k_per_brand": top_k,
                "encoder": model_name,
                "gates": [
                    f"recent PAID or repeated (>={CONFLICT_MIN_REPEAT}) competitor collaboration "
                    f"within {CONFLICT_WINDOW_DAYS} days -> hard block (multiplier 0)",
                    "single unpaid competitor mention in window -> x0.80 (warning, not a veto)",
                    "promo rate > 55% -> x0.65",
                    "promo rate > 38% -> x0.85",
                ],
                "conflict_window_days": CONFLICT_WINDOW_DAYS,
                "conflict_min_repeat": CONFLICT_MIN_REPEAT,
            },
            indent=2,
        )
    )
    return out


def run(top_k: int = 60) -> pd.DataFrame:
    from src.features.build_features import FEATURE_DIR
    from src.nlp.pipeline import NLP_DIR

    influencers = pd.read_parquet(FEATURE_DIR / "influencer_features.parquet")
    brands = pd.read_parquet(PROCESSED_DIR / "brands.parquet")

    # Build the competitor-conflict input: for each creator, which brands they
    # mentioned RECENTLY, how often, and whether any of those posts was a
    # disclosed paid partnership. Recency and payment are what make a mention a
    # commercial conflict rather than a passing reference.
    pf = NLP_DIR / "post_features.parquet"
    raw_posts_path = PROCESSED_DIR / "posts.parquet"
    influencers["competitor_activity"] = ""

    if pf.exists() and raw_posts_path.exists():
        pfeat = pd.read_parquet(pf)
        if "brands_mentioned" in pfeat.columns:
            recency = pd.read_parquet(raw_posts_path)[["post_id", "days_ago"]]
            df = pfeat.merge(recency, on="post_id", how="left")
            df = df[df["days_ago"].fillna(9999) <= CONFLICT_WINDOW_DAYS]

            paid_col = "has_disclosure" if "has_disclosure" in df.columns else "has_promo"
            df = df.assign(b=df["brands_mentioned"].fillna("").str.split("|")).explode("b")
            df = df[df["b"].astype(str).str.strip() != ""]
            df["b"] = df["b"].str.strip().str.lower()

            if len(df):
                agg = (
                    df.groupby(["influencer_id", "b"])
                    .agg(count=("post_id", "count"), paid=(paid_col, "max"))
                    .reset_index()
                )
                agg["tok"] = (
                    agg["b"] + ":" + agg["count"].astype(str) + ":"
                    + agg["paid"].fillna(0).astype(int).astype(str)
                )
                packed = (
                    agg.groupby("influencer_id")["tok"]
                    .apply(lambda s: "|".join(sorted(s)))
                    .rename("competitor_activity_new")
                )
                influencers = influencers.merge(packed, on="influencer_id", how="left")
                influencers["competitor_activity"] = (
                    influencers.pop("competitor_activity_new").fillna("")
                )
                print(f"    competitor activity: {len(agg):,} creator-brand relationships "
                      f"within {CONFLICT_WINDOW_DAYS} days "
                      f"({int(agg['paid'].fillna(0).sum()):,} paid)")

    print("  building brand-fit matrix ...")
    m = build_matrix(influencers, brands, top_k=top_k)
    blocked = int((m["gate_multiplier"] == 0).sum())
    print(f"    {len(m):,} scored pairs, {blocked:,} blocked by competitor conflict")
    print(f"    mean fit {m['brand_fit'].mean():.3f}, "
          f"top-decile threshold {m['brand_fit'].quantile(0.9):.3f}")
    return m


if __name__ == "__main__":
    run()
