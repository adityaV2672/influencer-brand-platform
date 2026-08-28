"""
Campaigns, and the per-campaign fit the Discover page ranks on.

Campaign Fit and Organisation Fit are NOT decorative numbers. They come from
the brand-fit matrix the pipeline already computes (semantic similarity,
category affinity, audience overlap, content safety, consistency) and from the
LightGBM performance model. The reason strings under "Why this creator?" are
generated from the creator's actual percentile position on the features the
model weights most heavily - so a claim on a card can always be traced back to
a number in the feature table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Six campaigns: enough to populate every state the UI has to render (live,
# draft, completed) without inventing a brand universe that does not exist.
# Campaign budgets are set here rather than taken from the brand row, because
# `budget_inr` on the brand table is that brand's *annual* creator-marketing
# budget; a single campaign draws a slice of it.
#
# The figures below are deliberately larger than the ones in the design mock.
# The fee model is calibrated to published Indian creator rates, where a
# 150K-follower Reel is tens of thousands of rupees, not four. At mock-scale
# budgets the only affordable creators were nano accounts under 5K followers,
# every campaign spent 14% of its budget, and total reach came out in the tens
# of thousands. Real budgets buy the mid-tier creators the reporting page is
# supposed to be about.
#
# The six budgets also encode six different strategies, which is what makes the
# Reporting comparison worth looking at: Summer Edit buys macro reach, while
# Weeknight Kitchen buys micro efficiency. Cost per engagement should — and
# does — come out lower for the second.
CAMPAIGN_PLAN = [
    # (category, name, status, objective, deliverables, budget, per-creator cap, min audience)
    #
    # The per-creator cap is a POLICY the brand sets on the brief, not budget/N.
    # Keeping it independent is what lets a campaign run a small budget against
    # mid-tier creators (Weeknight Kitchen) or a large budget against macro ones
    # (Summer Edit) without the two numbers fighting each other.
    #
    # The minimum audience is the other half of that policy. Without it, a brief
    # with a 2 lakh per-creator cap happily signed 800-follower accounts purely
    # because they scored well on fit - a shortlist no brand would accept. The
    # floors are set near the lower quartile of each brief's candidate pool:
    # high enough to exclude accounts that cannot deliver, low enough to leave
    # a shortlist worth ranking. They are genuinely low because this creator
    # population is genuinely nano- and micro-heavy, which is what the Indian
    # market actually looks like.
    ("Beauty",     "Monsoon Skin Reset",  "Live",      "Awareness",     [("Reel", 2), ("Story", 3)],    750_000, 160_000,  3_000),
    ("Fitness",    "Everyday Energy",     "Live",      "Consideration", [("Reel", 2), ("Carousel", 1)], 420_000,  90_000,  1_500),
    ("Fashion",    "Summer Edit",         "Live",      "Awareness",     [("Reel", 1), ("Carousel", 2)], 900_000, 210_000,  8_000),
    ("Technology", "Tech Unboxed Q3",     "Draft",     "Consideration", [("Reel", 1), ("Story", 2)],    450_000, 120_000,  4_000),
    ("Food",       "Weeknight Kitchen",   "Completed", "Conversion",    [("Reel", 2), ("Story", 2)],    460_000,  60_000,  2_000),
    ("Travel",     "Offbeat Monsoon",     "Completed", "Awareness",     [("Reel", 1), ("Story", 3)],    830_000, 140_000,  3_000),
]

OBJECTIVE_BLURB = {
    "Awareness": "Maximise qualified reach in the target segment.",
    "Consideration": "Drive profile visits and saved posts.",
    "Conversion": "Drive code redemptions on the landing page.",
}


def build_campaigns(brands: pd.DataFrame, seed: int = 20260904) -> pd.DataFrame:
    """Anchor each campaign to a real brand from the generated brand table."""
    rng = np.random.default_rng(seed)
    rows = []
    used = set()
    for i, (cat, name, status, objective, mix, budget, cap, min_aud) in enumerate(CAMPAIGN_PLAN):
        pool = brands[(brands.category == cat) & (~brands.brand_id.isin(used))]
        if pool.empty:
            pool = brands[~brands.brand_id.isin(used)]
        # Best-funded brand in the category: the one most likely to be running
        # a campaign of this size.
        brand = pool.sort_values("budget_inr", ascending=False).iloc[0]
        used.add(brand.brand_id)
        budget = float(budget)
        # `progress` is a placeholder here. export_nectar overwrites it with
        # committed spend / budget once the request pipeline exists, so the
        # Campaigns table and the Reporting page cannot disagree.
        progress = 0.0 if status == "Draft" else 1.0
        rows.append({
            "campaign_id": f"CMP{i:02d}",
            "brand_id": brand.brand_id,
            "brand_name": brand.brand_name,
            "name": name,
            "category": cat,
            "status": status,
            "objective": objective,
            "description": OBJECTIVE_BLURB[objective],
            "budget_inr": budget,
            "annual_brand_budget_inr": float(brand.budget_inr),
            "spent_inr": 0.0,
            "progress": round(progress, 3),
            "start_date": "2026-08-01" if status != "Draft" else "2026-09-15",
            "end_date": "2026-09-30" if status != "Completed" else "2026-07-31",
            "target_geo": brand.target_geo,
            "target_age_band": brand.target_age_band,
            "brand_keywords": brand.brand_keywords,
            "competitor_brands": brand.competitor_brands,
            "deliverables": [{"type": t, "qty": q, "platform": "Instagram"} for t, q in mix],
            "deliverable_label": ", ".join(f"{q} {t}" for t, q in mix),
            "max_per_creator": float(cap),
            "min_followers": int(min_aud),
            "payment_type": "Paid",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Fit
# --------------------------------------------------------------------------

def _pct(series: pd.Series) -> pd.Series:
    return series.rank(pct=True) * 100


def build_fit(fit_matrix: pd.DataFrame, creators: pd.DataFrame,
              campaigns: pd.DataFrame) -> pd.DataFrame:
    """One row per (campaign, creator) for the 60 candidates scored per brand.

    fit_composite - the brand-fit composite before safety gating, 0-100. This
                    is the number that decomposes into the five named
                    components, and the one to quote when explaining a score.
    campaign_fit  - that composite expressed as a PERCENTILE within this
                    campaign's candidate pool, which is what the card shows.
                    The composite's own range is narrow (an audit found an
                    inter-quartile range of five points), so displaying it raw
                    implied a precision it does not have. Gating is surfaced
                    separately as safety flags, because a brand needs to see
                    *why* a creator was screened out, not just that they
                    vanished.
    org_fit       - how well the creator suits the organisation rather than
                    this brief: content safety, posting consistency, and how
                    close their audience sits to the brand's target segment.
    """
    f = fit_matrix.merge(
        campaigns[["campaign_id", "brand_id", "name", "budget_inr", "max_per_creator",
                   "min_followers", "deliverables", "deliverable_label"]],
        on="brand_id", how="inner",
    )
    cols = [
        "influencer_id", "name", "nectar_handle", "initials", "avatar_color", "city",
        "bio", "categories", "platform_names", "followers", "follower_tier",
        "engagement_rate", "er_vs_benchmark", "avg_views", "follower_growth_rate",
        "comments_to_likes", "content_promo_rate", "content_irony_rate",
        "pagerank_pct", "primary_niche", "verified", "availability",
        "available_window", "rate_reel", "rate_story", "rate_carousel", "rate_from",
        "price_estimate_inr", "score_rate", "score_reach", "score_balanced",
        "predicted_campaign_er", "audience_age", "audience_gender",
        "audience_locations", "platforms", "posting_frequency_month",
    ]
    cols = [c for c in cols if c in creators.columns]
    f = f.merge(creators[cols], on="influencer_id", how="inner",
                suffixes=("_campaign", ""))

    # The raw composite is displayed as a 0-100 score, and an audit showed the
    # implied precision is not there: across 12,000 pairs the inter-quartile
    # range was 54-59, because the category term carries 28% of the weight with
    # only three possible values and the semantic term is a per-brand constant
    # for most rows. "86% fit" against "84% fit" was noise.
    #
    # So the card now shows the creator's PERCENTILE within this campaign's
    # candidate pool. That is the number a brand can act on - "better than 86%
    # of the creators eligible for this brief" - and it uses the full 0-100
    # range by construction. The raw composite is kept alongside it, because it
    # is what decomposes into the five named components on the drawer.
    f["fit_composite"] = (f["brand_fit_ungated"] * 100).round(1)
    # One decimal, not zero: with a pool of 2,000 the top twenty all round to
    # 100 and three cards in a row read "100th", which looks like a bug.
    f["campaign_fit"] = (
        f.groupby("campaign_id")["brand_fit_ungated"].rank(pct=True) * 100
    ).round(1)
    f["org_composite"] = (
        (0.45 * f["fit_content_safety"] + 0.35 * f["fit_consistency"]
         + 0.20 * f["fit_audience_match"]) * 100
    ).round(1)
    f["org_fit"] = (f["org_composite"].rank(pct=True) * 100).round(1)

    # Percentile cuts: the top fifth is a high fit for this brief, the next
    # third medium. Absolute thresholds on the raw composite put almost every
    # creator in one band, which told a brand nothing.
    f["fit_band"] = pd.cut(f["campaign_fit"], bins=[0, 50, 80, 101],
                           labels=["Low fit", "Medium fit", "High fit"],
                           right=False).astype(str)
    f["org_band"] = pd.cut(f["org_fit"], bins=[0, 50, 80, 101],
                           labels=["Low fit", "Medium fit", "High fit"],
                           right=False).astype(str)

    f["blocked"] = f["gate_multiplier"] == 0
    f["ad_saturated"] = (f["gate_multiplier"] > 0) & (f["gate_multiplier"] < 1)
    f["safety_flags"] = f["gate_reasons"].fillna("").map(
        lambda s: [x.strip() for x in str(s).split(";") if x.strip()]
    )
    # Price the WHOLE brief, not one deliverable. The fee model predicts a
    # per-deliverable rate; a brief asking for 2 Reels and 3 Stories costs
    # 2*reel + 3*story. Comparing a single-deliverable rate against a
    # whole-brief cap was silently letting creators through at ~3x the budget.
    unit = {"Reel": "rate_reel", "Story": "rate_story", "Carousel": "rate_carousel"}
    brief = np.zeros(len(f))
    for kind, col in unit.items():
        qty = f["deliverables"].map(
            lambda ds, k=kind: sum(d["qty"] for d in ds if d["type"] == k))
        brief = brief + f[col].to_numpy() * qty.to_numpy()
    f["brief_fee_inr"] = np.round(brief, -2)
    f["within_budget"] = f["brief_fee_inr"] <= f["max_per_creator"]
    f["meets_audience_floor"] = f["followers"] >= f["min_followers"]
    f["eligible"] = (~f["blocked"]) & f["within_budget"] & f["meets_audience_floor"]

    # Budget fit: how comfortably the brief price sits under the per-creator
    # cap. At half the cap this is a clean 100; at the cap it is 60 (workable,
    # not comfortable); at twice the cap it is 0.
    ratio = f["brief_fee_inr"] / f["max_per_creator"].replace(0, np.nan)
    f["fit_budget"] = np.clip(120 - 60 * ratio, 0, 100).fillna(50).round(0)

    # Fit breakdown, straight from the composite's own components.
    f["fit_audience"] = (f["fit_audience_match"] * 100).round(0)
    f["fit_category"] = (f["fit_category_match"] * 100).round(0)
    f["fit_deliverable"] = (f["fit_consistency"] * 100).round(0)
    # Rank within campaign under each objective the UI offers.
    #
    # "Best match" breaks ties on the model's predicted performance. With the
    # full creator base scored, hundreds of creators share an identical fit
    # composite (same niche, same geo, same age band), and ranking on fit alone
    # ordered them arbitrarily by row position. Fit decides who is a candidate;
    # the model decides who is first.
    # Rank on the underlying composite, not the displayed percentile - the
    # percentile is rounded to whole numbers and would create 100 giant ties.
    f["_best"] = f["fit_composite"] * 1000 + f["score_balanced"]
    # Rank inside the ELIGIBLE pool. Discover only shows creators this brief can
    # afford and that clear its audience floor, so "#4 of 2,000" was counting a
    # denominator the page never displays. Ineligible rows are ranked after all
    # eligible ones so the ordering still holds when the gates are switched off.
    for key, col in [("best", "_best"), ("engagement", "engagement_rate"),
                     ("reach", "followers")]:
        within = f[col].where(f["eligible"])
        r_ok = f.assign(_v=within).groupby("campaign_id")["_v"].rank(
            ascending=False, method="first")
        n_ok = f.groupby("campaign_id")["eligible"].transform("sum")
        r_no = f.assign(_v=f[col].where(~f["eligible"])).groupby(
            "campaign_id")["_v"].rank(ascending=False, method="first")
        f[f"rank_{key}"] = np.where(f["eligible"], r_ok, n_ok + r_no).astype(int)
    f = f.drop(columns=["_best"])

    # The size of the pool each rank is out of, so a card can say "#7 of 1,252"
    # rather than leaving the reader to guess what 99th percentile means here.
    f["pool_size"] = f.groupby("campaign_id")["influencer_id"].transform("size")
    f["eligible_pool_size"] = f.groupby("campaign_id")["eligible"].transform("sum")

    f = f.sort_values(["campaign_id", "rank_best"]).reset_index(drop=True)
    return f


# --------------------------------------------------------------------------
# "Why this creator?" - reasons traceable to the feature table
# --------------------------------------------------------------------------
# Every rule below reads a column the model actually uses and states the
# creator's position on it. Nothing here is a stock phrase attached at random:
# if a card says "engagement is 1.8x the published benchmark", er_vs_benchmark
# for that creator is 1.8.

def _reason_rules(r) -> tuple[list[str], list[str]]:
    helped, held = [], []

    if r.er_vs_benchmark >= 1.25:
        helped.append(f"Engagement is {r.er_vs_benchmark:.1f}× the published benchmark for this size")
    elif r.er_vs_benchmark < 0.85:
        held.append(f"Engagement is {r.er_vs_benchmark:.1f}× the benchmark for this size")

    if r.fit_category_match >= 0.99:
        helped.append(f"{r.primary_niche} is exactly the brief's category")
    elif r.fit_category_match < 0.5:
        held.append(f"{r.primary_niche} is adjacent to, not inside, the brief's category")

    if r.fit_audience_match >= 0.9:
        helped.append("Audience geography and age both match the target segment")
    elif r.fit_audience_match < 0.55:
        held.append("Audience sits outside the target geography or age band")

    if r.comments_to_likes >= 0.02:
        helped.append("Comment-to-like ratio in the top band — engagement is conversational, not passive")

    if r.follower_growth_rate >= 0.02:
        helped.append(f"Growing {r.follower_growth_rate * 100:+.1f}% a month — audience is still compounding")
    elif r.follower_growth_rate <= -0.005:
        held.append(f"Audience is shrinking {r.follower_growth_rate * 100:.1f}% a month")

    promo = float(r.content_promo_rate or 0)
    if promo <= 0.20:
        helped.append(f"Low ad load — only {promo:.0%} of recent posts are promotional")
    elif promo >= 0.45:
        held.append(f"Ad-saturated feed — {promo:.0%} of recent posts are promotional")

    if r.pagerank_pct >= 80:
        helped.append("Central in the creator graph for this niche — content travels")

    irony = float(r.content_irony_rate or 0)
    if irony >= 0.25:
        held.append(f"Sarcasm detected in {irony:.0%} of captions — check tone against the brief")

    if not helped:
        helped.append("Clears every screening threshold without standing out on any one signal")
    return helped[:4], held[:3]


def add_reasons(fit: pd.DataFrame) -> pd.DataFrame:
    """Attach the reason strings. Kept separate from build_fit so the rules can
    be unit-tested against the columns they claim to read."""
    pairs = [_reason_rules(r) for r in fit.itertuples()]
    fit = fit.copy()
    fit["what_helped"] = [p[0] for p in pairs]
    fit["what_held_back"] = [p[1] for p in pairs]
    # The card shows two lines; the drawer shows all of them.
    fit["match_reasons"] = fit["what_helped"].map(lambda x: x[:2])
    return fit


# --------------------------------------------------------------------------
# Are the composite weights load-bearing?
# --------------------------------------------------------------------------
# The five brand-fit weights (34/28/18/12/8) are asserted, not learned - there
# is no label for "was this a good fit", so there is nothing to fit them to.
# That is defensible, but only if the answer does not swing on the exact
# numbers. This measures it: perturb every weight, renormalise, rebuild the
# composite, and see how much of the top-20 shortlist survives.
#
# A shortlist that is stable under perturbation means the arbitrariness does not
# matter in practice. A shortlist that is not means the weights are doing the
# ranking and should be defended far more carefully.

WEIGHT_PERTURBATIONS = [0.5, 0.75, 1.25, 1.5]


def weight_sensitivity(fit: pd.DataFrame, weights: dict[str, float],
                       top_n: int = 20, seed: int = 20260904) -> pd.DataFrame:
    """Top-N shortlist overlap when each weight is scaled up or down."""
    comp_cols = {
        "semantic_similarity": "fit_semantic_similarity",
        "category_match": "fit_category_match",
        "audience_match": "fit_audience_match",
        "content_safety": "fit_content_safety",
        "consistency": "fit_consistency",
    }
    comp_cols = {k: v for k, v in comp_cols.items() if v in fit.columns}
    eligible = fit[fit["eligible"]] if "eligible" in fit.columns else fit

    def shortlists(w: dict[str, float]) -> dict[str, set]:
        total = sum(w.values())
        score = sum((w[k] / total) * eligible[c] for k, c in comp_cols.items())
        tmp = eligible.assign(_s=score)
        return {cid: set(g.nlargest(top_n, "_s").influencer_id)
                for cid, g in tmp.groupby("campaign_id")}

    baseline = shortlists(weights)
    rows = []
    for name in comp_cols:
        for mult in WEIGHT_PERTURBATIONS:
            w = dict(weights)
            w[name] = weights[name] * mult
            got = shortlists(w)
            overlaps = [len(baseline[c] & got[c]) / top_n for c in baseline]
            rows.append({
                "component": name,
                "multiplier": mult,
                "weight_from": round(weights[name], 3),
                "weight_to": round(w[name] / sum(w.values()), 3),
                "mean_overlap": round(float(np.mean(overlaps)), 3),
                "min_overlap": round(float(np.min(overlaps)), 3),
                "top_n": top_n,
            })
    return pd.DataFrame(rows)
