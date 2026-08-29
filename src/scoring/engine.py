"""
The three-tier scoring engine, with a reason attached to every number.

    Creator Quality     how strong is this creator, independent of any brand
    Organisation Fit    how well do this creator and this brand suit each other
    Campaign Fit        is this creator right for this brief, right now

The separation is the whole methodology. Creator Quality is a property of the
creator and does not move when a different brand looks at them. Organisation
Fit is a relationship and changes per brand. Campaign Fit is contextual and
changes per brief - a creator can be an excellent long-term match for a brand
and wrong for the campaign running this month because they are booked, or
because the brief needs Carousels they do not make.

Hard gates block; they do not deduct
------------------------------------
A creator who cannot legally or operationally take the work is returned as
BLOCKED with a reason, not as a low percentage. "Campaign Fit 34%" invites a
brand to scroll past; "BLOCKED - competitor conflict" tells them why the
creator is not there at all.

Every weight below is an argued starting point, not a learned one. What the
behavioural log says about them is in src/reco/ranker.py, which found that
learned weights barely beat these on brands the model has not seen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Weights. Exposed so the app can render them rather than restate them.
# --------------------------------------------------------------------------
CREATOR_QUALITY_WEIGHTS = {
    "engagement_quality": 0.22,   # how hard the audience works, not how big
    "audience_quality": 0.20,     # is the audience real (audience_quality model)
    "comment_quality": 0.16,      # what the comment section looks like
    "reach_strength": 0.14,       # size, deliberately not the largest term
    "content_consistency": 0.12,  # predictable topic range
    "reliability": 0.10,          # posts regularly, grows steadily
    "visual_coherence": 0.06,     # the feed looks like one feed
}

ORG_FIT_WEIGHTS = {
    "category_affinity": 0.24,
    "semantic_similarity": 0.20,
    "audience_alignment": 0.16,
    "brand_safety": 0.16,
    "creator_quality": 0.14,
    "visual_similarity": 0.10,
}

CAMPAIGN_FIT_WEIGHTS = {
    "organisation_fit": 0.28,
    "audience_match": 0.18,
    "semantic_match": 0.16,
    "budget_fit": 0.14,
    "deliverable_fit": 0.12,
    "availability_fit": 0.12,
}

BANDS = [(80, "Excellent"), (65, "Strong"), (50, "Moderate"), (0, "Weak")]


def band_of(score_0_100: float) -> str:
    for edge, name in BANDS:
        if score_0_100 >= edge:
            return name
    return "Weak"


def _pct(s: pd.Series) -> pd.Series:
    return s.rank(pct=True).fillna(0.5)


# ==========================================================================
# 1. Creator Quality
# ==========================================================================
def creator_quality(creators: pd.DataFrame, aq: pd.DataFrame, comm: pd.DataFrame,
                    vis: pd.DataFrame, insights: pd.DataFrame) -> pd.DataFrame:
    c = creators.copy()
    c["influencer_id"] = c.influencer_id.astype(str)
    for other in (aq, comm, vis, insights):
        o = other.copy(); o["influencer_id"] = o.influencer_id.astype(str)
        c = c.merge(o, on="influencer_id", how="left", suffixes=("", "_dup"))

    # Engagement quality blends what everyone can see with what only a
    # connected creator can prove. Saves and shares are considered acts; likes
    # are not, so they carry more weight when they are available at all.
    er = _pct(c.er_vs_benchmark.fillna(c.engagement_rate))
    ctl = _pct(c.comments_to_likes)
    save = _pct(c.get("save_rate", pd.Series(np.nan, index=c.index)))
    share = _pct(c.get("share_rate", pd.Series(np.nan, index=c.index)))
    has_insights = c.get("n_posts_with_insights", pd.Series(np.nan, index=c.index)).notna()
    eng_public = 0.65 * er + 0.35 * ctl
    eng_verified = 0.40 * er + 0.20 * ctl + 0.22 * save + 0.18 * share
    c["q_engagement_quality"] = np.where(has_insights, eng_verified, eng_public)

    c["q_audience_quality"] = (c.get("audience_quality_score",
                                     pd.Series(50.0, index=c.index)).fillna(50) / 100)
    c["q_comment_quality"] = c.get("comment_quality_index",
                                   pd.Series(0.5, index=c.index)).fillna(0.5)
    c["q_reach_strength"] = _pct(np.log1p(c.followers))
    c["q_content_consistency"] = np.clip(
        1.0 - c.content_topic_entropy.fillna(1.5) / 3.0, 0, 1)

    growth_stability = 1.0 - _pct((c.follower_growth_rate.fillna(0) - 0.02).abs())
    posting = _pct(c.posting_frequency_month.fillna(0))
    c["q_reliability"] = (0.55 * posting + 0.45 * growth_stability).clip(0, 1)
    c["q_visual_coherence"] = c.get("visual_coherence",
                                    pd.Series(0.5, index=c.index)).fillna(0.5)

    total = sum(CREATOR_QUALITY_WEIGHTS[k] * c[f"q_{k}"]
                for k in CREATOR_QUALITY_WEIGHTS)
    c["creator_quality"] = (100 * total.clip(0, 1)).round(1)
    c["creator_quality_band"] = c.creator_quality.apply(band_of)
    c["verified_metrics"] = has_insights.fillna(False)

    cols = (["influencer_id", "creator_quality", "creator_quality_band",
             "verified_metrics"]
            + [f"q_{k}" for k in CREATOR_QUALITY_WEIGHTS])
    out = c[cols].copy()
    for k in CREATOR_QUALITY_WEIGHTS:
        out[f"q_{k}"] = out[f"q_{k}"].astype(float).round(4)
    out["creator_quality_reasons"] = [
        " · ".join(_quality_reasons(r)) for r in out.itertuples()]
    return out


_QUALITY_LABEL = {
    "engagement_quality": ("Audience engages far above their tier",
                           "Engagement is below the benchmark for their size"),
    "audience_quality": ("Audience looks authentic",
                         "Audience shows signs of inauthentic activity"),
    "comment_quality": ("Comment section is real conversation",
                        "Comment section is thin or automated"),
    "reach_strength": ("Large reach", "Small audience"),
    "content_consistency": ("Posts on a narrow, predictable set of topics",
                            "Topic range is scattered"),
    "reliability": ("Posts regularly and grows steadily",
                    "Posting is irregular"),
    "visual_coherence": ("Feed has a consistent look", "Feed looks inconsistent"),
}


def _quality_reasons(row, up: float = 0.72, down: float = 0.34) -> list[str]:
    strengths, weaknesses = [], []
    for k in CREATOR_QUALITY_WEIGHTS:
        v = float(getattr(row, f"q_{k}", 0.5))
        if v >= up:
            strengths.append(_QUALITY_LABEL[k][0])
        elif v <= down:
            weaknesses.append(_QUALITY_LABEL[k][1])
    return (strengths[:3] + weaknesses[:2]) or ["Middling on every component"]


# ==========================================================================
# 2. Organisation Fit
# ==========================================================================
def organisation_fit(pairs: pd.DataFrame, quality: pd.DataFrame,
                     creators: pd.DataFrame, comm: pd.DataFrame,
                     aq: pd.DataFrame, visual_sim: pd.Series | None = None
                     ) -> pd.DataFrame:
    """`pairs` must carry brand_id, influencer_id and the existing semantic and
    category components from the brand-fit matrix."""
    d = pairs.copy()
    d["influencer_id"] = d.influencer_id.astype(str)
    d = (d.merge(quality[["influencer_id", "creator_quality"]], on="influencer_id", how="left")
          .merge(comm[["influencer_id", "comment_toxicity_rate"]], on="influencer_id", how="left")
          .merge(aq[["influencer_id", "audience_quality_score"]], on="influencer_id", how="left"))

    d["o_category_affinity"] = d.get("fit_category_match", 0.5)
    d["o_semantic_similarity"] = d.get("fit_semantic_similarity", 0.5)
    d["o_audience_alignment"] = d.get("fit_audience_match", 0.5)
    d["o_visual_similarity"] = (visual_sim.values if visual_sim is not None
                                else 0.5)
    d["o_creator_quality"] = d.creator_quality.fillna(50) / 100

    # Brand safety is now three things, not one: what the creator writes, what
    # their audience writes back, and whether that audience is real.
    d["o_brand_safety"] = np.clip(
        0.50 * d.get("fit_content_safety", 0.8)
        + 0.25 * (1 - d.comment_toxicity_rate.fillna(0.1))
        + 0.25 * (d.audience_quality_score.fillna(50) / 100), 0, 1)

    total = sum(ORG_FIT_WEIGHTS[k] * d[f"o_{k}"] for k in ORG_FIT_WEIGHTS)
    gate = d.get("gate_multiplier", pd.Series(1.0, index=d.index)).fillna(1.0)
    d["org_fit_ungated"] = (100 * total.clip(0, 1)).round(1)
    d["org_fit"] = (d.org_fit_ungated * gate).round(1)
    d["org_fit_band"] = d.org_fit.apply(band_of)
    d["org_blocked"] = gate.eq(0)
    d["org_fit_reasons"] = [" · ".join(_org_reasons(r)) for r in d.itertuples()]
    return d


_ORG_LABEL = {
    "category_affinity": ("Works in exactly this category",
                          "Different content category"),
    "semantic_similarity": ("Content language matches the brand's",
                            "Talks about different things than the brand"),
    "audience_alignment": ("Audience matches the brand's target",
                           "Audience does not match the brand's target"),
    "brand_safety": ("Clean, brand-safe profile", "Brand-safety concerns"),
    "creator_quality": ("Strong creator on their own merits",
                        "Weak on core creator quality"),
    "visual_similarity": ("Feed looks like the brand's world",
                          "Visual style is off-brand"),
}


def _org_reasons(row, up: float = 0.75, down: float = 0.35) -> list[str]:
    if bool(getattr(row, "org_blocked", False)):
        return [str(getattr(row, "gate_reasons", "") or "Blocked")]
    s, w = [], []
    for k in ORG_FIT_WEIGHTS:
        v = float(getattr(row, f"o_{k}", 0.5))
        if v >= up:
            s.append(_ORG_LABEL[k][0])
        elif v <= down:
            w.append(_ORG_LABEL[k][1])
    return (s[:3] + w[:2]) or ["No component stands out either way"]


# ==========================================================================
# 3. Campaign Fit
# ==========================================================================
_CAMPAIGN_LABEL = {
    "audience_match": ("Audience matches the brief's target",
                       "Audience is not who the brief asks for"),
    "semantic_match": ("Content is close to what the brief describes",
                       "Content is far from the brief"),
    "budget_fit": ("Comfortably inside the per-creator budget",
                   "Priced near the top of the budget"),
    "deliverable_fit": ("Strong in the formats the brief needs",
                        "Weak in the formats the brief needs"),
    "availability_fit": ("Free for the campaign window",
                         "Limited availability in the window"),
    "organisation_fit": ("Strong long-term match with this brand",
                         "Weak long-term match with this brand"),
}


def competitor_conflict(campaign, mentions: pd.DataFrame,
                        window_days: int = 180, min_repeat: int = 2) -> set:
    """Creators the brief cannot legally use.

    The batch fit matrix could not apply this gate at all: it reads a
    `competitor_activity` column that lives in the modelling table and is not
    exported, so `gate_multiplier` was 1.0 or a soft penalty for every one of
    the 12,000 pairs and nothing was ever blocked. The evidence it needs IS
    exported - nectar_brand_mentions - so the veto is applied here instead,
    with the same rule the typed-brief matcher uses.
    """
    names = {c.strip().lower()
             for c in str(getattr(campaign, "competitor_brands", "") or "").split("|")
             if c.strip()}
    if not names or mentions is None or mentions.empty:
        return set()
    m = mentions[mentions.brand.isin(names)
                 & (mentions.days_ago_min <= window_days)]
    hard = m[(m.n_paid > 0) | (m.n_mentions >= min_repeat)]
    return set(hard.influencer_id.astype(str))


def campaign_fit(campaign, creators: pd.DataFrame, org: pd.DataFrame,
                 capability: pd.DataFrame, cap_module,
                 conflicted: set | None = None) -> pd.DataFrame:
    """Score every creator against ONE campaign.

    Hard gates are collected first and, where any fires, the composite is not
    reported at all. A number next to a creator a brand cannot book is worse
    than no number: it implies a choice that does not exist.
    """
    c = creators.copy(); c["influencer_id"] = c.influencer_id.astype(str)
    d = c.merge(org, on="influencer_id", how="left", suffixes=("", "_org"))
    d = d.merge(capability, on="influencer_id", how="left", suffixes=("", "_cap"))

    n_by_type = {}
    for item in (campaign.deliverables if campaign.deliverables is not None else []):
        t = str(item.get("type", "")) if isinstance(item, dict) else str(item)
        n_by_type[t] = n_by_type.get(t, 0) + int(item.get("qty", 1)
                                                 if isinstance(item, dict) else 1)
    fee = sum(d.get(f"rate_{t.lower()}", 0).fillna(0) * q
              for t, q in n_by_type.items()) if n_by_type else pd.Series(0.0, index=d.index)
    d["brief_fee"] = fee.round(0)

    cap = float(campaign.max_per_creator or 0) or float("inf")
    d["c_budget_fit"] = np.clip(1.0 - (d.brief_fee / cap) * 0.85, 0.0, 1.0)

    geo_ok = d.audience_geo.eq(campaign.target_geo)
    age_ok = d.audience_age_band.eq(campaign.target_age_band)
    d["c_audience_match"] = (0.55 * np.where(geo_ok, 1.0, 0.42)
                             + 0.45 * np.where(age_ok, 1.0, 0.52))
    d["c_semantic_match"] = d.get("o_semantic_similarity", 0.5)
    d["c_organisation_fit"] = d.org_fit.fillna(50) / 100

    deliv, avail, blocked, gate_reasons = [], [], [], []
    for r in d.itertuples():
        ds, dr, dblock = cap_module.deliverable_fit(r, campaign.deliverables)
        as_, ar, ablock = cap_module.availability_fit(
            r, campaign.start_date, campaign.end_date)
        reasons = []
        block = False
        if conflicted and str(r.influencer_id) in conflicted:
            block = True
            reasons.append("Blocked: recent paid or repeated work with a competitor")
        if bool(getattr(r, "org_blocked", False)):
            block = True
            reasons.append(str(getattr(r, "gate_reasons", "") or
                               "Competitor conflict"))
        if dblock:
            block = True; reasons += dr
        if ablock:
            block = True; reasons += ar
        if float(r.followers) < float(campaign.min_followers or 0):
            block = True
            reasons.append(f"Below the brief's {int(campaign.min_followers):,} "
                           f"follower floor")
        if float(getattr(r, "brief_fee", 0)) > cap:
            block = True
            reasons.append("Brief price is above the per-creator cap")
        deliv.append(ds); avail.append(as_); blocked.append(block)
        gate_reasons.append(" · ".join(reasons))

    d["c_deliverable_fit"] = deliv
    d["c_availability_fit"] = avail
    d["blocked"] = blocked
    d["block_reasons"] = gate_reasons

    total = sum(CAMPAIGN_FIT_WEIGHTS[k] * d[f"c_{k}"] for k in CAMPAIGN_FIT_WEIGHTS)
    d["campaign_fit"] = np.where(d.blocked, np.nan,
                                 (100 * total.clip(0, 1)).round(1))
    d["campaign_fit_band"] = [band_of(x) if pd.notna(x) else "Blocked"
                              for x in d.campaign_fit]
    d["campaign_id"] = campaign.campaign_id
    d["campaign_fit_reasons"] = [" · ".join(_campaign_reasons(r))
                                 for r in d.itertuples()]

    keep = (["campaign_id", "influencer_id", "campaign_fit", "campaign_fit_band",
             "blocked", "block_reasons", "campaign_fit_reasons", "brief_fee",
             "org_fit", "org_fit_band", "org_fit_reasons"]
            + [f"c_{k}" for k in CAMPAIGN_FIT_WEIGHTS]
            + [f"o_{k}" for k in ORG_FIT_WEIGHTS])
    out = d[[k for k in keep if k in d.columns]].copy()
    # Percentile within the eligible pool, because the raw composite's spread
    # is only a few points wide and a brand reading 71 vs 69 would infer a gap
    # that is not there.
    elig = ~out.blocked
    out["campaign_fit_pct"] = np.nan
    out.loc[elig, "campaign_fit_pct"] = (
        out.loc[elig, "campaign_fit"].rank(pct=True) * 100).round(1)
    return out


def _campaign_reasons(row, up: float = 0.78, down: float = 0.40) -> list[str]:
    if bool(getattr(row, "blocked", False)):
        return [str(getattr(row, "block_reasons", "") or "Not eligible")]
    s, w = [], []
    for k in CAMPAIGN_FIT_WEIGHTS:
        v = float(getattr(row, f"c_{k}", 0.5))
        if v >= up:
            s.append(_CAMPAIGN_LABEL[k][0])
        elif v <= down:
            w.append(_CAMPAIGN_LABEL[k][1])
    return (s[:3] + w[:2]) or ["Solid but unremarkable on every component"]
