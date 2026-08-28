"""
Creator presentation layer: the model's feature table -> Nectar's creator card.

Nothing in this file feeds the model. It runs strictly downstream of scoring
and turns numbers the model produced into the fields the product shows: a
person, their platforms, their audience mix and their rate card.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.nectar import names as N

# Follower share by platform. Instagram is the anchor for every creator in this
# universe; the others are conditional on signals we actually measured.
YT_MIN_VIEW_RATIO = 0.45          # views_to_followers above this implies video
SHORTFORM_TIERS = {"Nano", "Micro"}
IN_GEOS = {"IN-North", "IN-South", "IN-West", "IN-East"}

AGE_BANDS = ["13-17", "18-24", "25-34", "35-44", "45+"]
# Audience age decays away from the creator's dominant band. An earlier version
# rolled a fixed weight vector, which wrapped: a 13-17 creator was handed 22% of
# their audience in the 45+ bucket. Distance-based decay cannot wrap.
AGE_DECAY = 0.62


def _age_distribution(band: str) -> list[dict]:
    if band not in AGE_BANDS:
        band = "25-34"
    centre = AGE_BANDS.index(band)
    w = np.array([AGE_DECAY ** abs(i - centre) for i in range(len(AGE_BANDS))])
    w = w / w.sum()
    out = [{"range": b, "pct": int(round(p * 100))} for b, p in zip(AGE_BANDS, w)]
    # Force the percentages to sum to exactly 100 - a card that reads 99% looks
    # like a bug even when the underlying numbers are fine.
    out[centre]["pct"] += 100 - sum(o["pct"] for o in out)
    return [o for o in out if o["pct"] > 0]


def _gender_split(skew: float) -> list[dict]:
    female = int(round(float(skew) * 100))
    female = min(max(female, 3), 97)
    return [{"label": "Female", "pct": female}, {"label": "Male", "pct": 100 - female}]


def _audience_locations(influencer_id: str, geo: str) -> list[dict]:
    pool = N.CITIES.get(geo, N.CITIES["IN-West"])
    idx = N._h(influencer_id, "aud") % len(pool)
    ordered = pool[idx:] + pool[:idx]
    shares = [38, 24, 16, 12]
    out = [{"city": c, "pct": p} for c, p in zip(ordered[:4], shares)]
    out.append({"city": "Other", "pct": 100 - sum(shares)})
    return out


def _platforms(row) -> dict:
    """Split the audience across platforms using measured behaviour.

    view-through rate decides whether this creator is a video creator, and
    Indian short-form platforms (Moj, Josh) only appear for India-facing
    small and mid creators - which is where they actually have reach.
    """
    f = int(row.followers)
    out = {"Instagram": f}
    if row.views_to_followers >= YT_MIN_VIEW_RATIO:
        share = 0.28 + (N._h(row.influencer_id, "yt") % 25) / 100
        out["YouTube"] = int(f * share)
    if row.audience_geo in IN_GEOS and row.follower_tier in SHORTFORM_TIERS:
        if N._h(row.influencer_id, "moj") % 100 < 55:
            out["Moj"] = int(f * (0.10 + (N._h(row.influencer_id, "mojs") % 20) / 100))
        if N._h(row.influencer_id, "josh") % 100 < 35:
            out["Josh"] = int(f * (0.08 + (N._h(row.influencer_id, "joshs") % 15) / 100))
    if N._h(row.influencer_id, "snap") % 100 < 18:
        out["Snapchat"] = int(f * 0.12)
    return {k: v for k, v in out.items() if v > 0}


def _availability(influencer_id: str) -> tuple[str, str]:
    r = N._h(influencer_id, "avail") % 100
    if r < 62:
        status = "Available"
    elif r < 86:
        status = "Busy"
    else:
        status = "Unavailable"
    starts = ["1", "5", "8", "10", "12", "15", "18", "22"]
    s = int(N.pick(influencer_id, "avstart", starts))
    length = 6 + (N._h(influencer_id, "avlen") % 18)
    end = s + length
    window = f"{s}–{end} Sept" if end <= 30 else f"{s} Sept–{end - 30} Oct"
    if status != "Available":
        window = "Booked through Sept"
    return status, window


def build(inf: pd.DataFrame) -> pd.DataFrame:
    """inf = app_data/influencers.parquet (the scored feature table)."""
    df = inf.copy()

    df["name"] = [
        N.display_name(i, g) for i, g in zip(df.influencer_id, df.audience_gender_skew)
    ]
    # Handles must be unique: two creators can legitimately draw the same name.
    base = df["name"].str.lower().str.replace(" ", "", regex=False)
    dupe_rank = base.groupby(base).cumcount()
    df["nectar_handle"] = np.where(dupe_rank == 0, "@" + base,
                                   "@" + base + dupe_rank.astype(str))
    df["handle_original"] = df["handle"]
    df["initials"] = df["name"].map(N.initials)
    df["avatar_color"] = df["influencer_id"].map(N.avatar)
    df["city"] = [N.city(i, g) for i, g in zip(df.influencer_id, df.audience_geo)]
    df["bio"] = df["primary_niche"].map(N.bio)

    df["categories"] = [
        [c for c in (p, s) if isinstance(c, str) and c]
        for p, s in zip(df.primary_niche, df.secondary_niche)
    ]
    df["platforms"] = [ _platforms(r) for r in df.itertuples() ]
    df["platform_names"] = df["platforms"].map(lambda d: list(d.keys()))

    df["audience_age"] = df["audience_age_band"].map(_age_distribution)
    df["audience_gender"] = df["audience_gender_skew"].map(_gender_split)
    df["audience_locations"] = [
        _audience_locations(i, g) for i, g in zip(df.influencer_id, df.audience_geo)
    ]

    avail = [_availability(i) for i in df.influencer_id]
    df["availability"] = [a[0] for a in avail]
    df["available_window"] = [a[1] for a in avail]

    # Rate card. The price model predicts a per-deliverable fee; Story and
    # Carousel are priced off it using the ratios the Indian market actually
    # uses (a Story is worth far less than a Reel because it expires).
    fee = df["price_estimate_inr"].astype(float)
    df["rate_reel"] = fee.round(-2)
    df["rate_story"] = (fee * 0.35).round(-2)
    df["rate_carousel"] = (fee * 0.70).round(-2)
    df["rate_from"] = df[["rate_reel", "rate_story", "rate_carousel"]].min(axis=1)

    # Verified: a platform badge is granted on scale and consistency, not on
    # performance. Reproduced here as reach + a low-volatility posting habit.
    df["verified"] = (
        (df["followers"] >= 25_000)
        & (df["posting_frequency_month"] >= 6)
        & (df["content_promo_rate"].fillna(0) < 0.55)
    )
    return df
