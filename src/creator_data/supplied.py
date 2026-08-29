"""
Owner-only Instagram metrics, supplied by creators who connect their account.

The product fact this package exists to encode
----------------------------------------------
Saves, shares, average watch time, profile visits and audience demographics are
returned by the Instagram API only to the authenticated account owner. A
platform cannot scrape them; it has to be given them. So Nectar asks the
creator to connect, and in exchange the creator gets access to campaigns.

That has three consequences the product must handle honestly, and does:

  * Not every creator connects. Unconnected creators simply have no private
    metrics, and the app says so rather than inventing a number.
  * A connected creator's numbers are verified; an unconnected creator's are
    inferred from public data. Brands should be able to tell which is which.
  * Scores that depend on private metrics have to degrade gracefully when they
    are absent, or the product punishes creators for not having signed up yet.

SIMULATION NOTE
---------------
No Instagram account has been connected to anything. These figures are
generated from the same latent creator traits that drive the rest of the
synthetic universe, so they are internally consistent with likes, comments and
reach rather than being independent noise.

What makes them worth having rather than decorative: saves and shares are
driven mainly by CONTENT QUALITY, while likes are driven mainly by REACH. That
is the real distinction - a like is cheap and a save is a considered act - and
it means the private metrics carry information the public ones do not. If they
were generated from the same driver as likes they would be a second copy of a
column the model already has.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from src.config import SEED

# Share of creators who have connected their account. Real creator-marketing
# platforms report connection rates well below 100%; the product has to work
# for the rest.
CONNECTED_SHARE = 0.62

# Scopes a creator grants at connection. Named after the real ones so the
# onboarding screen is not fiction about how the permission model works.
SCOPES = [
    "instagram_business_basic",
    "instagram_business_manage_insights",
    "instagram_business_manage_comments",
]

AGE_BUCKETS = ["13-17", "18-24", "25-34", "35-44", "45-54", "55+"]
GENDERS = ["female", "male", "other"]


def _unit(key: str, salt: str) -> float:
    h = hashlib.sha256(f"{salt}:{key}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


def connection_status(influencer_ids) -> pd.DataFrame:
    """Who has connected, when, and with which scopes.

    Deterministic per creator, so a creator does not flicker between connected
    and not between rebuilds.
    """
    rows = []
    for iid in [str(i) for i in influencer_ids]:
        u = _unit(iid, "connect")
        connected = u < CONNECTED_SHARE
        rows.append({
            "influencer_id": iid,
            "account_connected": connected,
            # Days since the creator connected. Drives "insights available
            # since" copy and, later, how much history the platform holds.
            "connected_days_ago": int(14 + 320 * _unit(iid, "since")) if connected else None,
            "scopes_granted": "|".join(SCOPES) if connected else "",
            "insights_available": connected,
        })
    return pd.DataFrame(rows)


def post_private_metrics(posts: pd.DataFrame, latents: pd.DataFrame,
                         connected: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Per-post owner-only metrics, for connected creators only.

    Every quantity is anchored to a public one so the table cannot contradict
    what the brand can already see: shares never exceed likes, saves scale with
    reach, watch time cannot exceed the video's length.
    """
    rng = np.random.default_rng(seed + 2207)

    p = posts.copy()
    p["influencer_id"] = p["influencer_id"].astype(str)
    lat = latents.copy()
    lat["influencer_id"] = lat["influencer_id"].astype(str)
    p = p.merge(lat[["influencer_id", "content_quality", "authenticity", "niche_focus"]],
                on="influencer_id", how="left")
    p = p.merge(connected[["influencer_id", "account_connected"]],
                on="influencer_id", how="left")
    p = p[p.account_connected.fillna(False)].reset_index(drop=True)
    n = len(p)

    likes = p.likes.to_numpy(dtype=float)
    comments = p.comments.to_numpy(dtype=float)
    views = p.views.fillna(0).to_numpy(dtype=float)
    quality = p.content_quality.fillna(0.5).to_numpy()
    focus = p.niche_focus.fillna(0.6).to_numpy()

    # ---- saves ----------------------------------------------------------
    # A save is a bet that the post will be useful later, so it tracks content
    # quality far more than reach. Typical save rates sit near 1-4% of likes
    # for lifestyle content and higher for how-to and education.
    save_rate = np.clip(0.012 + 0.085 * quality + 0.02 * focus
                        + rng.normal(0, 0.012, n), 0.001, 0.35)
    p["saves"] = np.round(likes * save_rate).astype(int)

    # ---- shares ---------------------------------------------------------
    # Sharing is a social act - it puts the sharer's taste on the line - so it
    # is rarer than saving and even more quality-driven.
    share_rate = np.clip(0.006 + 0.055 * quality + rng.normal(0, 0.008, n),
                         0.0005, 0.25)
    p["shares"] = np.round(likes * share_rate).astype(int)

    # ---- watch time -----------------------------------------------------
    # Only video posts have it. Length is drawn first so watch time can be
    # bounded by it; a dwell figure longer than the clip is the classic
    # giveaway of a fabricated metrics table.
    # Which posts are video is decided in ONE place - src/audio/simulate.py -
    # so a post that has a voice track in the audio pipeline is the same post
    # that has watch time here. Deriving it separately (views > 0) made every
    # post a video and left dwell_seconds an exact duplicate of watch time.
    from src.audio.simulate import is_video as _is_video
    is_video = _is_video(p.post_id)
    length_s = np.where(is_video, np.clip(rng.gamma(4.0, 6.0, n), 7, 90), 0.0)
    through = np.clip(0.22 + 0.55 * quality + rng.normal(0, 0.09, n), 0.05, 0.99)
    p["video_length_s"] = np.round(length_s, 1)
    p["watch_through_rate"] = np.where(is_video, np.round(through, 4), np.nan)
    p["avg_watch_time_s"] = np.where(is_video, np.round(length_s * through, 2), np.nan)

    # Dwell on a still post: seconds spent before scrolling on. Much shorter,
    # and the honest ceiling is low - most stills get under ten seconds.
    dwell_still = np.clip(1.4 + 7.5 * quality + rng.normal(0, 1.1, n), 0.5, 25.0)
    p["dwell_seconds"] = np.round(
        np.where(is_video, length_s * through, dwell_still), 2)

    # ---- downstream actions --------------------------------------------
    # Profile visits and follows are what a brand actually wants: attention
    # that moved somewhere. Both are driven by quality and by how coherent the
    # creator's niche is, because a viewer follows a theme, not a post.
    visit_rate = np.clip(0.02 + 0.09 * quality + 0.04 * focus
                         + rng.normal(0, 0.015, n), 0.002, 0.4)
    p["profile_visits"] = np.round(np.maximum(likes, 1) * visit_rate).astype(int)
    follow_rate = np.clip(0.03 + 0.14 * quality + rng.normal(0, 0.02, n), 0.002, 0.5)
    p["follows_from_post"] = np.round(p.profile_visits * follow_rate).astype(int)

    # Reach and impressions: owner-only, and the denominator every rate the
    # brand sees should really have been computed against.
    p["reach"] = np.round(np.maximum(likes, 1)
                          / np.clip(0.03 + 0.10 * quality, 0.01, 0.5)).astype(int)
    p["impressions"] = np.round(p.reach * (1.0 + rng.gamma(1.6, 0.22, n))).astype(int)

    p["engagements_total"] = (likes + comments + p.saves + p.shares).astype(int)
    p["save_rate"] = np.round(p.saves / np.maximum(p.reach, 1), 5)
    p["share_rate"] = np.round(p.shares / np.maximum(p.reach, 1), 5)

    cols = ["post_id", "influencer_id", "saves", "shares", "reach", "impressions",
            "video_length_s", "avg_watch_time_s", "watch_through_rate",
            "dwell_seconds", "profile_visits", "follows_from_post",
            "engagements_total", "save_rate", "share_rate"]
    return p[cols]


def audience_demographics(influencer_ids, influencers: pd.DataFrame,
                          connected: pd.DataFrame,
                          seed: int = SEED) -> pd.DataFrame:
    """The audience breakdown a creator can see and a brand cannot.

    Built around the coarse band the public feature table already carries, so
    the detailed split agrees with it instead of contradicting it.
    """
    rng = np.random.default_rng(seed + 3301)
    inf = influencers.set_index("influencer_id")
    rows = []
    for iid in [str(i) for i in influencer_ids]:
        if not bool(connected.set_index("influencer_id").at[iid, "account_connected"]):
            continue
        band = str(inf.at[iid, "audience_age_band"]) if iid in inf.index else "25-34"
        centre = AGE_BUCKETS.index(band) if band in AGE_BUCKETS else 2
        # Mass decays with distance from the creator's dominant band, so the
        # detailed distribution and the coarse label cannot disagree.
        w = np.array([0.58 ** abs(i - centre) for i in range(len(AGE_BUCKETS))])
        w = w * (1 + rng.normal(0, 0.10, len(w)))
        w = np.clip(w, 0.001, None); w = w / w.sum()

        skew = float(inf.at[iid, "audience_gender_skew"]) if iid in inf.index else 0.5
        other = float(np.clip(rng.beta(1.4, 60), 0.0, 0.06))
        # `other` is taken out of the total FIRST, then the remainder is split.
        # Clipping female to 0.97 independently let female + other exceed 1,
        # male clamp to zero, and the three shares sum to 1.03.
        female = float(np.clip(skew + rng.normal(0, 0.05), 0.02, 0.98)) * (1.0 - other)
        male = max(0.0, 1.0 - female - other)

        rows.append({
            "influencer_id": iid,
            **{f"audience_age_{b.replace('-', '_').replace('+', 'plus')}":
               round(float(x), 4) for b, x in zip(AGE_BUCKETS, w)},
            "audience_female_pct": round(female, 4),
            "audience_male_pct": round(male, 4),
            "audience_other_pct": round(other, 4),
            "audience_top_country_pct": round(
                float(np.clip(0.55 + 0.35 * _unit(iid, "geo"), 0.4, 0.97)), 4),
            "audience_language_match_pct": round(
                float(np.clip(0.5 + 0.45 * _unit(iid, "lang"), 0.35, 0.99)), 4),
        })
    return pd.DataFrame(rows)
