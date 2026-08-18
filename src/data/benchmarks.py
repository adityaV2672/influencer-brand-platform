"""
Published industry benchmarks used to calibrate the synthetic universe and to
seed the Phase-1 rule-based price model.

IMPORTANT - provenance and honesty
----------------------------------
These figures come from industry/marketing publications, NOT peer-reviewed
research. Platforms do not publish engagement or rate-card data, and academic
figures are usually years stale, so industry aggregators are the only available
source. They should be read as order-of-magnitude anchors, not ground truth.

Every number below is traceable to the source recorded in `SOURCES`. Nothing
here is invented. If a value was interpolated rather than quoted, it is marked
`assumed=True` and justified.

Retrieved: 2026-08-17.
"""
from __future__ import annotations

SOURCES = {
    "engagement": {
        "publisher": "Nowadays Media",
        "title": "Engagement Rates by Follower Tier (2026): TikTok, Instagram & YouTube Benchmarks",
        "url": "https://nowadays.media/blog/influencer-engagement-rate-benchmarks-2026-by-platform-niche-follower-tier/",
        "basis": "Stated as an analysis of 15,000+ creator accounts",
        "retrieved": "2026-08-17",
        "caveat": "Industry publication, methodology not independently auditable.",
    },
    "pricing": {
        "publisher": "upGrowth",
        "title": "Influencer Pricing India 2026: Rates by Follower Tier",
        "url": "https://upgrowth.in/influencer-marketing-pricing-india-2026/",
        "basis": "Published INR rate bands per deliverable, Indian market",
        "retrieved": "2026-08-17",
        "caveat": "Industry publication; rates vary widely by usage rights and exclusivity.",
    },
}

# --------------------------------------------------------------------------
# Follower tiers - aligned to the tier definitions used by both sources above
# so the benchmarks can be applied without re-bucketing.
# (lower_inclusive, upper_exclusive, label)
# --------------------------------------------------------------------------
TIERS = [
    (0,          10_000,      "Nano"),
    (10_000,     100_000,     "Micro"),
    (100_000,    500_000,     "Mid"),
    (500_000,    2_000_000,   "Macro"),
    (2_000_000,  10**12,      "Mega"),
]

# Quoted engagement-rate bands (static feed posts). Values are fractions.
ENGAGEMENT_BANDS = {
    "Nano":  (0.035, 0.060),
    "Micro": (0.015, 0.035),
    "Mid":   (0.010, 0.025),
    "Macro": (0.008, 0.015),
    "Mega":  (0.005, 0.010),
}

# Quoted INR price bands per deliverable (Instagram post/reel, India).
PRICE_BANDS_INR = {
    "Nano":  (2_000,     8_000),
    "Micro": (8_000,     80_000),
    "Mid":   (50_000,    350_000),
    "Macro": (200_000,   1_200_000),
    "Mega":  (800_000,   10_000_000),
}

# Quoted average engagement rate by niche. Overall Instagram average quoted as
# 1.2% static / 3.8% Reels; the niche figures below are blended averages.
NICHE_ENGAGEMENT_AVG = {
    "Education":    0.042,
    "Beauty":       0.038,
    "Food":         0.035,
    "Fitness":      0.032,
    "Fashion":      0.028,
    "Travel":       0.025,
    "Technology":   0.021,
    "Gaming":       0.021,
}

# Niches present in our taxonomy but not itemised by the source. Interpolated
# from the "General Lifestyle: 1.8%" anchor and adjacent categories.
# These are ASSUMPTIONS, flagged as such in the report.
NICHE_ENGAGEMENT_ASSUMED = {
    "Finance":      0.019,   # low-frequency, high-consideration, near tech
    "Parenting":    0.030,   # community-heavy, near fitness/food
    "Automotive":   0.020,   # near tech
    "Home & Decor": 0.026,   # near travel/fashion
}

NICHE_ENGAGEMENT = {**NICHE_ENGAGEMENT_AVG, **NICHE_ENGAGEMENT_ASSUMED}
NICHE_ASSUMED_FLAG = {k: (k in NICHE_ENGAGEMENT_ASSUMED) for k in NICHE_ENGAGEMENT}

# Multiplier relative to the cross-niche mean, used by the generator and by the
# rule-based price model.
_MEAN_NICHE_ER = sum(NICHE_ENGAGEMENT.values()) / len(NICHE_ENGAGEMENT)
NICHE_ER_MULTIPLIER = {k: v / _MEAN_NICHE_ER for k, v in NICHE_ENGAGEMENT.items()}

# --------------------------------------------------------------------------
# Fitted curves
# --------------------------------------------------------------------------
# Power law fitted (OLS in log-log) to the geometric midpoints of the quoted
# engagement bands. See notebooks/00_calibration.ipynb for the fit.
#   ER(followers) = ER_A * (followers / 10_000) ** ER_B
ER_A = 0.0339
ER_B = -0.251

# Power law fitted to the geometric midpoints of the quoted INR price bands.
#   fee(followers) = FEE_A * followers ** FEE_B
FEE_A = 2.39
FEE_B = 0.90


def tier_of(followers: int) -> str:
    for lo, hi, label in TIERS:
        if lo <= followers < hi:
            return label
    return "Mega"


def expected_er(followers: float, niche: str | None = None) -> float:
    """Benchmark engagement rate for a follower count, optionally niche-adjusted."""
    er = ER_A * (followers / 10_000.0) ** ER_B
    if niche:
        er *= NICHE_ER_MULTIPLIER.get(niche, 1.0)
    return float(er)


def expected_fee(followers: float, niche: str | None = None) -> float:
    """Benchmark INR fee for a follower count, optionally niche-adjusted."""
    fee = FEE_A * followers ** FEE_B
    if niche:
        # Price follows engagement demand, but less than proportionally.
        fee *= NICHE_ER_MULTIPLIER.get(niche, 1.0) ** 0.5
    return float(fee)
