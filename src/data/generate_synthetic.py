"""
Synthetic influencer universe generator.

Design notes (these matter for the report)
------------------------------------------
This is a *simulation study*, not a scrape. Every influencer is generated from a
small set of latent traits that the model is never shown:

    content_quality   - how good the content actually is
    authenticity      - what fraction of the audience is real
    consistency       - how reliably they post
    promo_saturation  - how ad-heavy the feed already is

Observable features (followers, likes, comments, growth, captions, network
position) are *noisy functions* of those latents. The supervised target is the
engagement rate achieved on **sponsored campaign posts**, which is a separate
noisy draw from the same latents.

That structure matters for three reasons:

1. There is no target leakage. Campaign performance is never a deterministic
   function of any observable feature, so the model has to genuinely learn.
2. There is a known ceiling. Because we control the noise, we know the maximum
   achievable R^2, and can report how close the model gets to it.
3. It mirrors the real problem stated in the proposal (Phase 2: "once outcome
   data exists, e.g. actual campaign engagement lift").

Known limitation, stated openly in the report: synthetic captions cannot
validate NLP method quality. That is done separately against real, human-
labelled corpora in src/benchmark/.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import (
    N_BRANDS,
    N_INFLUENCERS,
    NICHES,
    POSTS_PER_INFLUENCER,
    AMPLIFICATION_OBSERVED_SHARE,
    PROCESSED_DIR,
    SEED,
    tier_of,
)
from src.data import benchmarks as bm
from src.data.lexicon import (
    CTA_PHRASES,
    GENERIC_HASHTAGS,
    NEGATIVE_OPENERS,
    NEUTRAL_OPENERS,
    NICHE_LEXICON,
    POSITIVE_OPENERS,
    PROMO_PHRASES,
    QUESTION_PHRASES,
    SARCASM_ABSURD,
    SARCASM_FAILS,
    SARCASM_GREAT,
    SARCASM_TEMPLATES,
)

# ==========================================================================
# Generative constants
# ==========================================================================

# Sponsored posts consistently underperform the same creator's organic content.
# The exact discount is not published by any source we could verify, so 0.75 is
# an ASSUMPTION. It is declared in the report and is a single tunable constant.
SPONSORED_DISCOUNT = 0.75

# Irreducible per-campaign noise. Because we set it, we know the theoretical
# ceiling on model R^2 and can report how close the model gets to it.
CAMPAIGN_NOISE_SIGMA = 0.32


# ==========================================================================
# Latent traits
# ==========================================================================


@dataclass
class Latents:
    """Hidden generative traits. Never exposed as model features."""

    content_quality: float
    authenticity: float
    consistency: float
    promo_saturation: float
    niche_focus: float  # 1.0 = single-niche purist, 0.0 = scattered


def _draw_latents(rng: np.random.Generator, n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "content_quality": rng.beta(2.2, 2.2, n),
            "authenticity": rng.beta(5.0, 1.8, n),
            "consistency": rng.beta(3.0, 2.0, n),
            "promo_saturation": rng.beta(1.6, 4.0, n),
            "niche_focus": rng.beta(3.5, 1.8, n),
        }
    )


# ==========================================================================
# Profile / reach / engagement
# ==========================================================================


def _generate_profiles(rng: np.random.Generator, lat: pd.DataFrame) -> pd.DataFrame:
    n = len(lat)

    # Followers: heavy-tailed. Better creators skew larger, but weakly - plenty
    # of large accounts are mediocre, which is the whole premise of the product.
    base_log = rng.normal(9.6, 1.72, n)                    # ~e^9.6 = 15k median
    quality_pull = (lat["content_quality"].to_numpy() - 0.5) * 1.1
    followers = np.exp(base_log + quality_pull).astype(int).clip(800, 25_000_000)

    niches = rng.choice(NICHES, n)

    # Engagement rate decays with account size. The base curve is the power law
    # fitted to published 2026 tier benchmarks (see src/data/benchmarks.py), so
    # the synthetic universe reproduces real-world tier medians rather than
    # arbitrary numbers.
    size_decay = bm.ER_A * (followers / 10_000.0) ** bm.ER_B
    niche_mult = np.array([bm.NICHE_ER_MULTIPLIER.get(x, 1.0) for x in niches])

    quality_mult = (
        (0.45 + 1.20 * lat["content_quality"].to_numpy())
        * (0.55 + 0.75 * lat["authenticity"].to_numpy())
        * (0.80 + 0.35 * lat["consistency"].to_numpy())
        * rng.lognormal(0.0, 0.28, n)
    )
    # Mean-normalise so the multipliers redistribute engagement between
    # creators without shifting the population off the benchmark curve.
    quality_mult = quality_mult / quality_mult.mean()

    er = (size_decay * niche_mult * quality_mult).clip(0.0008, 0.18)

    total_eng = er * followers
    # Comments-to-likes ratio is a quality signal: real conversation is harder to buy.
    ctl = (0.010 + 0.075 * lat["content_quality"].to_numpy() * lat["authenticity"].to_numpy()) * rng.lognormal(0, 0.30, n)
    ctl = ctl.clip(0.002, 0.35)

    avg_likes = (total_eng / (1.0 + ctl)).round().astype(int)
    avg_comments = (total_eng - avg_likes).round().astype(int).clip(0, None)

    # Views: only a fraction of followers see a post; authenticity drives it hard.
    views_ratio = (0.18 + 0.55 * lat["authenticity"].to_numpy()) * (0.7 + 0.6 * lat["content_quality"].to_numpy())
    views_ratio = (views_ratio * rng.lognormal(0, 0.22, n)).clip(0.03, 1.9)
    avg_views = (followers * views_ratio).round().astype(int)
    avg_reach = (avg_views * rng.uniform(0.72, 0.95, n)).round().astype(int)

    # Growth: momentum from quality + consistency, minus saturation drag.
    growth = (
        0.004
        + 0.055 * lat["content_quality"].to_numpy()
        + 0.030 * lat["consistency"].to_numpy()
        - 0.045 * lat["promo_saturation"].to_numpy()
        - 0.020 * np.log10(followers / 10_000.0).clip(0, None)
        + rng.normal(0, 0.014, n)
    )

    posting_freq = (1.0 + 16.0 * lat["consistency"].to_numpy() * rng.lognormal(0, 0.35, n)).clip(0.5, 45)

    # Following: low-authenticity accounts follow-spam.
    following = (
        followers ** 0.42 * (1.0 + 9.0 * (1.0 - lat["authenticity"].to_numpy())) * rng.lognormal(0, 0.4, n)
    ).round().astype(int).clip(20, 90_000)

    secondary = np.array([rng.choice([x for x in NICHES if x != p]) for p in niches])

    return pd.DataFrame(
        {
            "influencer_id": [f"INF{idx:05d}" for idx in range(n)],
            "handle": [f"@{_fake_handle(rng)}" for _ in range(n)],
            "primary_niche": niches,
            "secondary_niche": secondary,
            "followers": followers,
            "following": following,
            "avg_likes": avg_likes,
            "avg_comments": avg_comments,
            "avg_views": avg_views,
            "avg_reach": avg_reach,
            "engagement_rate": er,
            "comments_to_likes": ctl,
            "views_to_followers": avg_views / followers,
            "follower_growth_rate": growth,
            "posting_frequency_month": posting_freq,
            "audience_geo": rng.choice(
                ["IN-North", "IN-South", "IN-West", "IN-East", "SEA", "MENA", "US/EU"],
                n,
                p=[0.20, 0.20, 0.19, 0.11, 0.13, 0.07, 0.10],
            ),
            "audience_age_band": rng.choice(
                ["13-17", "18-24", "25-34", "35-44", "45+"], n, p=[0.07, 0.36, 0.34, 0.16, 0.07]
            ),
            "audience_gender_skew": rng.beta(3, 3, n),  # fraction female
        }
    )


_HANDLE_A = ["the", "real", "just", "daily", "urban", "wild", "quiet", "little", "hey", "its"]
_HANDLE_B = ["maya", "arjun", "reva", "kabir", "nina", "dev", "sana", "ravi", "tara", "ish",
             "leo", "zoya", "amit", "priya", "noor", "vik", "anya", "raj", "mira", "sam"]
_HANDLE_C = ["cooks", "lifts", "styles", "builds", "travels", "reads", "makes", "reviews",
             "daily", "diaries", "studio", "lab", "co", "official", "hq"]


def _fake_handle(rng: np.random.Generator) -> str:
    parts = [rng.choice(_HANDLE_B)]
    if rng.random() < 0.55:
        parts.insert(0, rng.choice(_HANDLE_A))
    if rng.random() < 0.65:
        parts.append(rng.choice(_HANDLE_C))
    sep = rng.choice(["", "", "_", "."])
    h = sep.join(parts)
    if rng.random() < 0.25:
        h += str(rng.integers(1, 999))
    return h


# ==========================================================================
# Caption generation
# ==========================================================================


def _make_caption(
    rng: np.random.Generator,
    niche: str,
    lat_row: pd.Series,
    force_label: str | None = None,
) -> dict:
    """Return one post's caption plus its generative ground-truth labels."""
    lex = NICHE_LEXICON[niche]

    # Decide the rhetorical mode.
    if force_label is not None:
        mode = force_label
    else:
        # Higher promo saturation -> more sarcasm-bait and more promo.
        p_sarcastic = 0.06 + 0.10 * float(lat_row["promo_saturation"])
        p_negative = 0.10 + 0.10 * (1 - float(lat_row["content_quality"]))
        p_positive = 0.45 + 0.20 * float(lat_row["content_quality"])
        probs = np.array([p_positive, p_negative, p_sarcastic])
        probs = np.append(probs, max(0.05, 1 - probs.sum()))
        probs = probs / probs.sum()
        mode = rng.choice(["positive", "negative", "sarcastic", "neutral"], p=probs)

    subject = rng.choice(lex["subject"])
    obj = rng.choice(lex["object"])
    desc = rng.choice(lex["descriptor"])
    action = rng.choice(lex["action"])
    brand = rng.choice(lex["brands"])
    product = rng.choice(lex["products"])

    if mode == "sarcastic":
        tmpl = rng.choice(SARCASM_TEMPLATES)
        body = tmpl.format(
            great=rng.choice(SARCASM_GREAT),
            object=obj,
            fails=rng.choice(SARCASM_FAILS),
            subject=subject,
            absurd=rng.choice(SARCASM_ABSURD),
            descriptor=desc,
        )
        sentiment_truth = "negative"   # sarcasm here is negative intent, positive surface
    elif mode == "positive":
        body = f"{rng.choice(POSITIVE_OPENERS)} {subject}. I {action} the {obj} and it is genuinely {desc}."
        sentiment_truth = "positive"
    elif mode == "negative":
        body = f"{rng.choice(NEGATIVE_OPENERS)} {subject}. I {action} the {obj} and it is not {desc} at all."
        sentiment_truth = "negative"
    else:
        body = f"{rng.choice(NEUTRAL_OPENERS)} {subject}: {action} the {obj}, went for something more {desc}."
        sentiment_truth = "neutral"

    # Optional embellishments -------------------------------------------------
    has_promo = rng.random() < (0.10 + 0.55 * float(lat_row["promo_saturation"]))
    has_cta = rng.random() < (0.20 + 0.45 * float(lat_row["promo_saturation"]))
    has_question = rng.random() < 0.22
    mentions_brand = has_promo or rng.random() < 0.22
    mentions_product = mentions_brand or rng.random() < 0.18

    extras = []
    if mentions_brand and not has_promo:
        extras.append(f"Wearing {brand}." if niche == "Fashion" else f"Using {brand}.")
    if has_promo:
        extras.append(
            rng.choice(PROMO_PHRASES).format(
                brand=brand, code=f"{brand.split()[0].upper()[:6]}{rng.integers(10, 99)}", pct=rng.choice([10, 15, 20, 25, 30])
            )
        )
    if mentions_product and rng.random() < 0.6:
        extras.append(f"The {product} is the one I keep reaching for.")
    if has_question:
        extras.append(rng.choice(QUESTION_PHRASES))
    if has_cta:
        extras.append(rng.choice(CTA_PHRASES))

    caption_text = " ".join([body] + extras)

    # Hashtags: niche-focused creators use tighter, more on-topic tag sets.
    n_tags = int(rng.integers(2, 12))
    focus = float(lat_row["niche_focus"])
    tags = []
    for _ in range(n_tags):
        if rng.random() < 0.25 + 0.55 * focus:
            tags.append(rng.choice(lex["hashtags"]))
        else:
            tags.append(rng.choice(GENERIC_HASHTAGS))
    tags = list(dict.fromkeys(tags))
    caption = caption_text + " " + " ".join(f"#{t}" for t in tags)

    return {
        "caption": caption,
        "hashtags": tags,
        "gen_mode": mode,
        "gen_sentiment": sentiment_truth,
        "gen_is_sarcastic": int(mode == "sarcastic"),
        "gen_has_promo": int(has_promo),
        "gen_has_cta": int(has_cta),
        "gen_has_question": int(has_question),
        "gen_brand": brand if mentions_brand else None,
        "gen_product": product if mentions_product else None,
    }


def _generate_posts(rng: np.random.Generator, profiles: pd.DataFrame, lat: pd.DataFrame) -> pd.DataFrame:
    rows = []
    lo, hi = POSTS_PER_INFLUENCER
    for i, prof in profiles.iterrows():
        lat_row = lat.iloc[i]
        n_posts = int(rng.integers(lo, hi + 1))
        for p in range(n_posts):
            niche = prof["primary_niche"] if rng.random() < (0.55 + 0.42 * float(lat_row["niche_focus"])) \
                else prof["secondary_niche"]
            cap = _make_caption(rng, niche, lat_row)
            # Per-post engagement varies around the creator's mean.
            mult = float(rng.lognormal(0, 0.45))
            likes = int(prof["avg_likes"] * mult)
            comments = int(prof["avg_comments"] * mult * rng.lognormal(0, 0.2))
            views = int(prof["avg_views"] * mult * rng.lognormal(0, 0.25))
            rows.append(
                {
                    "post_id": f"{prof['influencer_id']}_P{p:03d}",
                    "influencer_id": prof["influencer_id"],
                    "post_niche": niche,
                    "likes": likes,
                    "comments": comments,
                    "views": views,
                    "days_ago": int(rng.integers(1, 365)),
                    **cap,
                }
            )
    df = pd.DataFrame(rows)
    df["hashtags"] = df["hashtags"].apply(lambda x: "|".join(x))
    return df


# ==========================================================================
# Brands and campaign outcomes (the supervised target)
# ==========================================================================


def _generate_brands(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for b in range(N_BRANDS):
        niche = rng.choice(NICHES)
        lex = NICHE_LEXICON[niche]
        rows.append(
            {
                "brand_id": f"BRD{b:04d}",
                "brand_name": f"{rng.choice(lex['brands'])} {rng.choice(['Labs', 'Co', 'India', 'Studio', 'Group'])}",
                "category": niche,
                "target_geo": rng.choice(["IN-North", "IN-South", "IN-West", "IN-East", "SEA", "MENA", "US/EU"]),
                "target_age_band": rng.choice(["18-24", "25-34", "35-44"], p=[0.42, 0.42, 0.16]),
                "budget_inr": int(rng.choice([50_000, 150_000, 400_000, 1_000_000, 3_000_000])),
                "brand_keywords": "|".join(rng.choice(lex["products"], size=3, replace=False)),
                "competitor_brands": "|".join(rng.choice(lex["brands"], size=2, replace=False)),
            }
        )
    return pd.DataFrame(rows)


def generate_campaigns(
    profiles: pd.DataFrame,
    lat: pd.DataFrame,
    brands: pd.DataFrame,
    network: pd.DataFrame,
    seed: int = SEED,
    label_fraction: float = 0.55,
) -> pd.DataFrame:
    """
    Simulate historical sponsored collaborations. PASS 2 of generation.

    This runs *after* the network graph has been built, because campaign
    performance genuinely depends on where a creator sits in the topical
    network - a creator embedded in a dense category cluster gets more
    within-category amplification than an isolated one.

    Amplification, and why it is only PARTLY observable
    ---------------------------------------------------
    An earlier version set the amplification term to a deterministic linear
    function of measured PageRank percentile. That was circular in the worst
    way: PageRank percentile is itself a model feature, so "the network pillar
    earns its place" reduced to "the model can learn a straight line through a
    column it was handed". An audit measured it - a five-term regression on the
    generator's own observable inputs recovered R^2 0.505 of the 0.694 the full
    model reached, i.e. 73% of the headline number was arithmetic.

    The fix is to say what is actually true of the real world: a creator has an
    underlying propensity to be amplified, and topical centrality *measures part
    of it*. AMPLIFICATION_OBSERVED_SHARE of the term comes from measured
    PageRank; the rest comes from a latent the feature table cannot see. The
    model can still recover the observable part - it should, that part is real -
    but it now has to treat the remainder as signal it cannot reach, which is
    exactly the situation a real deployment is in.

    The latent is drawn here rather than in pass 1 on purpose: pass 1 produces
    the posts, and touching it would invalidate every NLP artifact downstream.
    It is appended to latents.parquet, which is never joined into any feature
    table.
    """
    rng = np.random.default_rng(seed + 7)
    n = len(profiles)
    labelled_idx = rng.choice(n, size=int(n * label_fraction), replace=False)

    # Measured topical centrality, aligned to profile order.
    net = network.set_index("influencer_id").reindex(profiles["influencer_id"])
    amp_pct = net["pagerank_pct"].fillna(0.5).to_numpy()

    # The unobservable half of amplification. Uniform on [0, 1] to match the
    # marginal of a percentile, and drawn from its own stream so adding it does
    # not disturb any other random draw in this function.
    pull_rng = np.random.default_rng(seed + 991)
    network_pull = pull_rng.uniform(0.0, 1.0, size=n)

    amp_driver = (AMPLIFICATION_OBSERVED_SHARE * amp_pct
                  + (1.0 - AMPLIFICATION_OBSERVED_SHARE) * network_pull)
    amplification = 0.72 + 0.56 * amp_driver       # ranges ~0.72 .. 1.28

    rows = []
    for i in labelled_idx:
        prof = profiles.iloc[i]
        lt = lat.iloc[i]
        n_campaigns = int(rng.integers(1, 4))
        for c in range(n_campaigns):
            # Brands mostly hire in-category, sometimes adjacent.
            if rng.random() < 0.78:
                pool = brands[brands["category"] == prof["primary_niche"]]
            else:
                pool = brands
            if len(pool) == 0:
                pool = brands
            brand = pool.iloc[int(rng.integers(0, len(pool)))]

            fit = 1.0 if brand["category"] == prof["primary_niche"] else (
                0.62 if brand["category"] == prof["secondary_niche"] else 0.30
            )
            geo_match = 1.0 if brand["target_geo"] == prof["audience_geo"] else 0.72
            age_match = 1.0 if brand["target_age_band"] == prof["audience_age_band"] else 0.80

            # Sponsored posts underperform organic content - a well-known and
            # consistently reported effect. Base curve is the benchmark ER for
            # the brand's category, discounted for being an ad.
            base = bm.expected_er(prof["followers"], brand["category"]) * SPONSORED_DISCOUNT

            campaign_er = (
                base
                * (0.35 + 1.30 * float(lt["content_quality"])) ** 1.15
                * (0.40 + 0.90 * float(lt["authenticity"])) ** 1.30
                * (0.85 + 0.30 * float(lt["consistency"]))
                * (1.25 - 0.75 * float(lt["promo_saturation"]))
                * (0.55 + 0.55 * fit * float(lt["niche_focus"]))
                * geo_match
                * age_match
                * amplification[i]
                * rng.lognormal(0, CAMPAIGN_NOISE_SIGMA)   # irreducible noise
            )

            # Fee: benchmark rate card, adjusted for the creator's engagement
            # premium/discount relative to the benchmark for their size, plus
            # negotiation noise.
            bench_er = bm.expected_er(prof["followers"], prof["primary_niche"])
            er_premium = float(np.clip(prof["engagement_rate"] / bench_er, 0.4, 2.5))
            fee = (
                bm.expected_fee(prof["followers"], brand["category"])
                * er_premium ** 0.55
                * rng.lognormal(0, 0.22)
            )

            rows.append(
                {
                    "campaign_id": f"CMP{len(rows):06d}",
                    "influencer_id": prof["influencer_id"],
                    "brand_id": brand["brand_id"],
                    "brand_category": brand["category"],
                    "category_fit_true": fit,
                    "campaign_engagement_rate_raw": campaign_er,
                    "benchmark_er": base,
                    "followers_snapshot": int(prof["followers"]),
                    "fee_inr": int(fee),
                }
            )

    df = pd.DataFrame(rows)

    # Record the latent alongside the others. Written to disk for ceiling
    # reporting and audit; never joined into a feature table.
    lat_path = PROCESSED_DIR / "latents.parquet"
    if lat_path.exists():
        stored = pd.read_parquet(lat_path)
        stored["network_pull"] = network_pull[: len(stored)]
        stored.to_parquet(lat_path, index=False)

    # Mean-normalise the multiplier chain so that, across the population,
    # campaign engagement sits at SPONSORED_DISCOUNT of the published benchmark
    # curve rather than at whatever the product of the multipliers happens to
    # average out to. Relative ordering between creators is untouched.
    ratio = (df["campaign_engagement_rate_raw"] / df["benchmark_er"]).mean()
    df["campaign_engagement_rate"] = (
        df["campaign_engagement_rate_raw"] / ratio * SPONSORED_DISCOUNT
    ).clip(0.0004, 0.22)
    df["campaign_engagements"] = (
        df["campaign_engagement_rate"] * df["followers_snapshot"]
    ).round().astype(int)

    return df.drop(columns=["campaign_engagement_rate_raw", "benchmark_er", "followers_snapshot"])


# ==========================================================================
# Entry point
# ==========================================================================


def generate(seed: int = SEED, n_influencers: int = N_INFLUENCERS) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    random.seed(seed)

    lat = _draw_latents(rng, n_influencers)
    profiles = _generate_profiles(rng, lat)
    profiles["follower_tier"] = profiles["followers"].apply(tier_of)

    posts = _generate_posts(rng, profiles, lat)
    brands = _generate_brands(rng)

    # Latents saved separately - used ONLY for reporting the achievable ceiling,
    # never merged into the feature table.
    latents = lat.copy()
    latents.insert(0, "influencer_id", profiles["influencer_id"])

    out = {
        "profiles": profiles,
        "posts": posts,
        "brands": brands,
        "latents": latents,
    }
    for name, df in out.items():
        df.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)
    return out


def generate_campaigns_step(seed: int = SEED) -> pd.DataFrame:
    """PASS 2 - run after src.network.sna has produced network features."""
    from src.network.sna import GRAPH_DIR

    profiles = pd.read_parquet(PROCESSED_DIR / "profiles.parquet")
    latents = pd.read_parquet(PROCESSED_DIR / "latents.parquet")
    brands = pd.read_parquet(PROCESSED_DIR / "brands.parquet")
    net_path = GRAPH_DIR / "network_features.parquet"
    if not net_path.exists():
        raise FileNotFoundError(
            "Network features not found. Run `python -m src.network.sna` first - "
            "campaign outcomes depend on measured topical centrality."
        )
    network = pd.read_parquet(net_path)

    lat = latents.drop(columns=["influencer_id"])
    campaigns = generate_campaigns(profiles, lat, brands, network, seed=seed)
    campaigns.to_parquet(PROCESSED_DIR / "campaigns.parquet", index=False)
    return campaigns


if __name__ == "__main__":
    import time

    t0 = time.time()
    data = generate()
    print(f"Generated in {time.time() - t0:.1f}s")
    for k, v in data.items():
        print(f"  {k:12s} {len(v):>8,} rows  {v.shape[1]:>3} cols")
    print("\nNext: python -m src.network.sna, then python -m src.data.generate_synthetic --campaigns")
