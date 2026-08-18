"""
Rule-based content feature extraction: hashtags, brands, products, promotional
language, calls-to-action, questions, and length statistics.

These are cheap, deterministic and fully auditable - which is exactly why they
belong in a brand-safety pipeline. A brand manager can be shown *why* a creator
was flagged as ad-saturated ("7 of the last 20 posts contain a discount code"),
which is not something an embedding can offer.

Promotional-language detection uses an explicit, inspectable pattern list rather
than a classifier. Disclosure regulations differ by market and change often;
a visible list can be updated by a non-ML person in minutes.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Patterns
# --------------------------------------------------------------------------

HASHTAG_RE = re.compile(r"#(\w+)")
MENTION_RE = re.compile(r"@(\w+)")
URL_RE = re.compile(r"https?://\S+|\bwww\.\S+")
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]"
)

# Explicit ad-disclosure markers - the strongest promotional signal.
DISCLOSURE_PATTERNS = [
    r"#ad\b", r"#sponsored\b", r"#paidpartnership\b", r"#gifted\b", r"#partner\b",
    r"\bpaid partnership\b", r"\bsponsored by\b", r"\bin partnership with\b",
    r"\bgifted by\b", r"\bthanks .{0,30} for sponsoring\b", r"\bad\s*[:|-]\s",
]

# Commercial-intent language that is not a formal disclosure.
PROMO_PATTERNS = [
    r"\buse (my |the )?code\b", r"\bdiscount code\b", r"\bpromo code\b",
    r"\b\d{1,2}%\s*off\b", r"\bexclusive (discount|offer|deal)\b",
    r"\blimited[- ]time\b", r"\boffer ends\b", r"\bshop now\b",
    r"\bavailable now\b", r"\bbuy (it |them )?(now|here)\b",
    r"\bswipe up\b", r"\bfor my followers\b",
]

CTA_PATTERNS = [
    r"\blink in bio\b", r"\bcomment below\b", r"\bsave this\b", r"\bshare this\b",
    r"\bfollow for more\b", r"\bswipe\b", r"\btap the link\b", r"\bdrop a\b",
    r"\bcheck out\b", r"\bsign up\b", r"\bdm me\b", r"\bclick the link\b",
    r"\bhit (the )?follow\b", r"\bturn on notifications\b", r"\btag someone\b",
]

QUESTION_PATTERNS = [r"\?", r"\bwhat do you\b", r"\banyone else\b", r"\bthoughts\b"]

_DISCLOSURE_RE = re.compile("|".join(DISCLOSURE_PATTERNS), re.I)
_PROMO_RE = re.compile("|".join(PROMO_PATTERNS), re.I)
_CTA_RE = re.compile("|".join(CTA_PATTERNS), re.I)
_QUESTION_RE = re.compile("|".join(QUESTION_PATTERNS), re.I)

STOPWORDS = set(
    """a an the and or but if then than that this these those i me my we our you your he she it
    they them his her its their of in on at to for with without from by as is are was were be been
    being do does did doing have has had having will would can could should may might must not no
    so just very really more most much many some any all one two also into over under out up down
    about after before again more too own same s t don now here there when where why how what which
    who whom while because until against between during through above below off further once each
    few both other such only nor own too""".split()
)

WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z'\-]{2,}")


# --------------------------------------------------------------------------
# Per-post extraction
# --------------------------------------------------------------------------


def extract_post(caption: str, brand_vocab: set[str] | None = None,
                 product_vocab: set[str] | None = None) -> dict:
    text = caption or ""
    lower = text.lower()

    hashtags = [h.lower() for h in HASHTAG_RE.findall(text)]
    body = HASHTAG_RE.sub(" ", URL_RE.sub(" ", text))
    words = WORD_RE.findall(body.lower())

    brands_found = sorted({b for b in (brand_vocab or set()) if b.lower() in lower})
    products_found = sorted({p for p in (product_vocab or set()) if p.lower() in lower})

    return {
        "n_hashtags": len(hashtags),
        "n_unique_hashtags": len(set(hashtags)),
        "n_mentions": len(MENTION_RE.findall(text)),
        "n_urls": len(URL_RE.findall(text)),
        "n_emoji": len(EMOJI_RE.findall(text)),
        "n_words": len(words),
        "n_chars": len(text),
        "avg_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "has_disclosure": int(bool(_DISCLOSURE_RE.search(text))),
        "n_promo_cues": len(_PROMO_RE.findall(text)),
        "has_promo": int(bool(_PROMO_RE.search(text)) or bool(_DISCLOSURE_RE.search(text))),
        "has_cta": int(bool(_CTA_RE.search(text))),
        "n_cta_cues": len(_CTA_RE.findall(text)),
        "has_question": int(bool(_QUESTION_RE.search(text))),
        "caps_ratio": (sum(c.isupper() for c in text) / len(text)) if text else 0.0,
        "exclamations": text.count("!"),
        "brands_mentioned": "|".join(brands_found),
        "products_mentioned": "|".join(products_found),
        "n_brands_mentioned": len(brands_found),
        "n_products_mentioned": len(products_found),
        "_hashtags": hashtags,
        "_words": [w for w in words if w not in STOPWORDS],
    }


def extract_frame(
    posts: pd.DataFrame,
    caption_col: str = "caption",
    brand_vocab: set[str] | None = None,
    product_vocab: set[str] | None = None,
) -> pd.DataFrame:
    recs = [
        extract_post(c, brand_vocab, product_vocab) for c in posts[caption_col].fillna("")
    ]
    df = pd.DataFrame(recs)
    df["post_id"] = posts["post_id"].to_numpy()
    df["influencer_id"] = posts["influencer_id"].to_numpy()
    return df


# --------------------------------------------------------------------------
# Keyword extraction (per influencer)
# --------------------------------------------------------------------------


def top_keywords(
    docs_by_influencer: dict[str, list[str]], top_n: int = 15, min_df: int = 2
) -> dict[str, list[tuple[str, float]]]:
    """TF-IDF keywords per influencer over their aggregated post vocabulary.

    Deliberately TF-IDF rather than a neural keyword extractor: keywords are
    surfaced directly to brand users in the dashboard, and TF-IDF terms are
    always literally present in the creator's captions, so a user who clicks
    through can verify every one of them.
    """
    n_docs = len(docs_by_influencer)
    df_counts: Counter = Counter()
    for words in docs_by_influencer.values():
        df_counts.update(set(words))

    idf = {
        t: np.log((1 + n_docs) / (1 + c)) + 1.0
        for t, c in df_counts.items()
        if c >= min_df
    }

    out: dict[str, list[tuple[str, float]]] = {}
    for inf_id, words in docs_by_influencer.items():
        tf = Counter(w for w in words if w in idf)
        if not tf:
            out[inf_id] = []
            continue
        total = sum(tf.values())
        scored = sorted(
            (((c / total) * idf[t], t) for t, c in tf.items()), reverse=True
        )[:top_n]
        out[inf_id] = [(t, round(float(s), 5)) for s, t in scored]
    return out


def top_hashtags(
    tags_by_influencer: dict[str, list[str]], top_n: int = 12
) -> dict[str, list[tuple[str, int]]]:
    return {
        k: Counter(v).most_common(top_n) for k, v in tags_by_influencer.items()
    }
