"""
Score a typed brief against the whole creator database, at request time.

This is the engine behind the brand intake page. A brand that has never used
the platform types who it is, what the campaign is, what it will pay and what
it wants out of it, and gets a ranked shortlist back.

How this relates to the batch fit matrix
----------------------------------------
Everything a brand sees elsewhere in the app comes from nectar_fit.parquet,
which was scored offline by src/models/brandfit.py. That cannot serve a brand
that does not exist yet, so this module re-implements the same composite -
identical components, identical weights, identical gates - with one deliberate
substitution:

    semantic_similarity   SBERT cosine   ->   TF-IDF cosine

The hosted app loads no neural model on purpose, so text that arrives at
request time cannot be embedded. TF-IDF matches shared words, not shared
meaning: "cruelty free cosmetics" will not find a creator whose captions say
"vegan makeup" unless a word actually overlaps. The page reports exactly which
of the brand's words landed and which were ignored, rather than hiding the
weaker method behind a confident number.

When a brief matches fewer than MIN_QUERY_TERMS vocabulary terms the semantic
component is held at a neutral 0.5 for everyone instead of scoring nearly
everyone at zero. A component that is noise for all 2,000 creators should not
be allowed to reorder them.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import data

# Copied from src/models/brandfit.py rather than imported: the app package is
# importable on its own on the hosted machine, where src/ is present but the
# modelling dependencies are not. tests/test_match.py asserts the two stay
# equal, so a change to the weights cannot silently diverge.
COMPONENT_WEIGHTS = {
    "semantic_similarity": 0.34,
    "category_match": 0.28,
    "audience_match": 0.18,
    "content_safety": 0.12,
    "consistency": 0.08,
}

CONFLICT_WINDOW_DAYS = 180
CONFLICT_MIN_REPEAT = 2
MIN_QUERY_TERMS = 2

OBJECTIVES = {
    "Awareness":     ("score_reach", "how far the post travels"),
    "Consideration": ("score_balanced", "reach and engagement together"),
    "Conversion":    ("score_rate", "how hard the audience engages"),
}

# How much the campaign goal is allowed to move the ranking away from pure fit.
# Fit decides who is appropriate; the goal decides who is most useful among
# them. At 0.40 a creator two deciles better on the goal can overtake a
# marginally better-fitting one, which is the intended behaviour, but a poor
# fit cannot be rescued by reach alone.
GOAL_WEIGHT = 0.40

TOKEN_RE = re.compile(r"[a-z][a-z0-9]{2,}")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def vocab() -> pd.DataFrame:
    return data.require("nectar_vocab.parquet", "Lexical vocabulary")


def creator_terms() -> pd.DataFrame:
    return data.require("nectar_creator_terms.parquet", "Creator term profiles")


def brand_mentions() -> pd.DataFrame:
    df = data.load("nectar_brand_mentions.parquet")
    return df if df is not None else pd.DataFrame(
        columns=["influencer_id", "brand", "n_mentions", "n_paid", "days_ago_min"])


# --------------------------------------------------------------------------
# Text -> weighted query
# --------------------------------------------------------------------------
def _tokens(text: str) -> list[str]:
    from src.nectar.build_terms import STOPWORDS  # noqa: PLC0415
    return [t for t in TOKEN_RE.findall(str(text or "").lower()) if t not in STOPWORDS]


def _tokens_safe(text: str) -> list[str]:
    """Tokenise without importing src/, which is not guaranteed on the host."""
    try:
        return _tokens(text)
    except Exception:                                          # noqa: BLE001
        return [t for t in TOKEN_RE.findall(str(text or "").lower())
                if t not in _FALLBACK_STOPWORDS]


_FALLBACK_STOPWORDS = frozenset("""
and the for with from that this our your are was were you they them their
new all any can has have had not but who how why when what which
""".split())


def query_terms(text: str, vcb: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return (weighted query, matched terms, ignored words)."""
    idf = dict(zip(vcb["term"], vcb["idf"]))
    toks = _tokens_safe(text)
    tf = Counter(t for t in toks if t in idf)
    ignored = sorted({t for t in toks if t not in idf})
    if not tf:
        return pd.DataFrame(columns=["term", "q"]), [], ignored
    w = {t: (1.0 + math.log(c)) * idf[t] for t, c in tf.items()}
    norm = math.sqrt(sum(v * v for v in w.values())) or 1.0
    q = pd.DataFrame({"term": list(w), "q": [v / norm for v in w.values()]})
    return q, sorted(tf), ignored


def lexical_similarity(text: str) -> tuple[pd.Series, dict]:
    """Cosine of the typed text against every creator's term profile."""
    vcb, terms = vocab(), creator_terms()
    q, matched, ignored = query_terms(text, vcb)
    info = {"matched": matched, "ignored": ignored,
            "vocab_size": int(len(vcb)), "fallback": len(matched) < MIN_QUERY_TERMS}
    if info["fallback"]:
        return pd.Series(dtype="float64"), info
    hit = terms.merge(q, on="term", how="inner")
    cos = hit.assign(p=hit["weight"] * hit["q"]).groupby("influencer_id")["p"].sum()
    return cos.clip(0.0, 1.0), info


# --------------------------------------------------------------------------
# The brief
# --------------------------------------------------------------------------
@dataclass
class Brief:
    category: str
    brand_text: str = ""
    campaign_text: str = ""
    competitors: list[str] = field(default_factory=list)
    geos: list[str] = field(default_factory=list)
    ages: list[str] = field(default_factory=list)
    min_followers: int = 0
    budget: int = 0
    cap: int = 10 ** 9
    n_reel: int = 0
    n_story: int = 0
    n_carousel: int = 0
    objective: str = "Consideration"

    @property
    def text(self) -> str:
        return f"{self.brand_text} {self.campaign_text}".strip()


# --------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------
def _conflict_gate(creators: pd.DataFrame, competitors: list[str]) -> pd.DataFrame:
    """Reproduces eligibility() in brandfit.py against the exported evidence.

    A paid post, or repeated mentions inside the window, is a veto. A single
    unpaid mention is a 0.80 flag, not a disqualification - the first version
    of that gate blocked on any mention ever and removed 85% of every pool,
    because naming the category's best-known brand is near-universal within
    the category.
    """
    gate = pd.Series(1.0, index=creators.influencer_id, name="gate")
    reason = pd.Series("", index=creators.influencer_id, name="gate_reason")
    names = {c.strip().lower() for c in competitors if c and c.strip()}
    if not names:
        return pd.DataFrame({"gate": gate, "gate_reason": reason})

    m = brand_mentions()
    m = m[m["brand"].isin(names) & (m["days_ago_min"] <= CONFLICT_WINDOW_DAYS)]
    for iid, grp in m.groupby("influencer_id"):
        hard = grp[(grp.n_paid > 0) | (grp.n_mentions >= CONFLICT_MIN_REPEAT)]
        soft = grp[(grp.n_paid == 0) & (grp.n_mentions < CONFLICT_MIN_REPEAT)]
        if iid not in gate.index:
            continue
        if len(hard):
            who = ", ".join(sorted(hard.brand.str.title()))
            gate[iid] = 0.0
            reason[iid] = f"Blocked: recent paid or repeated work with {who}"
        elif len(soft):
            who = ", ".join(sorted(soft.brand.str.title()))
            gate[iid] = 0.80
            reason[iid] = f"Mentioned {who} once, unpaid - worth checking"
    return pd.DataFrame({"gate": gate, "gate_reason": reason})


def _promo_penalty(creators: pd.DataFrame) -> pd.DataFrame:
    promo = creators.set_index("influencer_id")["content_promo_rate"].fillna(0.0)
    mult = pd.Series(1.0, index=promo.index)
    note = pd.Series("", index=promo.index)
    heavy, elevated = promo > 0.55, (promo > 0.38) & (promo <= 0.55)
    mult[heavy] = 0.65
    mult[elevated] = 0.85
    note[heavy] = "Heavily ad-saturated feed"
    note[elevated] = "Elevated ad load"
    return pd.DataFrame({"promo_mult": mult, "promo_note": note})


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def score(brief: Brief) -> tuple[pd.DataFrame, dict]:
    c = data.creators().copy()
    c["influencer_id"] = c["influencer_id"].astype(str)

    cos, info = lexical_similarity(brief.text)
    if info["fallback"]:
        sem = pd.Series(0.5, index=c.influencer_id)
    else:
        sem = (cos.reindex(c.influencer_id).fillna(0.0) + 1.0) / 2.0
    c["fit_semantic_similarity"] = sem.values

    c["fit_category_match"] = np.where(
        c.primary_niche == brief.category, 1.0,
        np.where(c.secondary_niche == brief.category, 0.55, 0.15))

    geo_ok = c.audience_geo.isin(brief.geos) if brief.geos else True
    age_ok = c.audience_age_band.isin(brief.ages) if brief.ages else True
    c["fit_audience_match"] = (0.55 * np.where(geo_ok, 1.0, 0.45)
                               + 0.45 * np.where(age_ok, 1.0, 0.55))

    neg = c.content_share_negative.fillna(0.0)
    irony = c.content_irony_rate.fillna(0.0)
    c["fit_content_safety"] = np.clip(1.0 - 0.8 * neg - 0.5 * irony, 0.0, 1.0)

    ent = c.content_topic_entropy
    c["fit_consistency"] = np.where(ent.notna(), np.clip(1.0 - ent / 3.0, 0.0, 1.0), 0.5)

    c["fit_ungated"] = sum(
        w * c[f"fit_{k}"] for k, w in COMPONENT_WEIGHTS.items())

    gates = _conflict_gate(c, brief.competitors)
    promo = _promo_penalty(c)
    c = c.merge(gates, left_on="influencer_id", right_index=True, how="left")
    c = c.merge(promo, left_on="influencer_id", right_index=True, how="left")
    c["gate"] = c["gate"].fillna(1.0) * c["promo_mult"].fillna(1.0)
    c["fit"] = c["fit_ungated"] * c["gate"]

    c["fee"] = (c.rate_reel * brief.n_reel + c.rate_story * brief.n_story
                + c.rate_carousel * brief.n_carousel)

    c["eligible"] = (
        (c.followers >= brief.min_followers)
        & (c.fee <= brief.cap)
        & (c.gate > 0)
    )

    obj_col = OBJECTIVES.get(brief.objective, OBJECTIVES["Consideration"])[0]
    c["goal_score"] = c[obj_col]
    c["goal_pct"] = c["goal_score"].rank(pct=True)
    c["fit_pct"] = c["fit"].rank(pct=True)
    c["match"] = (1 - GOAL_WEIGHT) * c["fit_pct"] + GOAL_WEIGHT * c["goal_pct"]

    c["fit_display"] = (c["fit"] * 100).round(1)
    c["match_display"] = (c["match"] * 100).round(1)

    c = c.sort_values(["eligible", "match"], ascending=[False, False]).reset_index(drop=True)
    c["rank"] = np.arange(1, len(c) + 1)

    info.update({
        "eligible": int(c.eligible.sum()),
        "pool": int(len(c)),
        "blocked": int((c.gate == 0).sum()),
        "objective": brief.objective,
        "objective_column": obj_col,
        "median_fee": float(c.loc[c.eligible, "fee"].median()) if c.eligible.any() else 0.0,
    })
    if brief.budget and info["median_fee"]:
        info["creators_affordable"] = int(min(brief.budget // info["median_fee"],
                                              info["eligible"]))
    else:
        info["creators_affordable"] = 0
    return c, info


def reasons(row, brief: Brief, info: dict) -> list[str]:
    """Plain-language explanation for one row, strongest first."""
    out: list[str] = []
    if row.fit_category_match >= 1.0:
        out.append(f"Primary niche is {brief.category}")
    elif row.fit_category_match >= 0.55:
        out.append(f"Covers {brief.category} as a secondary niche")
    if not info["fallback"] and row.fit_semantic_similarity > 0.55:
        out.append("Captions use the language of your brief")
    if brief.geos and row.audience_geo in brief.geos:
        out.append(f"Audience is in {row.audience_geo}")
    if brief.ages and row.audience_age_band in brief.ages:
        out.append(f"Audience skews {row.audience_age_band}")
    if row.fit_consistency > 0.7:
        out.append("Posts on a narrow, predictable set of topics")
    if getattr(row, "gate_reason", ""):
        out.append(row.gate_reason)
    if getattr(row, "promo_note", ""):
        out.append(row.promo_note)
    if not row.eligible:
        if row.followers < brief.min_followers:
            out.append("Below your audience floor")
        elif row.fee > brief.cap:
            out.append("Brief price is above your per-creator cap")
    return out
