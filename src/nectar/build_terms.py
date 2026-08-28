"""
Lexical term profiles for creators, so a brand can be matched from free text.

Why this exists, and why it is not SBERT
----------------------------------------
The batch brand-fit matrix in src/models/brandfit.py uses SBERT cosine for its
semantic term. That is the right tool, but it needs a 90 MB neural model at
inference time, and the hosted app deliberately carries no model at all - it
reads precomputed parquet and nothing else. A brand typing a fresh description
into the app is, by definition, not precomputed.

So the intake page scores the typed brief against a TF-IDF profile of each
creator instead. This is a weaker method than SBERT and the page says so: it
matches shared vocabulary, not shared meaning, and a brief that says "cruelty
free cosmetics" will not match a creator whose captions say "vegan makeup"
unless the words themselves overlap. Every other component of the fit score -
category affinity, audience overlap, content safety, consistency, and the
eligibility gates - is computed with the exact same code as the batch matrix,
so only the one term differs.

Writes two small tables into app_data/:

    nectar_vocab.parquet          term, df, idf
    nectar_creator_terms.parquet  influencer_id, term, weight
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pandas as pd

# Terms carried per creator. The full matrix is 2,000 x ~4,000 and mostly
# zeros; keeping the heaviest 40 terms each turns it into ~80,000 rows, which
# is about 700 KB of parquet and loses almost nothing - the tail weights are
# small enough that they cannot move a cosine ranking.
TOP_TERMS = 40

# A term in fewer than this many creator profiles cannot generalise; a term in
# more than this share of them cannot discriminate.
MIN_DF = 3
MAX_DF_SHARE = 0.55

TOKEN_RE = re.compile(r"[a-z][a-z0-9]{2,}")

STOPWORDS = frozenset("""
a about above after again against all also am an and any are aren as at be
because been before being below between both but by can cannot could did do
does doing don down during each few for from further had has have having he
her here hers herself him himself his how into its itself just me more most
my myself nor not now off once only other ought our ours ourselves out over
own same she should so some such than that the their theirs them themselves
then there these they this those through too under until very was we were
what when where which while who whom why will with would you your yours
yourself yourselves content creator covers also get make made new one two
things thing lot really much many best good great love like
""".split())


def tokenise(text) -> list[str]:
    """Lowercase word tokens, stopwords and one/two-letter tokens removed.

    Hashtags survive as their bare word: #cleanbeauty and "clean beauty" should
    not be two unrelated vocabulary entries, but splitting a run-on hashtag is
    guesswork, so #cleanbeauty becomes the single token "cleanbeauty" and a
    brief has to use the hashtag to match it. Stated on the page.
    """
    return [t for t in TOKEN_RE.findall(str(text or "").lower()) if t not in STOPWORDS]


def profile_text(row) -> str:
    """The document that represents a creator in the lexical space.

    Deliberately the same source fields as influencer_profile_text() in
    brandfit.py, plus the presentation bio and category chips that only exist
    in the product layer, because a brand typing a description is describing
    the creator as the product shows them.
    """
    # Parquet gives list columns back as numpy arrays, and `arr or ""` raises
    # rather than falling back, so the sequence case is handled explicitly.
    cats = row.get("categories")
    if isinstance(cats, (list, tuple, np.ndarray)):
        cats = ", ".join(str(c) for c in cats)
    else:
        cats = "" if cats is None or (isinstance(cats, float) and np.isnan(cats)) else str(cats)
    parts = [
        f"{row.get('primary_niche', '')} content creator",
        f"also covers {row.get('secondary_niche', '')}" if row.get("secondary_niche") else "",
        str(row.get("top_keywords", "") or "").replace("|", ", "),
        str(row.get("top_hashtags", "") or "").replace("|", ", "),
        str(row.get("bio", "") or ""),
        cats,
    ]
    return ". ".join(p for p in parts if p).strip()


def build(creators: pd.DataFrame,
          posts: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (creator_terms, vocab).

    `posts` is optional but strongly recommended. Built from the niche, keyword
    and hashtag fields alone the vocabulary is only ~700 terms, because those
    fields are drawn from a small generator lexicon; a brand writing "cruelty
    free" or "monsoon" then matches nothing at all. Folding in the creator's
    actual captions is what gives the typed brief something to land on.
    """
    # to_dict("records") rather than itertuples: profile_text uses .get so it
    # tolerates a column that a rebuild has not produced yet, and a namedtuple
    # has no .get.
    extra: dict[str, str] = {}
    if posts is not None and len(posts):
        extra = (posts.groupby("influencer_id")["caption"]
                 .apply(lambda s: " ".join(s.astype(str)))
                 .to_dict())
        extra = {str(k): v for k, v in extra.items()}

    docs = {}
    for r in creators.to_dict("records"):
        iid = str(r["influencer_id"])
        docs[iid] = tokenise(profile_text(r) + " " + extra.get(iid, ""))
    n_docs = len(docs)

    df = Counter()
    for toks in docs.values():
        df.update(set(toks))

    max_df = max(MIN_DF, int(MAX_DF_SHARE * n_docs))
    keep = {t for t, c in df.items() if MIN_DF <= c <= max_df}

    idf = {t: math.log((1.0 + n_docs) / (1.0 + df[t])) + 1.0 for t in keep}

    rows = []
    for iid, toks in docs.items():
        tf = Counter(t for t in toks if t in keep)
        if not tf:
            continue
        # Sublinear tf: a creator who says "skincare" nine times is not nine
        # times more about skincare than one who says it once.
        w = {t: (1.0 + math.log(c)) * idf[t] for t, c in tf.items()}
        top = sorted(w.items(), key=lambda kv: -kv[1])[:TOP_TERMS]
        norm = math.sqrt(sum(v * v for _, v in top)) or 1.0
        rows.extend({"influencer_id": iid, "term": t, "weight": v / norm}
                    for t, v in top)

    terms = pd.DataFrame(rows)
    terms["weight"] = terms["weight"].astype("float32")

    vocab = pd.DataFrame(
        [{"term": t, "df": int(df[t]), "idf": float(idf[t])} for t in sorted(keep)]
    )
    return terms, vocab


def query_vector(text: str, vocab: pd.DataFrame) -> pd.DataFrame:
    """Turn typed text into the same weighted, L2-normalised representation.

    Shared with the app so the two sides of the cosine cannot drift apart.
    """
    idf = dict(zip(vocab["term"], vocab["idf"]))
    tf = Counter(t for t in tokenise(text) if t in idf)
    if not tf:
        return pd.DataFrame(columns=["term", "weight"])
    w = {t: (1.0 + math.log(c)) * idf[t] for t, c in tf.items()}
    norm = math.sqrt(sum(v * v for v in w.values())) or 1.0
    return pd.DataFrame({"term": list(w), "weight": [v / norm for v in w.values()]})


# ==========================================================================
# Competitor-conflict evidence
# ==========================================================================
#
# The batch fit matrix reads a `competitor_activity` string that only exists in
# the modelling table, which the hosted app does not ship. The intake page has
# to apply the same veto against a competitor list the brand types in, so the
# evidence is exported as its own small table instead: who mentioned which
# brand, how often, and whether any of it was a disclosed paid post.

CONFLICT_WINDOW_DAYS = 180


def build_brand_mentions(posts: pd.DataFrame) -> pd.DataFrame:
    """influencer_id, brand, n_mentions, n_paid, days_ago_min."""
    p = posts.dropna(subset=["gen_brand"]).copy()
    p["brand"] = p["gen_brand"].astype(str).str.strip().str.lower()
    p["paid"] = p["gen_has_promo"].astype(int)
    out = (p.groupby(["influencer_id", "brand"])
             .agg(n_mentions=("brand", "size"),
                  n_paid=("paid", "sum"),
                  days_ago_min=("days_ago", "min"))
             .reset_index())
    out["influencer_id"] = out["influencer_id"].astype(str)
    return out
