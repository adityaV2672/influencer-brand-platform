"""
UCI Online News Popularity: 39,644 real Mashable articles and their real share
counts.

Why this dataset is in this project
-----------------------------------
Every predictive number elsewhere in this repository is measured on data the
project generated for itself. That makes the modelling pipeline demonstrable
and the RESULTS unfalsifiable - a model trained to predict a target drawn from
a known equation is inverting the equation, and its R^2 says nothing about
influencer marketing.

This dataset has the same problem shape as the performance model - predict how
widely a piece of content travels, from features of the content - and none of
that problem. The articles are real, the share counts are real, and the same
pipeline (log target, Duan smearing, honest baselines, held-out scoring) runs
over it unchanged. Whatever it scores is the first predictive number in this
project that describes the world.

Provenance and its one weakness
-------------------------------
Canonical home: UCI Machine Learning Repository, dataset 332, from

    Fernandes, K., Vinagre, P., Cortez, P. (2015). A Proactive Intelligent
    Decision Support System for Predicting the Popularity of Online News.
    Proceedings of the 17th EPIA, Portuguese Conference on Artificial
    Intelligence, Coimbra, Portugal.

archive.ics.uci.edu is not reachable from this environment, so the file is
fetched from GitHub mirrors instead. That is a real weakness and it is handled
rather than ignored: the fetcher requires the bytes to match a recorded
SHA-256, and that digest was established by downloading from THREE unrelated
mirrors and confirming all three were byte-identical. Shape and schema are
checked against the published description as a second, independent test.
"""
from __future__ import annotations

import hashlib
import io
import json
import urllib.request
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR

REAL_DIR = DATA_DIR / "real"
REAL_DIR.mkdir(parents=True, exist_ok=True)
CSV = REAL_DIR / "OnlineNewsPopularity.csv"
PROVENANCE = REAL_DIR / "news_popularity_provenance.json"

# Byte-identical across all three mirrors below, verified 2026-08-29.
SHA256 = "b66d9088632308cc27fa35af847650d174a5a50503987c4e511de94a99d1c218"

MIRRORS = [
    "https://raw.githubusercontent.com/susobhang70/OnlineNewsPopularity/master/OnlineNewsPopularity.csv",
    "https://raw.githubusercontent.com/adrian-ramirezc/online-news-popularity/main/OnlineNewsPopularity.csv",
    "https://raw.githubusercontent.com/ymdong/MLND-Online-News-Popularity-Prediction/master/OnlineNewsPopularity.csv",
]

CITATION = ("Fernandes, K., Vinagre, P., Cortez, P. (2015). A Proactive Intelligent "
            "Decision Support System for Predicting the Popularity of Online News. "
            "EPIA 2015. UCI Machine Learning Repository, dataset 332.")

# Published description: 39,644 instances, 61 attributes (58 predictive,
# 2 non-predictive, 1 target).
EXPECTED_SHAPE = (39644, 61)
NON_PREDICTIVE = ["url", "timedelta"]
TARGET = "shares"

# The paper frames popularity as a binary task at the median share count.
# Reproducing that threshold is what makes any comparison to published
# accuracy figures legitimate.
POPULAR_THRESHOLD = 1400


def _digest(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fetch(force: bool = False) -> Path:
    """Download once, verify, cache. Refuses to write bytes it cannot verify."""
    if CSV.exists() and not force and _digest(CSV.read_bytes()) == SHA256:
        return CSV

    errors = []
    for url in MIRRORS:
        try:
            raw = urllib.request.urlopen(
                urllib.request.Request(url, headers={"User-Agent": "research/1.0"}),
                timeout=90).read()
        except Exception as exc:                                # noqa: BLE001
            errors.append(f"{url.split('/')[3]}: {type(exc).__name__}")
            continue
        got = _digest(raw)
        if got != SHA256:
            # A mirror that has drifted is worse than a mirror that is down,
            # because it fails silently. Skip it and say so.
            errors.append(f"{url.split('/')[3]}: digest {got[:12]} != expected")
            continue
        CSV.write_bytes(raw)
        PROVENANCE.write_text(json.dumps({
            "dataset": "UCI Online News Popularity (id 332)",
            "citation": CITATION,
            "canonical_host": "https://archive.ics.uci.edu/dataset/332/"
                              "online+news+popularity",
            "fetched_from": url,
            "reason_not_canonical": "archive.ics.uci.edu is unreachable from this "
                                    "environment; the digest below was agreed by "
                                    "three unrelated GitHub mirrors.",
            "mirrors_checked": MIRRORS,
            "sha256": SHA256,
            "bytes": len(raw),
            "provenance": "REAL - human-authored articles, observed share counts. "
                          "Nothing in this file is generated by this project.",
        }, indent=2))
        return CSV

    raise RuntimeError("Could not fetch a verified copy. " + "; ".join(errors))


def load() -> pd.DataFrame:
    """The verified table, with its schema checked against the publication."""
    df = pd.read_csv(fetch())
    df.columns = [c.strip() for c in df.columns]

    if df.shape != EXPECTED_SHAPE:
        raise ValueError(f"shape {df.shape} != published {EXPECTED_SHAPE}")
    missing = [c for c in NON_PREDICTIVE + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"schema does not match the publication: missing {missing}")
    if df.isna().any().any():
        raise ValueError("the published dataset has no missing values; this copy does")
    return df


def features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """X, y. `url` and `timedelta` are dropped because the publication marks
    them non-predictive - timedelta is days between publication and
    acquisition, so it encodes the collection window rather than anything a
    publisher could know in advance."""
    X = df.drop(columns=NON_PREDICTIVE + [TARGET])
    return X, df[TARGET]


def publication_day(df: pd.DataFrame) -> pd.Series:
    """Recover an ordering in time from the non-predictive `timedelta`.

    Used ONLY to build a chronological split, never as a feature. An article
    with a larger timedelta was published earlier, so ranking by -timedelta
    orders the corpus from oldest to newest.
    """
    return (-df["timedelta"]).rank(method="dense").astype(int)
