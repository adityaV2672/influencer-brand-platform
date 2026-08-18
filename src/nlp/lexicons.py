"""
Lexicon-based methods: Bing (Hu & Liu), VADER, and NRC EmoLex.

These are the "primitive" end of the spectrum. They are included deliberately,
not as filler: the project's headline empirical finding is *how badly* word-list
methods fail on irony, and you cannot demonstrate that without running them.

Ordering by sophistication:
  Bing  - pure positive/negative word membership, no negation, no intensity
  VADER - valence-scored words + negation, intensifiers, punctuation, caps, emoji
  NRC   - 8 emotions + 2 polarities, richer affect but still bag-of-words
"""
from __future__ import annotations

import json
import re
import urllib.request
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.config import RAW_DIR
from src.nlp.base import MethodMeta, TextMethod

_TOKEN_RE = re.compile(r"[a-zA-Z']+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# ==========================================================================
# Bing / Hu & Liu opinion lexicon
# ==========================================================================

_BING_URLS = {
    "positive": [
        "https://raw.githubusercontent.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107/master/data/opinion-lexicon-English/positive-words.txt",
        "https://raw.githubusercontent.com/shekhargulati/sentiment-analysis-python/master/opinion-lexicon-English/positive-words.txt",
    ],
    "negative": [
        "https://raw.githubusercontent.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107/master/data/opinion-lexicon-English/negative-words.txt",
        "https://raw.githubusercontent.com/shekhargulati/sentiment-analysis-python/master/opinion-lexicon-English/negative-words.txt",
    ],
}


@lru_cache(maxsize=1)
def load_bing() -> tuple[frozenset[str], frozenset[str]]:
    """Download (once) and cache the Hu & Liu opinion lexicon."""
    out = {}
    for polarity, urls in _BING_URLS.items():
        cache = RAW_DIR / f"bing_{polarity}.txt"
        if cache.exists():
            words = cache.read_text(encoding="latin-1")
        else:
            words = None
            for url in urls:
                try:
                    with urllib.request.urlopen(url, timeout=45) as r:
                        words = r.read().decode("latin-1")
                    cache.write_text(words, encoding="latin-1")
                    break
                except Exception:  # noqa: BLE001
                    continue
            if words is None:
                raise RuntimeError(
                    f"Could not download the Bing {polarity} lexicon. "
                    f"Place the file manually at {cache}."
                )
        toks = {
            ln.strip().lower()
            for ln in words.splitlines()
            if ln.strip() and not ln.startswith(";")
        }
        out[polarity] = frozenset(toks)
    return out["positive"], out["negative"]


class BingSentiment(TextMethod):
    """Count positive vs negative words. No negation handling whatsoever.

    This is the method the project brief describes as the Python default and
    the one the supervisor flagged as primitive. It is the floor of the
    comparison.
    """

    meta = MethodMeta(
        name="Bing (Hu & Liu opinion lexicon)",
        family="lexicon",
        supervised=False,
        citation="Hu, M. & Liu, B. (2004). Mining and Summarizing Customer Reviews. KDD 2004.",
        notes="Pure word membership. No negation, intensifiers, or context.",
    )
    _classes = ["negative", "neutral", "positive"]

    def __init__(self, neutral_band: float = 0.0):
        self.pos, self.neg = load_bing()
        self.neutral_band = neutral_band

    def raw_score(self, text: str) -> float:
        toks = tokenize(text)
        if not toks:
            return 0.0
        p = sum(1 for t in toks if t in self.pos)
        n = sum(1 for t in toks if t in self.neg)
        if p + n == 0:
            return 0.0
        return (p - n) / (p + n)

    def predict(self, texts: list[str]) -> list[str]:
        out = []
        for t in texts:
            s = self.raw_score(t)
            if s > self.neutral_band:
                out.append("positive")
            elif s < -self.neutral_band:
                out.append("negative")
            else:
                out.append("neutral")
        return out


# ==========================================================================
# VADER
# ==========================================================================


class VaderSentiment(TextMethod):
    """VADER compound score thresholded into three classes.

    Thresholds 0.05 / -0.05 are the values recommended by the VADER authors.
    """

    meta = MethodMeta(
        name="VADER",
        family="lexicon",
        supervised=False,
        citation="Hutto, C.J. & Gilbert, E. (2014). VADER. ICWSM 2014.",
        notes="Valence-scored lexicon with negation, intensifier, caps and emoji rules. "
              "Tuned for social media.",
        params={"pos_threshold": 0.05, "neg_threshold": -0.05},
    )
    _classes = ["negative", "neutral", "positive"]

    def __init__(self, pos_threshold: float = 0.05, neg_threshold: float = -0.05):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self._an = SentimentIntensityAnalyzer()
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold

    def raw_score(self, text: str) -> float:
        return self._an.polarity_scores(text)["compound"]

    def scores(self, texts: list[str]) -> np.ndarray:
        return np.array([self.raw_score(t) for t in texts])

    def predict(self, texts: list[str]) -> list[str]:
        out = []
        for t in texts:
            c = self.raw_score(t)
            if c >= self.pos_threshold:
                out.append("positive")
            elif c <= self.neg_threshold:
                out.append("negative")
            else:
                out.append("neutral")
        return out


# ==========================================================================
# NRC EmoLex
# ==========================================================================

NRC_EMOTIONS = [
    "anger", "anticipation", "disgust", "fear",
    "joy", "sadness", "surprise", "trust",
]
NRC_POLARITIES = ["positive", "negative"]

_NRC_URLS = [
    "https://raw.githubusercontent.com/metalcorebear/NRCLex/master/nrclex/nrc_en.json",
    "https://raw.githubusercontent.com/metalcorebear/NRCLex/master/nrc_en.json",
]


@lru_cache(maxsize=1)
def load_nrc() -> dict[str, list[str]]:
    """Word -> list of associated affect categories."""
    cache = RAW_DIR / "nrc_en.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    # Preferred: the copy bundled with the installed nrclex package (no network).
    try:
        import nrclex  # type: ignore

        pkg_root = Path(nrclex.__file__).parent
        for candidate in (pkg_root / "data" / "nrc_en.json", pkg_root / "nrc_en.json"):
            if candidate.exists():
                data = json.loads(candidate.read_text(encoding="utf-8"))
                cache.write_text(json.dumps(data), encoding="utf-8")
                return data
    except Exception:  # noqa: BLE001
        pass

    for url in _NRC_URLS:
        try:
            with urllib.request.urlopen(url, timeout=45) as r:
                data = json.loads(r.read().decode("utf-8"))
            cache.write_text(json.dumps(data), encoding="utf-8")
            return data
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError(
        "Could not obtain the NRC Emotion Lexicon. Install `nrclex` or place "
        f"nrc_en.json at {cache}."
    )


class NRCAffect:
    """Shared NRC scorer used by both the sentiment and emotion wrappers."""

    def __init__(self):
        self.lex = load_nrc()

    def counts(self, text: str) -> Counter:
        c: Counter = Counter()
        for tok in tokenize(text):
            for cat in self.lex.get(tok, ()):
                c[cat] += 1
        return c

    def emotion_vector(self, text: str, normalise: bool = True) -> np.ndarray:
        c = self.counts(text)
        v = np.array([c.get(e, 0) for e in NRC_EMOTIONS], dtype=float)
        if normalise and v.sum() > 0:
            v = v / v.sum()
        return v


class NRCSentiment(TextMethod):
    """NRC positive/negative word counts, thresholded into three classes.

    This is the method the supervisor referred to as "NRC (R uses it)" - it is
    the affect lexicon behind the `syuzhet` / `tidytext` NRC option in R.
    """

    meta = MethodMeta(
        name="NRC EmoLex (polarity)",
        family="lexicon",
        supervised=False,
        citation="Mohammad, S.M. & Turney, P.D. (2013). Crowdsourcing a Word-Emotion "
                 "Association Lexicon. Computational Intelligence 29(3).",
        notes="Bag-of-words polarity counts from the NRC lexicon. Richer vocabulary "
              "than Bing but still no negation or context handling.",
    )
    _classes = ["negative", "neutral", "positive"]

    def __init__(self):
        self.nrc = NRCAffect()

    def raw_score(self, text: str) -> float:
        c = self.nrc.counts(text)
        p, n = c.get("positive", 0), c.get("negative", 0)
        return 0.0 if p + n == 0 else (p - n) / (p + n)

    def predict(self, texts: list[str]) -> list[str]:
        out = []
        for t in texts:
            s = self.raw_score(t)
            out.append("positive" if s > 0 else "negative" if s < 0 else "neutral")
        return out


class NRCEmotion(TextMethod):
    """NRC 8-emotion classification: argmax over the emotion counts.

    TweetEval-emotion only labels 4 of the 8 NRC categories
    (anger / joy / optimism / sadness), so for benchmarking we map:
        anticipation + trust -> optimism
        disgust + fear       -> folded into anger (closest available label)
        surprise             -> dropped (no corresponding label)
    That mapping is lossy and is reported as such - it is a genuine limitation
    of scoring an 8-way lexicon against a 4-way benchmark, not a bug.
    """

    meta = MethodMeta(
        name="NRC EmoLex (8-emotion)",
        family="lexicon",
        supervised=False,
        citation="Mohammad, S.M. & Turney, P.D. (2013). Computational Intelligence 29(3).",
        notes="Argmax over NRC emotion counts, mapped onto the 4 TweetEval emotion labels.",
    )
    _classes = ["anger", "joy", "optimism", "sadness"]

    TWEETEVAL_MAP = {
        "anger": "anger",
        "disgust": "anger",
        "fear": "anger",
        "joy": "joy",
        "anticipation": "optimism",
        "trust": "optimism",
        "sadness": "sadness",
        "surprise": None,
    }

    def __init__(self):
        self.nrc = NRCAffect()

    def predict_native(self, texts: list[str]) -> list[str]:
        """Predict over the full 8 NRC emotions (used for feature extraction)."""
        out = []
        for t in texts:
            c = self.nrc.counts(t)
            scores = {e: c.get(e, 0) for e in NRC_EMOTIONS}
            best = max(scores, key=lambda k: scores[k])
            out.append(best if scores[best] > 0 else "neutral")
        return out

    def predict(self, texts: list[str]) -> list[str]:
        out = []
        for t in texts:
            c = self.nrc.counts(t)
            agg: Counter = Counter()
            for emo, mapped in self.TWEETEVAL_MAP.items():
                if mapped:
                    agg[mapped] += c.get(emo, 0)
            if not agg or max(agg.values()) == 0:
                out.append("joy")           # majority-class fallback
            else:
                out.append(max(agg, key=lambda k: agg[k]))
        return out
