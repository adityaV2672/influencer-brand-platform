"""
Download the REAL, human-labelled corpora used to evaluate the NLP methods.

Why this file exists
--------------------
The influencer universe in this project is synthetic. Synthetic text cannot
validate an NLP method: if we generate a caption and label it "sarcastic", then
measure how well a detector recovers that label, we are measuring how well the
detector reverse-engineers our own template, not how well it detects sarcasm.

So every sentiment / emotion / irony claim in the report is measured on real
text written by real people and labelled by real annotators:

  TweetEval (Barbieri et al., Findings of EMNLP 2020)
      - sentiment : 3-class (negative / neutral / positive), SemEval-2017 Task 4
      - emotion   : 4-class (anger / joy / optimism / sadness), SemEval-2018 Task 1
      - irony     : binary irony, SemEval-2018 Task 3
      HF hub: cardiffnlp/tweet_eval
      Paper : https://aclanthology.org/2020.findings-emnlp.148/

  News Headlines Dataset for Sarcasm Detection (Misra & Arora)
      - binary sarcasm, headlines from The Onion vs HuffPost
      Paper : https://arxiv.org/abs/2212.06035

Using two independent sarcasm corpora from different domains (social media vs
news headlines) lets us report cross-domain generalisation rather than a single
in-domain number, which is a much harder and more honest test.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd

from src.config import BENCHMARK_DIR

# --------------------------------------------------------------------------
# Provenance, emitted into the report
# --------------------------------------------------------------------------
CITATIONS = {
    "tweeteval": {
        "name": "TweetEval",
        "authors": "Barbieri, F., Camacho-Collados, J., Espinosa-Anke, L., Neves, L.",
        "title": "TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification",
        "venue": "Findings of EMNLP 2020",
        "url": "https://aclanthology.org/2020.findings-emnlp.148/",
        "hub": "cardiffnlp/tweet_eval",
        "license": "See dataset card; derived from SemEval shared tasks.",
    },
    "sarcasm_headlines": {
        "name": "News Headlines Dataset for Sarcasm Detection",
        "authors": "Misra, R., Arora, P.",
        "title": "Sarcasm Detection using News Headlines Dataset",
        "venue": "AI Open / arXiv:2212.06035",
        "url": "https://arxiv.org/abs/2212.06035",
        "license": "CC BY 4.0 (per author's distribution)",
    },
    "nrc": {
        "name": "NRC Word-Emotion Association Lexicon (EmoLex)",
        "authors": "Mohammad, S. M., Turney, P. D.",
        "title": "Crowdsourcing a Word-Emotion Association Lexicon",
        "venue": "Computational Intelligence, 29(3), 2013",
        "url": "http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm",
        "license": "Free for research use.",
    },
    "vader": {
        "name": "VADER",
        "authors": "Hutto, C. J., Gilbert, E.",
        "title": "VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text",
        "venue": "ICWSM 2014",
        "url": "https://ojs.aaai.org/index.php/ICWSM/article/view/14550",
        "license": "MIT",
    },
    "sbert": {
        "name": "Sentence-BERT",
        "authors": "Reimers, N., Gurevych, I.",
        "title": "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
        "venue": "EMNLP 2019",
        "url": "https://aclanthology.org/D19-1410/",
        "license": "Apache-2.0 (model weights)",
    },
    "bertopic": {
        "name": "BERTopic",
        "authors": "Grootendorst, M.",
        "title": "BERTopic: Neural topic modeling with a class-based TF-IDF procedure",
        "venue": "arXiv:2203.05794",
        "url": "https://arxiv.org/abs/2203.05794",
        "license": "MIT",
    },
}

# Mirrors of the Misra & Arora release, most reliable first. The HuggingFace
# dataset repo `raquiba/Sarcasm_News_Headline` hosts the same records as
# train.json / test.json and needs no auth token.
SARCASM_HEADLINES_URLS = [
    "https://huggingface.co/datasets/raquiba/Sarcasm_News_Headline/resolve/main/train.json",
    "https://huggingface.co/datasets/raquiba/Sarcasm_News_Headline/resolve/main/test.json",
]


def _save(df: pd.DataFrame, name: str) -> Path:
    path = BENCHMARK_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    print(f"  saved {name:22s} {len(df):>7,} rows -> {path.name}")
    return path


# --------------------------------------------------------------------------
# TweetEval
# --------------------------------------------------------------------------

TWEETEVAL_LABELS = {
    "sentiment": ["negative", "neutral", "positive"],
    "emotion": ["anger", "joy", "optimism", "sadness"],
    "irony": ["non_irony", "irony"],
}


# The authors publish the raw splits as plain text in their own repository.
# We prefer this over the HuggingFace hub mirror: it is the primary source, it
# needs no auth token, and it works behind restrictive proxies.
TWEETEVAL_REPO = "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets"
_SPLIT_FILES = {"train": "train", "validation": "val", "test": "test"}


def _get_text(url: str, timeout: int = 60) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode("utf-8")
    except Exception:  # noqa: BLE001
        return None


def _fetch_tweeteval_github(sub: str) -> pd.DataFrame | None:
    frames = []
    for split, stem in _SPLIT_FILES.items():
        texts = _get_text(f"{TWEETEVAL_REPO}/{sub}/{stem}_text.txt")
        labels = _get_text(f"{TWEETEVAL_REPO}/{sub}/{stem}_labels.txt")
        if texts is None or labels is None:
            return None
        tl = texts.splitlines()
        ll = [int(x) for x in labels.split()]
        if len(tl) != len(ll):
            print(f"    ! {sub}/{split}: {len(tl)} texts vs {len(ll)} labels, skipping")
            return None
        frames.append(pd.DataFrame({"text": tl, "label_id": ll, "split": split}))
    return pd.concat(frames, ignore_index=True) if frames else None


def _fetch_tweeteval_hf(sub: str) -> pd.DataFrame | None:
    try:
        from datasets import load_dataset
    except ImportError:
        return None
    frames = []
    for split in ("train", "validation", "test"):
        try:
            ds = load_dataset("cardiffnlp/tweet_eval", sub, split=split)
        except Exception:  # noqa: BLE001
            return None
        frames.append(pd.DataFrame({"text": ds["text"], "label_id": ds["label"], "split": split}))
    return pd.concat(frames, ignore_index=True) if frames else None


def fetch_tweeteval(subsets: tuple[str, ...] = ("sentiment", "emotion", "irony")) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for sub in subsets:
        full = _fetch_tweeteval_github(sub)
        source = "github (authors' repo)"
        if full is None:
            full = _fetch_tweeteval_hf(sub)
            source = "huggingface hub"
        if full is None:
            print(f"  ! {sub}: all sources failed")
            continue
        names = TWEETEVAL_LABELS[sub]
        full["label"] = full["label_id"].map(dict(enumerate(names)))
        full = full.dropna(subset=["text", "label"])
        full["text"] = full["text"].astype(str).str.strip()
        full = full[full["text"].str.len() > 0].reset_index(drop=True)
        out[sub] = full
        print(f"  [{source}]", end=" ")
        _save(full, f"tweeteval_{sub}")
    return out


# --------------------------------------------------------------------------
# News-headline sarcasm
# --------------------------------------------------------------------------


def _parse_json_records(raw: str) -> list[dict]:
    """The file is distributed both as a JSON array and as newline-delimited JSON."""
    try:
        obj = json.loads(raw)
        return [obj] if isinstance(obj, dict) else list(obj)
    except json.JSONDecodeError:
        return [json.loads(line) for line in raw.splitlines() if line.strip()]


def fetch_sarcasm_headlines() -> pd.DataFrame | None:
    frames = []
    for url in SARCASM_HEADLINES_URLS:
        split = "train" if "train" in url.rsplit("/", 1)[-1] else "test"
        raw = _get_text(url, timeout=90)
        if raw is None:
            print(f"    {split}: unreachable")
            continue
        try:
            df = pd.DataFrame(_parse_json_records(raw))
        except Exception as exc:  # noqa: BLE001
            print(f"    {split}: parse failed ({exc})")
            continue

        cols = {c.lower(): c for c in df.columns}
        text_col = cols.get("headline") or cols.get("text")
        label_col = cols.get("is_sarcastic") or cols.get("label")
        if not text_col or not label_col:
            print(f"    {split}: unexpected schema {list(df.columns)}")
            continue

        df = df.rename(columns={text_col: "text", label_col: "label_id"})
        df["label_id"] = pd.to_numeric(df["label_id"], errors="coerce")
        df = df.dropna(subset=["text", "label_id"])
        df["label_id"] = df["label_id"].astype(int)
        df["label"] = df["label_id"].map({0: "non_sarcastic", 1: "sarcastic"})
        df["split"] = split
        frames.append(df[["text", "label_id", "label", "split"]])
        print(f"    {split}: {len(df):,} rows")

    if not frames:
        print("  ! all mirrors failed for the sarcasm headlines dataset")
        return None

    full = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["text"])
    _save(full, "sarcasm_headlines")
    return full


# --------------------------------------------------------------------------


def fetch_all() -> dict[str, pd.DataFrame]:
    print("Fetching real labelled benchmark corpora")
    print("-" * 60)
    out: dict[str, pd.DataFrame] = {}
    print("TweetEval (Barbieri et al., EMNLP Findings 2020):")
    out.update(fetch_tweeteval())
    print("News headlines sarcasm (Misra & Arora):")
    got = fetch_sarcasm_headlines()
    if got is not None:
        out["sarcasm_headlines"] = got

    (BENCHMARK_DIR / "CITATIONS.json").write_text(json.dumps(CITATIONS, indent=2))
    print("-" * 60)
    print(f"{len(out)} corpora available in {BENCHMARK_DIR}")
    return out


if __name__ == "__main__":
    fetch_all()
