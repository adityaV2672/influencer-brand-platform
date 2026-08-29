"""
Comment-level NLP. The models here are trained on REAL human-labelled data.

The one honest anchor in this feature
-------------------------------------
The comment corpus is generated (see comments.py). The classifiers applied to
it are not: both are fitted on TweetEval, a benchmark of real tweets annotated
by real people, and both are scored on TweetEval's own held-out test split
before they are ever pointed at Nectar's comments.

    sentiment    TweetEval sentiment  (SemEval-2017 Task 4)  3 classes
    toxicity     TweetEval offensive  (SemEval-2019 Task 6)  2 classes

That matters because it means the numbers this module reports about ITSELF -
its accuracy, its macro F1 - describe performance on real human judgements
about real short social text. Comments are short social text. Applying a model
validated on tweets to Instagram comments is a domain shift and the module
says so; it is not the same as validating on the target domain, and the report
should not claim it is.

The third label - whether a comment looks automated - is NOT learned from
TweetEval, because no such corpus is available here. It is a transparent rule
over surface features (emoji-only, follow-for-follow cues, template
duplication), and it is labelled as a rule everywhere it appears.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

from src.config import ARTIFACT_DIR, DATA_DIR, SEED

BENCH = DATA_DIR / "benchmarks"
OUT = ARTIFACT_DIR / "comment_nlp"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = OUT / "comment_model_results.json"

# Automation cues. Deliberately conservative: a short comment is not evidence
# of a bot on its own, or half of every real comment section would be flagged.
BOT_CUES = ("follow for follow", "follow back", "dm for", "dm me", "link in bio",
            "check bio", "free followers", "grow your page", "support back",
            "check out my page", "drop a like", "lets grow together")


def _fit_task(name: str, file: str) -> tuple[object, dict]:
    """Train on the real train split, score on the real test split."""
    df = pd.read_parquet(BENCH / file)
    tr = df[df.split == "train"]
    te = df[df.split == "test"]
    model = make_pipeline(
        TfidfVectorizer(max_features=40_000, ngram_range=(1, 2), min_df=2,
                        sublinear_tf=True, strip_accents="unicode"),
        LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced",
                           random_state=SEED),
    )
    model.fit(tr.text, tr.label)
    pred = model.predict(te.text)
    metrics = {
        "task": name,
        "corpus": file.replace(".parquet", ""),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "accuracy": round(float(accuracy_score(te.label, pred)), 4),
        "macro_f1": round(float(f1_score(te.label, pred, average="macro")), 4),
        "majority_baseline_accuracy": round(
            float((te.label == tr.label.value_counts().idxmax()).mean()), 4),
        "labels": sorted(df.label.unique().tolist()),
        "provenance": "REAL - trained and evaluated on human-annotated tweets",
    }
    return model, metrics


def train() -> tuple[dict, dict]:
    """Returns ({task: model}, results)."""
    models, results = {}, {"models": [], "caveats": {
        "domain_shift": "Both classifiers are trained and validated on tweets, "
                        "then applied to Instagram comments. Comments are "
                        "shorter, more emoji-heavy and more formulaic than "
                        "tweets, so real-world accuracy on comments will be "
                        "LOWER than the figures reported here. No labelled "
                        "comment corpus was available to measure the drop.",
        "automation_flag": "The bot/automation label is a transparent rule over "
                           "surface cues, not a learned model. It is not "
                           "validated against any human-labelled corpus.",
        "corpus": "The comments the models are applied to are SIMULATED "
                  "(src/creator_data/comments.py). The models are not.",
    }}
    for name, file in (("sentiment", "tweeteval_sentiment.parquet"),
                       ("toxicity", "tweeteval_offensive.parquet")):
        m, r = _fit_task(name, file)
        models[name] = m
        results["models"].append(r)
        print(f"    {name:10} on {r['corpus']:24} acc {r['accuracy']:.4f}  "
              f"macroF1 {r['macro_f1']:.4f}  (majority {r['majority_baseline_accuracy']:.4f})")
    RESULTS.write_text(json.dumps(results, indent=2))
    return models, results


def looks_automated(text: pd.Series, is_emoji_only: pd.Series) -> pd.Series:
    """Rule, not a model. A solicitation cue, or nothing but emoji.

    Exact-duplicate text was in this rule and had to come out. Real comment
    sections repeat themselves constantly - "love this" appears hundreds of
    times under a popular creator - so duplication flagged 70% of all comments
    as automated, which is not a detector, it is a description of how people
    write. Duplication survives as its own RATE feature, where the model can
    weigh it against everything else instead of it acting as a veto.
    """
    low = text.astype(str).str.lower()
    cue = low.apply(lambda s: any(c in s for c in BOT_CUES))
    return cue | is_emoji_only.fillna(False)


def score_comments(comments: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Attach model output to every comment."""
    c = comments.copy()
    c["comment_sentiment"] = models["sentiment"].predict(c.text)
    proba = models["sentiment"].predict_proba(c.text)
    classes = list(models["sentiment"].classes_)
    c["p_positive"] = proba[:, classes.index("positive")].round(4)
    c["p_negative"] = proba[:, classes.index("negative")].round(4)
    c["comment_toxicity"] = models["toxicity"].predict(c.text)
    tproba = models["toxicity"].predict_proba(c.text)
    tclasses = list(models["toxicity"].classes_)
    c["p_offensive"] = tproba[:, tclasses.index("offensive")].round(4)

    c["looks_automated"] = looks_automated(c.text, c.is_emoji_only)
    c["is_duplicate_text"] = c.duplicated(subset=["influencer_id", "text"],
                                          keep=False)
    return c


def aggregate(scored: pd.DataFrame) -> pd.DataFrame:
    """One row per creator: what their comment section looks like."""
    g = scored.groupby("influencer_id")
    out = pd.DataFrame({
        "n_comments_analysed": g.size(),
        "comment_sentiment_positive": g.comment_sentiment.apply(
            lambda s: float((s == "positive").mean())),
        "comment_sentiment_negative": g.comment_sentiment.apply(
            lambda s: float((s == "negative").mean())),
        "comment_p_positive_mean": g.p_positive.mean(),
        "comment_toxicity_rate": g.comment_toxicity.apply(
            lambda s: float((s == "offensive").mean())),
        "comment_automated_rate": g.looks_automated.mean(),
        "comment_duplicate_rate": g.is_duplicate_text.mean(),
        "comment_emoji_only_rate": g.is_emoji_only.mean(),
        "comment_generic_rate": g.is_generic.mean(),
        "comment_mean_words": g.n_words.mean(),
        "comment_link_cue_rate": g.has_link_cue.mean(),
    }).round(4).reset_index()
    # A comment section worth having: long, positive, human, non-repetitive.
    out["comment_quality_index"] = (
        0.35 * (1 - out.comment_automated_rate)
        + 0.25 * out.comment_sentiment_positive
        + 0.20 * (1 - out.comment_toxicity_rate)
        + 0.20 * np.clip(out.comment_mean_words / 8.0, 0, 1)
    ).round(4)
    return out
