"""
The models. These are real: really fitted, really cross-validated, really
ablated. Only their inputs are simulated.

Architecture - late fusion, three arms
--------------------------------------
    text branch    TF-IDF(caption + ASR transcript) -> logistic regression
    audio branch   prosody embedding (128-d)        -> logistic regression
    fusion         [text probs | audio probs | ASR quality] -> logistic regression

Late fusion rather than early: the two branches carry different amounts of
information per post and the fusion has to learn WHEN to trust which. Early
fusion (concatenating TF-IDF with a dense embedding) hands a linear model a
5,000-dimensional sparse block next to a 128-dimensional dense one and the
dense block is drowned. Late fusion also means the branches can be evaluated
on their own, which is what makes the ablation table meaningful.

Every split is GroupKFold on influencer_id. A random split would let the audio
branch memorise voices - speaker identity is deliberately the largest direction
in the embedding - and score high without reading affect at all. That is the
single most important methodological choice in this file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.config import SEED

N_SPLITS = 5
LABELS = ("negative", "neutral", "positive")

ASR_QUALITY_COLS = [
    "asr_mean_confidence", "asr_low_conf_share", "asr_filler_rate",
    "asr_words_per_min", "asr_n_words",
]


def _scores(y_true, y_pred) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted")), 4),
    }


def _oof_proba(fit_predict, X, y, groups) -> np.ndarray:
    """Out-of-fold probabilities, so the fusion never sees a branch's
    in-sample confidence. Training the fusion on in-sample probabilities is
    the classic stacking leak: the branches look far more reliable on the
    training rows than they will ever be at serving time, and the fusion
    learns to trust them too much."""
    out = np.zeros((len(y), len(LABELS)))
    for tr, te in GroupKFold(n_splits=N_SPLITS).split(X, y, groups):
        out[te] = fit_predict(X, y, tr, te)
    return out


def text_branch(df: pd.DataFrame, y, groups, max_features: int = 6000):
    """TF-IDF over the caption AND the ASR transcript."""
    docs = (df["caption"].fillna("") + " \n " + df["transcript"].fillna("")).to_numpy()

    def fit_predict(X, y, tr, te):
        vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2),
                              min_df=3, sublinear_tf=True)
        Xtr = vec.fit_transform(X[tr])
        clf = LogisticRegression(max_iter=1500, C=2.0, random_state=SEED)
        clf.fit(Xtr, y[tr])
        return clf.predict_proba(vec.transform(X[te]))

    return _oof_proba(fit_predict, docs, y, groups)


def audio_branch(emb: np.ndarray, y, groups):
    """The speech-emotion head on top of the frozen prosody encoder.

    A linear head on a frozen self-supervised encoder is exactly how wav2vec2
    and HuBERT are used for emotion in practice - the encoder is not fine-tuned
    on a corpus this size - so the shape of this model is right even though the
    encoder underneath it is simulated.
    """
    def fit_predict(X, y, tr, te):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1500, C=1.0, random_state=SEED)
        clf.fit(sc.transform(X[tr]), y[tr])
        return clf.predict_proba(sc.transform(X[te]))

    return _oof_proba(fit_predict, emb, y, groups)


def fusion(text_p: np.ndarray, audio_p: np.ndarray, quality: pd.DataFrame,
           y, groups):
    """Late fusion. ASR quality is included so the model can learn to fall
    back on prosody when the transcript is unreliable - which is the whole
    argument for having two modalities."""
    X = np.hstack([text_p, audio_p, quality[ASR_QUALITY_COLS].to_numpy(float)])

    def fit_predict(X, y, tr, te):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
        clf.fit(sc.transform(X[tr]), y[tr])
        return clf.predict_proba(sc.transform(X[te]))

    return _oof_proba(fit_predict, X, y, groups)


def evaluate(name: str, proba: np.ndarray, y, sarcastic: np.ndarray) -> dict:
    """Subgroup scores use ACCURACY, not macro F1.

    The sarcastic subset is nearly one class, so a three-class macro F1 over it
    is capped near 0.33 whatever the model does - the first version of this
    reported 0.3310 for all three arms and it read as total failure when it was
    in fact near-perfect. Accuracy on the subset says the thing that was meant.
    """
    pred = np.array(LABELS)[proba.argmax(axis=1)]
    out = {"arm": name, **_scores(y, pred)}
    out["accuracy_sarcastic"] = round(
        float(accuracy_score(y[sarcastic], pred[sarcastic])), 4) if sarcastic.any() else None
    out["accuracy_sincere"] = round(
        float(accuracy_score(y[~sarcastic], pred[~sarcastic])), 4)
    return out


def majority_baseline(y) -> dict:
    lab = pd.Series(y).value_counts().idxmax()
    pred = np.full(len(y), lab)
    return {"arm": "majority baseline", **_scores(y, pred),
            "accuracy_sarcastic": None, "accuracy_sincere": None}
