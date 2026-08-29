"""
Audience quality: is this creator's following real?

This is the single thing brands most want from an influencer agency, and the
platform had no signal for it at all before this module. Bought followers and
engagement pods are the industry's central fraud problem: a creator with
200,000 purchased followers looks identical to a real one on every public
number except the ones computed here.

What is predicted
-----------------
The share of a creator's comment section written by bots, engagement pods and
spam accounts, bucketed into three bands. The truth comes from the comment
generator's archetype labels, which the model NEVER sees.

Two arms, and the comparison between them is the point
------------------------------------------------------
    account_only    follower/following ratio, engagement versus the tier
                    benchmark, comment-to-like ratio, growth rate. These are
                    the signals anyone can compute from a public profile, and
                    they are what the "fake follower check" tools of the last
                    decade were built on.

    with_comments   the above plus what the comment section looks like -
                    automation cues, duplicate text, emoji-only share, mean
                    comment length.

The honest caveat, stated plainly: in THIS corpus the bot archetype writes
cue-laden text and the automation rule catches it, so the with_comments arm is
advantaged by construction. The gap between the arms is therefore an upper
bound, not an estimate. What is not by construction is that the account_only
arm struggles - and that is a real finding about the folk heuristics, because
those features are generated independently of the comment mix.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.config import ARTIFACT_DIR, SEED

OUT = ARTIFACT_DIR / "audience_quality"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = OUT / "audience_quality_results.json"

BANDS = ["Authentic", "Mixed", "Suspect"]
# Cut points on the inauthentic share of the comment section. Set from what the
# industry treats as actionable: under 10% is normal background noise, over
# 25% is the level at which a brand should refuse the creator.
# Even a clean comment section carries junk: the generator's authentic mix is
# already 8% bot, pod and spam, which is realistic - no real account is at
# zero. Edges set at 10 and 25 therefore put almost every creator in the
# bottom two bands and made "Authentic" a 5% class that nothing could learn.
BAND_EDGES = [0.0, 0.15, 0.30, 1.01]

ACCOUNT_FEATURES = [
    "follower_following_ratio", "er_vs_benchmark", "comments_to_likes",
    "follower_growth_rate", "views_to_followers", "log_followers",
]
COMMENT_FEATURES = [
    "comment_automated_rate", "comment_duplicate_rate", "comment_emoji_only_rate",
    "comment_generic_rate", "comment_mean_words", "comment_link_cue_rate",
    "comment_sentiment_positive",
]


def truth(comments: pd.DataFrame) -> pd.DataFrame:
    """Ground truth from the archetype labels. Never given to the model."""
    bad = comments.archetype.isin(["bot", "engagement_pod", "spam"])
    share = comments.assign(_bad=bad).groupby("influencer_id")["_bad"].mean()
    band = pd.cut(share, bins=BAND_EDGES, labels=BANDS, right=False)
    return pd.DataFrame({"inauthentic_share": share.round(4),
                         "audience_band_true": band.astype(str)}).reset_index()


def build_features(influencers: pd.DataFrame,
                   comment_agg: pd.DataFrame) -> pd.DataFrame:
    f = influencers.copy()
    f["influencer_id"] = f.influencer_id.astype(str)
    f["log_followers"] = np.log1p(f.followers)
    keep = ["influencer_id"] + [c for c in ACCOUNT_FEATURES if c in f.columns]
    out = f[keep].merge(comment_agg, on="influencer_id", how="inner")
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def _cv_scores(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict]:
    pred = np.empty(len(y), dtype=object)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=3000, class_weight="balanced",
                                 random_state=SEED)
        clf.fit(sc.transform(X[tr]), y[tr])
        pred[te] = clf.predict(sc.transform(X[te]))
    pred = np.array(list(pred))
    return pred, {
        "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
        "accuracy": round(float((pred == y).mean()), 4),
    }


def train(features: pd.DataFrame, truth_df: pd.DataFrame) -> dict:
    d = features.merge(truth_df, on="influencer_id", how="inner")
    y = d.audience_band_true.to_numpy()

    acct = [c for c in ACCOUNT_FEATURES if c in d.columns]
    comm = [c for c in COMMENT_FEATURES if c in d.columns]

    results = {"n_creators": int(len(d)), "bands": BANDS,
               "band_distribution": d.audience_band_true.value_counts(
                   normalize=True).round(4).to_dict(),
               "arms": []}

    # The folk heuristic on its own: "a real account follows far fewer people
    # than follow it". Quoted because it is what a brand manager actually uses.
    #
    # Calibrated to THIS data's tertiles rather than to absolute cut-offs. The
    # first version used > 8 and > 2, which on a population whose median ratio
    # is 82 classified everyone as Authentic and scored 0.109 accuracy. That
    # would have been a strawman: it tested my thresholds, not the heuristic.
    # Split at the tertiles, the rule gets the most favourable cut points the
    # data allows, and whatever it then scores is the heuristic's own fault.
    r = d.follower_following_ratio
    lo, hi = r.quantile(1 / 3), r.quantile(2 / 3)
    ratio_pred = np.where(r >= hi, "Authentic",
                          np.where(r >= lo, "Mixed", "Suspect"))
    results["arms"].append({
        "arm": "follower/following rule (tertile-calibrated)",
        "features": 1,
        "correlation_with_truth": None,
        "macro_f1": round(float(f1_score(y, ratio_pred, average="macro")), 4),
        "accuracy": round(float((ratio_pred == y).mean()), 4)})

    results["arms"][-1]["correlation_with_truth"] = round(
        float(np.corrcoef(r, d.inauthentic_share)[0, 1]), 4)

    majority = np.full(len(y), pd.Series(y).value_counts().idxmax())
    results["arms"].append({
        "arm": "majority baseline", "features": 0,
        "macro_f1": round(float(f1_score(y, majority, average="macro")), 4),
        "accuracy": round(float((majority == y).mean()), 4)})

    for name, cols in (("account signals only", acct),
                       ("account + comment section", acct + comm)):
        pred, sc = _cv_scores(d[cols].to_numpy(float), y)
        results["arms"].append({"arm": name, "features": len(cols), **sc})
        if name.startswith("account + comment"):
            results["confusion_matrix"] = {
                "labels": BANDS,
                "matrix": confusion_matrix(y, pred, labels=BANDS).tolist()}
            results["predictions_available"] = True

    # Serving model: fitted on everything, used to score every creator. Honest
    # about what that means - these are in-sample predictions, and the numbers
    # QUOTED anywhere are the cross-validated ones above, never these.
    cols = acct + comm
    sc = StandardScaler().fit(d[cols].to_numpy(float))
    clf = LogisticRegression(max_iter=3000, class_weight="balanced",
                             random_state=SEED)
    clf.fit(sc.transform(d[cols].to_numpy(float)), y)
    proba = clf.predict_proba(sc.transform(d[cols].to_numpy(float)))
    suspect = proba[:, list(clf.classes_).index("Suspect")]
    mixed = proba[:, list(clf.classes_).index("Mixed")]

    scored = pd.DataFrame({
        "influencer_id": d.influencer_id,
        "audience_band": clf.predict(sc.transform(d[cols].to_numpy(float))),
        "p_suspect": suspect.round(4),
        # 100 = clean. The score a brand actually reads.
        "audience_quality_score": np.round(
            100 * (1 - np.clip(suspect + 0.4 * mixed, 0, 1)), 1),
        "inauthentic_share_true": d.inauthentic_share,
    })

    results["caveats"] = {
        "construction": "In this corpus the bot archetype writes cue-laden text, "
                        "so the automation rule catches it and the comment arm is "
                        "advantaged BY CONSTRUCTION. Treat the gap between arms as "
                        "an upper bound on what comment analysis buys, not an "
                        "estimate of it.",
        "still_informative": "The account-only arm is not advantaged that way - its "
                             "features are generated independently of the comment "
                             "mix - so its weakness is a genuine result about the "
                             "public-profile heuristics brands currently rely on.",
        "provenance": "Ground truth is the SIMULATED comment corpus's archetype "
                      "labels. No real account has been audited.",
    }
    RESULTS.write_text(json.dumps(results, indent=2))
    return {"results": results, "scored": scored}
