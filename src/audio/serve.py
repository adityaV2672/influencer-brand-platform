"""
Serving: turn the trained fusion model's out-of-fold predictions into the two
tables the product reads.

Why out-of-fold rather than a refit-on-everything model
-------------------------------------------------------
Every label shipped to the dashboard is the prediction the model made for that
post while that post's CREATOR was held out of training. A model refit on all
the data would score its own training rows and every creator's voice profile
would be flattered by memorised speaker identity. The out-of-fold matrix is the
only honest thing to serve, and it costs nothing here because the pipeline is
offline anyway.

The hosted app still loads no model. This runs at build time and writes
parquet, exactly like every other scoring surface in the project.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.audio import simulate as S
from src.audio.models import LABELS
from src.config import ARTIFACT_DIR

AUDIO_DIR = ARTIFACT_DIR / "audio"
CORPUS = AUDIO_DIR / "corpus.parquet"
PROBA = AUDIO_DIR / "branch_probabilities.npz"
RESULTS = AUDIO_DIR / "audio_model_results.json"

MUSIC_CEILING = 0.72          # above this the clip is backing track, not voice


def available() -> bool:
    return CORPUS.exists() and PROBA.exists() and RESULTS.exists()


def build_posts(post_nlp: pd.DataFrame) -> pd.DataFrame:
    """One row per video post, labelled by the fusion model."""
    df = pd.read_parquet(CORPUS)
    probs = np.load(PROBA)["fusion"]
    assert len(probs) == len(df), "fusion probabilities do not match the corpus"

    df = pd.concat([df, S.recording_features(df)], axis=1)

    pred = np.array(LABELS)[probs.argmax(axis=1)]
    conf = probs.max(axis=1)
    p_pos = probs[:, LABELS.index("positive")]
    p_neg = probs[:, LABELS.index("negative")]

    mostly_music = df.music_ratio.to_numpy() > MUSIC_CEILING
    df["audio_sentiment"] = np.where(mostly_music, "neutral", pred)
    df["audio_is_speech"] = ~mostly_music
    df["audio_confidence"] = np.round(np.where(mostly_music, 0.25, conf), 3)

    # Valence is read off the model, not off the latent. Reporting the latent
    # would be showing the answer key and calling it a prediction.
    df["audio_valence"] = np.round(p_pos - p_neg, 3)
    df["audio_p_positive"] = np.round(p_pos, 4)
    df["audio_p_negative"] = np.round(p_neg, 4)

    df["speech_rate_wpm"] = df["asr_words_per_min"]

    cap = (post_nlp.set_index("post_id")
           .reindex(df.post_id)[["roberta_sentiment", "vader_label"]])
    df["caption_sentiment"] = (cap.roberta_sentiment.fillna(cap.vader_label)
                               .fillna("neutral").to_numpy())
    df["caption_source"] = np.where(cap.roberta_sentiment.notna().to_numpy(),
                                    "roberta", "vader")

    a, c = df.audio_sentiment.to_numpy(), df.caption_sentiment.to_numpy()
    df["tone_mismatch"] = (((c == "positive") & (a == "negative"))
                           | ((c == "negative") & (a == "positive")))
    df["model_correct"] = df.audio_sentiment.to_numpy() == df.gold_label.to_numpy()

    cols = ["post_id", "influencer_id", "audio_sentiment", "audio_valence",
            "audio_arousal", "audio_confidence", "audio_p_positive",
            "audio_p_negative", "speech_rate_wpm", "pause_ratio",
            "pitch_variation", "music_ratio", "audio_is_speech", "tone_mismatch",
            "caption_sentiment", "caption_source", "gold_label", "model_correct",
            "is_sarcastic", "asr_duration_s", "asr_mean_confidence",
            "asr_low_conf_share", "asr_filler_rate", "asr_wer_true",
            "spoken_disclosure", "transcript"]
    return df[cols].reset_index(drop=True)


def build_creators(audio_posts: pd.DataFrame) -> pd.DataFrame:
    g = audio_posts.groupby("influencer_id")
    out = pd.DataFrame({
        "n_video_posts": g.size(),
        "audio_valence_mean": g.audio_valence.mean().round(4),
        "audio_arousal_mean": g.audio_arousal.mean().round(4),
        "audio_speech_rate_mean": g.speech_rate_wpm.mean().round(1),
        "audio_pause_ratio_mean": g.pause_ratio.mean().round(4),
        "audio_music_ratio_mean": g.music_ratio.mean().round(4),
        "audio_confidence_mean": g.audio_confidence.mean().round(4),
        "asr_mean_confidence": g.asr_mean_confidence.mean().round(4),
        "spoken_disclosure_rate": g.spoken_disclosure.mean().round(4),
        "tone_mismatch_rate": g.tone_mismatch.mean().round(4),
    })
    shares = (audio_posts.groupby(["influencer_id", "audio_sentiment"]).size()
              .unstack(fill_value=0))
    shares = shares.div(shares.sum(axis=1), axis=0)
    for lab in LABELS:
        out[f"audio_share_{lab}"] = shares.get(lab, 0.0).round(4)
    return out.reset_index()


def model_card() -> dict:
    """What the dashboard is allowed to say about this model."""
    r = json.loads(RESULTS.read_text())
    arms = {a["arm"]: a for a in r["arms"]}
    fusion = arms["late fusion"]
    return {
        "architecture": "late fusion: TF-IDF(caption + Whisper-class transcript) "
                        "-> logistic regression; 128-d prosody embedding -> "
                        "logistic regression; both branches' out-of-fold "
                        "probabilities plus ASR quality -> logistic regression",
        "validation": f"GroupKFold by creator, {r['corpus']['n_creators']:,} creators, "
                      f"{r['corpus']['n_adjudicated']:,} adjudicated clips",
        "macro_f1": fusion["macro_f1"],
        "accuracy": fusion["accuracy"],
        "lift_over_text_only": round(
            fusion["macro_f1"] - arms["text only (caption + ASR)"]["macro_f1"], 4),
        "lift_over_audio_only": round(
            fusion["macro_f1"] - arms["audio only (prosody head)"]["macro_f1"], 4),
        "majority_baseline_macro_f1": arms["majority baseline"]["macro_f1"],
        "annotator_agreement_fleiss_kappa": r["corpus"]["fleiss_kappa"],
        "asr_word_error_rate": r["corpus"]["asr_wer_realised"],
        "arms": r["arms"],
        "sweeps": r.get("sweeps", {}),
        "corpus_diagnostics": r.get("corpus_diagnostics", {}),
        "caveats": r["caveats"],
    }
