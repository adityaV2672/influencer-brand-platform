"""
Simulated audio sentiment for video posts.

READ THIS BEFORE QUOTING ANY NUMBER FROM THIS MODULE
----------------------------------------------------
There is no audio in this project. No waveform was ever loaded, no speech
recogniser was run, no audio model was trained or evaluated. Everything here is
generated, exactly like the creator universe in src/data/generate_synthetic.py,
and every column it produces is labelled `simulated` in the data dictionary.

What it is for: the platform's brand-safety story is one-eyed without it. A
brand reads a caption, but an audience watches a Reel, and a creator whose
captions are cheerful can deliver a flat or contemptuous voice-over. That gap
is the single most useful thing audio would add to a real version of this
product, so the feature is built end to end - generator, aggregation,
scoring, and UI - as a working demonstration of how a second modality plugs
into the existing composite.

How the simulation is constructed, and why that matters
-------------------------------------------------------
Audio is generated as a SECOND NOISY VIEW of the same latent post sentiment the
caption generator used - not from the caption model's prediction. That
distinction is the whole point:

  * If audio were derived from `roberta_sentiment`, then "audio disagrees with
    caption" would be a restatement of the caption model's own uncertainty and
    would carry no information at all.
  * Derived from the latent instead, a disagreement means the two views of the
    same underlying affect diverged - which is what genuinely happens with
    sarcasm, and what a real multimodal system would be built to catch.

Sarcastic posts are given a negative voice while their text reads positive,
because that is what sarcasm sounds like. This makes the tone-mismatch rate
higher on sarcastic posts by construction. That is a property of this
generator, NOT a discovery, and `agreement_metrics()` reports it with that
warning attached. Do not present it as evidence that audio detects sarcasm.

What would make this real: a Whisper-class ASR pass for the words, a
wav2vec2 or HuBERT speech-emotion head for the prosody, and a fusion layer
trained on human-labelled video. None of that fits in a dashboard that
deliberately loads no model, and none of it is claimed here.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from src.config import SEED

# Share of posts that carry a voice track. Instagram's own mix is roughly half
# video by 2026; 0.58 keeps enough per-creator sample that a creator-level mean
# is not being computed over four posts.
VIDEO_SHARE = 0.58

# How strongly the latent post sentiment drives the voice. Deliberately below
# 1.0 with a wide noise term: a second modality that agreed with the first
# almost always would be a copy, not a second view, and the disagreement is the
# only part of this feature a brand would actually pay for.
VOICE_LOADING = 0.62
VOICE_NOISE = 0.32

# A sarcastic post's voice carries the real affect while the words do not.
SARCASM_VOICE = -0.80

# Label thresholds on valence in [-1, 1].
POS_THRESHOLD = 0.20
NEG_THRESHOLD = -0.20

_SENTIMENT_LATENT = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}


def _stable_unit(key: str, salt: str) -> float:
    """A deterministic number in [0, 1) from a string.

    Python's builtin hash() is salted per process, so a creator's voice would
    change between runs and the same demo would show different numbers on
    Tuesday. SHA-256 does not move.
    """
    h = hashlib.sha256(f"{salt}:{key}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


def creator_voices(influencer_ids) -> pd.DataFrame:
    """One stable set of vocal traits per creator.

    A person's speaking voice is a trait, not a per-post draw: someone who
    talks fast and bright does it in every video. Making these per-creator is
    what lets a creator-level mean mean anything.
    """
    ids = list(dict.fromkeys(str(i) for i in influencer_ids))
    rows = []
    for iid in ids:
        rows.append({
            "influencer_id": iid,
            # Warmth of the default delivery, before any post's content.
            "voice_valence_bias": (_stable_unit(iid, "valence") - 0.5) * 0.44,
            "voice_arousal_bias": (_stable_unit(iid, "arousal") - 0.5) * 0.30,
            # Words per minute. Indian English creator speech clusters near
            # 140 wpm; the spread here is deliberately wide.
            "voice_speech_rate": 118.0 + 46.0 * _stable_unit(iid, "rate"),
            "voice_pitch_var": 0.18 + 0.34 * _stable_unit(iid, "pitch"),
        })
    return pd.DataFrame(rows)


def simulate(posts: pd.DataFrame, post_nlp: pd.DataFrame,
             seed: int = SEED) -> pd.DataFrame:
    """One row per video post: what the voice track would have sounded like.

    `posts` supplies the latent (gen_sentiment, gen_is_sarcastic);
    `post_nlp` supplies the caption model's own prediction, which is compared
    against the audio label but never used to produce it.
    """
    rng = np.random.default_rng(seed + 4471)

    p = posts[["post_id", "influencer_id", "gen_sentiment", "gen_is_sarcastic"]].copy()
    p["influencer_id"] = p["influencer_id"].astype(str)
    p = p.merge(
        post_nlp[["post_id", "roberta_sentiment", "roberta_p_irony", "vader_label"]],
        on="post_id", how="left")

    # RoBERTa was run over a 20,000-post subsample, not all 52,089 - the
    # quantised transformer manages about five posts a second on this hardware.
    # The caption view therefore falls back to VADER where the transformer has
    # no label, exactly as the profile page does, and records which one spoke.
    p["caption_sentiment"] = p.roberta_sentiment.fillna(p.vader_label)
    p["caption_source"] = np.where(p.roberta_sentiment.notna(), "roberta", "vader")

    # Which posts are video. Deterministic per post, so the same post is a Reel
    # on every rebuild.
    is_video = np.array([_stable_unit(str(pid), "video") < VIDEO_SHARE
                         for pid in p.post_id])
    p = p[is_video].reset_index(drop=True)

    p = p.merge(creator_voices(p.influencer_id), on="influencer_id", how="left")

    latent = p.gen_sentiment.map(_SENTIMENT_LATENT).fillna(0.0).to_numpy(dtype=float)
    sarcastic = p.gen_is_sarcastic.fillna(False).astype(bool).to_numpy()
    voice_latent = np.where(sarcastic, SARCASM_VOICE, latent)

    n = len(p)
    valence = (VOICE_LOADING * voice_latent
               + p.voice_valence_bias.to_numpy()
               + rng.normal(0.0, VOICE_NOISE, n))
    # Rounded before the label is derived, not after. Rounding afterwards left
    # a post whose valence was 0.2004 labelled positive while the stored value
    # read 0.200, so the column and the label disagreed at the threshold.
    p["audio_valence"] = np.round(np.clip(valence, -1.0, 1.0), 3)

    arousal = (0.46 + 0.24 * np.abs(voice_latent)
               + p.voice_arousal_bias.to_numpy()
               + rng.normal(0.0, 0.11, n))
    p["audio_arousal"] = np.round(np.clip(arousal, 0.0, 1.0), 3)

    p["speech_rate_wpm"] = np.clip(
        p.voice_speech_rate.to_numpy() + 26.0 * (p.audio_arousal - 0.5)
        + rng.normal(0.0, 6.5, n), 80.0, 215.0).round(1)

    # Faster, more animated delivery leaves less silence.
    p["pause_ratio"] = np.clip(
        0.23 - 0.17 * (p.audio_arousal - 0.5) + rng.normal(0.0, 0.035, n),
        0.02, 0.60).round(3)

    p["pitch_variation"] = np.clip(
        p.voice_pitch_var.to_numpy() + 0.22 * (p.audio_arousal - 0.5)
        + rng.normal(0.0, 0.05, n), 0.02, 1.0).round(3)

    # Share of the track that is background music rather than speech. A post
    # that is nearly all music has little voice to read, so its label is held
    # at neutral below.
    p["music_ratio"] = np.clip(rng.beta(1.7, 4.2, n), 0.0, 1.0).round(3)

    label = np.where(p.audio_valence > POS_THRESHOLD, "positive",
                     np.where(p.audio_valence < NEG_THRESHOLD, "negative", "neutral"))
    mostly_music = p.music_ratio.to_numpy() > 0.72
    p["audio_sentiment"] = np.where(mostly_music, "neutral", label)
    p["audio_is_speech"] = ~mostly_music

    # Confidence falls off near the decision thresholds and with music.
    p["audio_confidence"] = np.clip(
        0.52 + 0.46 * np.abs(p.audio_valence) - 0.30 * p.music_ratio,
        0.20, 0.99).round(3)

    # Mismatch is reserved for an actual sign disagreement. Counting
    # positive-vs-neutral as a mismatch would flag a third of all posts and the
    # flag would stop meaning anything.
    cap = p.caption_sentiment.fillna("neutral").to_numpy()
    aud = p.audio_sentiment.to_numpy()
    p["tone_mismatch"] = (((cap == "positive") & (aud == "negative"))
                          | ((cap == "negative") & (aud == "positive")))

    cols = ["post_id", "influencer_id", "audio_sentiment", "audio_valence",
            "audio_arousal", "audio_confidence", "speech_rate_wpm", "pause_ratio",
            "pitch_variation", "music_ratio", "audio_is_speech", "tone_mismatch",
            "caption_sentiment", "caption_source", "gen_is_sarcastic"]
    return p[cols]


def aggregate(audio_posts: pd.DataFrame) -> pd.DataFrame:
    """Creator-level voice profile."""
    g = audio_posts.groupby("influencer_id")
    out = pd.DataFrame({
        "n_video_posts": g.size(),
        "audio_valence_mean": g.audio_valence.mean().round(4),
        "audio_arousal_mean": g.audio_arousal.mean().round(4),
        "audio_speech_rate_mean": g.speech_rate_wpm.mean().round(1),
        "audio_pause_ratio_mean": g.pause_ratio.mean().round(4),
        "audio_music_ratio_mean": g.music_ratio.mean().round(4),
        "audio_confidence_mean": g.audio_confidence.mean().round(4),
        "tone_mismatch_rate": g.tone_mismatch.mean().round(4),
    })
    shares = (audio_posts.groupby(["influencer_id", "audio_sentiment"]).size()
              .unstack(fill_value=0))
    shares = shares.div(shares.sum(axis=1), axis=0)
    for lab in ("positive", "neutral", "negative"):
        out[f"audio_share_{lab}"] = shares.get(lab, 0.0).round(4)
    return out.reset_index()


def agreement_metrics(audio_posts: pd.DataFrame) -> dict:
    """How often the two views agree, and where they part company."""
    a = audio_posts
    cap = a.caption_sentiment.fillna("neutral")
    agree = float((cap == a.audio_sentiment).mean())
    sar = a.gen_is_sarcastic.fillna(False).astype(bool)
    return {
        "n_video_posts": int(len(a)),
        "video_share": round(VIDEO_SHARE, 3),
        "caption_audio_agreement": round(agree, 4),
        "caption_label_source": {k: int(v) for k, v in
                                 a.caption_source.value_counts().items()},
        "tone_mismatch_rate_overall": round(float(a.tone_mismatch.mean()), 4),
        "tone_mismatch_rate_sarcastic": round(float(a.loc[sar, "tone_mismatch"].mean()), 4)
        if sar.any() else None,
        "tone_mismatch_rate_sincere": round(float(a.loc[~sar, "tone_mismatch"].mean()), 4)
        if (~sar).any() else None,
        "circularity_warning": (
            "Sarcastic posts are GIVEN a negative voice by the simulator, so a "
            "higher mismatch rate on them is arithmetic, not detection. These "
            "figures describe the generator, not any real audio capability."),
        "provenance": "simulated - src/nlp/audio_sim.py. No audio was recorded, "
                      "transcribed or classified at any point in this project.",
    }
