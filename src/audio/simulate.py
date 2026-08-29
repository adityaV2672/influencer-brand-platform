"""
The sensor layer: what a microphone would have produced, had there been one.

WHAT IS REAL AND WHAT IS NOT - read this before quoting anything
----------------------------------------------------------------
Not real: the audio. No waveform exists in this project. Whisper was not run,
wav2vec2 was not run, and no person has ever watched or labelled a video from
this dataset. This module GENERATES what those components would have emitted.

Real: everything downstream of this file. src/audio/models.py fits genuine
scikit-learn models on these outputs, with grouped cross-validation, held-out
scoring, ablations and parameter sweeps. The learning is real; the inputs are
simulated. That is the same footing as the performance model in
src/models/train.py, and it is stated the same way.

Why simulate at THIS level rather than faking a transcript
-----------------------------------------------------------
A fabricated transcript would let the text branch cheat: if the words were
generated from the label, predicting the label from the words is a lookup. So
the chain is built the way the physical one runs, and each stage only sees
what the real stage would see:

    latent affect z
        |-> spoken script      (words the creator says - carries z weakly)
        |     |-> ASR output   (script + word errors at a set WER)
        |-> prosody embedding  (128-d, carries z strongly, plus speaker nuisance)
        |-> caption            (already generated; INVERTED when sarcastic)
        |-> gold label         (z as three simulated annotators would judge it)

The one structural fact worth stating plainly: sarcastic posts have a caption
whose sentiment is the opposite of z, while prosody still carries z. So a
fusion model SHOULD beat a text-only model on sarcastic posts. That is a
consequence of how this generator is written, not a discovery about the world.
What is NOT predetermined, and is therefore worth measuring, is HOW MUCH word
error rate the text branch tolerates, how much prosody noise the audio branch
survives, and how much data the fusion needs. Those sweeps are in models.py
and their answers were not chosen by me.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd

from src.config import SEED

# --------------------------------------------------------------------------
# Physical / recording constants
# --------------------------------------------------------------------------
VIDEO_SHARE = 0.58            # share of posts carrying a voice track
EMBED_DIM = 128               # wav2vec2-base pools to 768; 128 keeps the
                              # parquet small and the geometry identical
SPEAKER_SCALE = 3.0           # timbre is a far larger direction than affect,
                              # which is why every split is grouped by creator
DEFAULT_WER = 0.11            # Whisper-large-v3 sits near 0.08-0.12 on
                              # accented conversational English
PROSODY_NOISE = 1.0           # multiplier, swept in models.py
PROSODY_SNR = 2.35            # recording noise sigma at multiplier 1.0.
                              # Published wav2vec2 linear-probe emotion
                              # heads land near 0.60-0.70 macro F1; an
                              # encoder that let a linear model reach 0.90
                              # would be the giveaway that it is not one.

SARCASM_VOICE = -0.85

# How much of the spoken affect the caption accounts for. Set from the product
# premise rather than tuned to a target score: roughly half of what a creator
# conveys on camera is visible in the copy they wrote, and the rest is the
# performance. At 1.0 the microphone is redundant and the whole feature is
# theatre; at 0.0 the two modalities are unrelated and fusion is just averaging
# noise. 0.45 is the honest middle and it is stated here so it can be argued
# with.
CAPTION_LOADING = 0.45

# How much of the spoken affect is the creator's standing disposition rather
# than this particular clip. Non-zero because affect is correlated within a
# creator in every real corpus, and because that correlation is exactly what
# makes a random train/test split dishonest for this task.
CREATOR_LOADING = 0.30
_SENTIMENT_LATENT = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
LABELS = ("negative", "neutral", "positive")

# Three simulated annotators. Real annotation of affect in short video runs at
# Krippendorff alpha in the 0.5-0.7 band; perfect agreement would be the
# giveaway that these are not people. Each annotator has a bias and a slip rate.
ANNOTATORS = [
    {"name": "A1", "bias": +0.06, "slip": 0.09},
    {"name": "A2", "bias": -0.05, "slip": 0.13},
    {"name": "A3", "bias": +0.01, "slip": 0.11},
]

FILLERS = ["um", "uh", "like", "you know", "basically", "actually"]
DISCLOSURE_PHRASES = [
    "paid partnership", "this is an ad", "sponsored by", "gifted by",
    "in collaboration with", "ad,", "thanks to",
]


def _unit(key: str, salt: str) -> float:
    """Deterministic [0,1) from a string. hash() is per-process salted."""
    h = hashlib.sha256(f"{salt}:{key}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


# ==========================================================================
# 1. Latent affect and the annotation set
# ==========================================================================
def latent_affect(posts: pd.DataFrame) -> pd.DataFrame:
    """z, and the affect the voice actually carries.

    The caption's sentiment and the spoken affect part company exactly when the
    post is sarcastic. Everywhere else they agree up to noise, which is what
    makes the sarcastic subset the interesting one.
    """
    p = posts.copy()
    z = p.gen_sentiment.map(_SENTIMENT_LATENT).fillna(0.0).to_numpy(dtype=float)
    sar = p.gen_is_sarcastic.fillna(False).astype(bool).to_numpy()

    # A caption is written marketing copy; a delivery is a performance on a
    # particular afternoon. They are related, not the same thing, and the whole
    # premise of adding a microphone is that the second is not recoverable from
    # the first.
    #
    # The first version of this file set z_spoken = z_caption except when
    # sarcastic, which made the gold label almost a function of the caption.
    # Every arm then scored macro F1 above 0.90 and the text branch alone
    # solved sarcasm too - the task was a lookup, and a lookup proves nothing
    # about fusion. CAPTION_LOADING is what makes the two views genuinely
    # different sources of evidence.
    delivery = np.array([
        (_unit(str(i), "d1") + _unit(str(i), "d2") + _unit(str(i), "d3") - 1.5) * 1.7
        for i in p.post_id])

    # A creator's standing disposition. Some people are warm on camera every
    # time and some are dry every time, so affect is not independent across a
    # creator's clips.
    #
    # This was missing from the first version and its absence had a specific
    # consequence: with speaker identity uncorrelated with the label, a random
    # train/test split leaked nothing, and the GroupKFold this module insists
    # on was guarding against a hazard the data did not contain. In a real
    # corpus the hazard is the main one there is - a model can score well by
    # learning that this particular voice is usually cheerful - so the
    # simulation now contains it, and tests/test_audio.py measures the gap
    # between a random and a grouped split rather than asserting it.
    disposition = np.array([(_unit(str(i), "disp") - 0.5) * 1.9
                            for i in p.influencer_id])

    z_spoken = (CAPTION_LOADING * z
                + CREATOR_LOADING * disposition
                + (1.0 - CAPTION_LOADING - CREATOR_LOADING) * delivery)
    # Sarcasm is the extreme case of the same phenomenon: the words say one
    # thing and the voice says the opposite.
    z_spoken = np.where(sar, SARCASM_VOICE + 0.15 * z + 0.20 * disposition, z_spoken)

    p["z_caption"] = z
    p["z_disposition"] = np.round(disposition, 4)
    p["z_delivery"] = np.round(delivery, 4)
    p["z_spoken"] = np.round(np.clip(z_spoken, -1.5, 1.5), 4)
    p["is_sarcastic"] = sar
    return p


def _label_from(value: np.ndarray) -> np.ndarray:
    return np.where(value > 0.25, "positive",
                    np.where(value < -0.25, "negative", "neutral"))


def annotate(posts: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Three simulated annotators judge the spoken affect; majority is gold.

    THIS IS NOT A HUMAN-LABELLED CORPUS. It is a simulated one, and the only
    honest thing to call it is a simulated one. What it does buy is a gold
    label that is NOT the generator's latent read straight off: annotator
    disagreement puts an irreducible error floor under every model trained on
    it, so a model that scores 0.99 here would be a bug rather than a triumph.
    """
    rng = np.random.default_rng(seed + 8821)
    n = len(posts)
    z = posts.z_spoken.to_numpy()
    votes = {}
    for a in ANNOTATORS:
        jitter = rng.normal(0.0, 0.30, n)
        seen = z + a["bias"] + jitter
        lab = _label_from(seen)
        slipped = rng.random(n) < a["slip"]
        if slipped.any():
            lab = lab.copy()
            lab[slipped] = rng.choice(LABELS, slipped.sum())
        votes[a["name"]] = lab

    v = pd.DataFrame(votes, index=posts.index)
    counts = pd.DataFrame({lab: (v == lab).sum(axis=1) for lab in LABELS})
    gold = counts.idxmax(axis=1)
    # A three-way split has no majority; those posts are dropped rather than
    # broken by a coin toss, exactly as an adjudication protocol would.
    unanimous_or_majority = counts.max(axis=1) >= 2
    out = posts.copy()
    for k in votes:
        out[f"label_{k}"] = votes[k]
    out["gold_label"] = gold
    out["annotator_agreement"] = counts.max(axis=1) / len(ANNOTATORS)
    out["adjudicated"] = unanimous_or_majority
    return out


def fleiss_kappa(votes: pd.DataFrame) -> float:
    """Chance-corrected agreement across the simulated annotators."""
    n_items, n_raters = len(votes), votes.shape[1]
    counts = np.stack([(votes == lab).sum(axis=1).to_numpy() for lab in LABELS], axis=1)
    p_i = ((counts ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = p_i.mean()
    p_j = counts.sum(axis=0) / (n_items * n_raters)
    p_e = (p_j ** 2).sum()
    return float((p_bar - p_e) / (1 - p_e)) if p_e < 1 else 1.0


# ==========================================================================
# 2. Spoken script and Whisper-class ASR
# ==========================================================================
# The words a creator says are drawn from their topic, not from their label.
# Affect leaks into word choice only through the closing phrase, weakly, which
# is why the text branch is beatable and the fusion has something to add.
_OPENERS = ["so", "okay so", "right", "alright", "listen", "quick one"]
_BODIES = [
    "i have been using this for about {k} weeks now",
    "let me show you how this actually works",
    "i picked this up last month and here is the thing",
    "a lot of you asked me about this one",
    "we tested this across {k} different sessions",
    "this is the part nobody talks about",
]
_CLOSERS = {
    "positive": ["honestly worth it", "i would buy it again", "this one stays",
                 "genuinely happy with it"],
    "neutral": ["make of that what you will", "your mileage may vary",
                "that is where it stands", "take a look yourself"],
    "negative": ["i would not spend on this", "it did not hold up",
                 "save your money", "this one is going back"],
}


def spoken_script(posts: pd.DataFrame, seed: int = SEED) -> pd.Series:
    """What the creator says out loud. Distinct from the caption on purpose."""
    rng = np.random.default_rng(seed + 3313)
    kws = posts.get("gen_product", pd.Series([""] * len(posts), index=posts.index))
    brands = posts.get("gen_brand", pd.Series([""] * len(posts), index=posts.index))
    out = []
    for i, (prod, brand, z) in enumerate(zip(kws.fillna("this"),
                                             brands.fillna(""),
                                             posts.z_spoken)):
        tone = "positive" if z > 0.25 else ("negative" if z < -0.25 else "neutral")
        parts = [rng.choice(_OPENERS)]
        body = rng.choice(_BODIES).format(k=rng.integers(2, 9))
        parts.append(body)
        if brand:
            parts.append(f"this is the {str(brand).lower()} {str(prod).lower()}")
        else:
            parts.append(f"this {str(prod).lower()}")
        # Affect reaches the words only here, and only 70% of the time.
        if rng.random() < 0.70:
            parts.append(str(rng.choice(_CLOSERS[tone])))
        # Spoken disclosure, sometimes present when the caption discloses.
        if rng.random() < 0.28:
            parts.append(str(rng.choice(DISCLOSURE_PHRASES)))
        out.append(" ".join(parts))
    return pd.Series(out, index=posts.index)


_CONFUSIONS = {
    "the": "a", "this": "these", "it": "that", "and": "an", "for": "four",
    "worth": "worse", "buy": "by", "not": "no", "here": "hear",
}


def transcribe(scripts: pd.Series, speech_rate: np.ndarray,
               wer: float = DEFAULT_WER, seed: int = SEED) -> pd.DataFrame:
    """A Whisper-class ASR pass, simulated at its OUTPUT interface.

    A real decoder emits tokens with timings and per-token confidence, and it
    gets words wrong at a rate that rises with accent, noise and speed. That
    interface is reproduced here: substitutions from a confusion table,
    deletions, filler insertions, per-word confidence that is lower on the
    words it got wrong, and timings derived from the speaker's rate.

    `wer` is a knob so that models.py can sweep it. That sweep is the honest
    experiment in this whole feature: the degradation curve was not chosen.
    """
    rng = np.random.default_rng(seed + 5507)
    rows = []
    for idx, (script, rate) in enumerate(zip(scripts, speech_rate)):
        words = script.split()
        kept, confs, errors = [], [], 0
        for w in words:
            r = rng.random()
            if r < wer * 0.55:                      # substitution
                kept.append(_CONFUSIONS.get(w, w[::-1] if len(w) > 4 else w))
                confs.append(float(rng.uniform(0.28, 0.62)))
                errors += 1
            elif r < wer * 0.80:                    # deletion
                errors += 1
                continue
            else:
                kept.append(w)
                confs.append(float(rng.uniform(0.72, 0.99)))
        if rng.random() < wer * 1.6:                # insertion of a filler
            pos = rng.integers(0, max(len(kept), 1))
            kept.insert(int(pos), str(rng.choice(FILLERS)))
            confs.insert(int(pos), float(rng.uniform(0.30, 0.55)))
            errors += 1

        n_words = max(len(kept), 1)
        duration = n_words / max(rate, 60.0) * 60.0
        text = " ".join(kept)
        low = float(np.mean(np.array(confs) < 0.65)) if confs else 0.0
        rows.append({
            "transcript": text,
            "asr_n_words": n_words,
            "asr_duration_s": round(duration, 2),
            "asr_words_per_min": round(n_words / max(duration, 0.1) * 60.0, 1),
            "asr_mean_confidence": round(float(np.mean(confs)) if confs else 0.0, 4),
            "asr_low_conf_share": round(low, 4),
            "asr_filler_rate": round(
                sum(text.count(f) for f in FILLERS) / n_words, 4),
            "asr_errors": errors,
            "asr_wer_true": round(errors / max(len(words), 1), 4),
            "spoken_disclosure": bool(
                re.search("|".join(re.escape(d) for d in DISCLOSURE_PHRASES), text)),
        })
    return pd.DataFrame(rows, index=scripts.index)


# ==========================================================================
# 3. wav2vec2 / HuBERT-class prosody encoder
# ==========================================================================
def _projection(seed: int = SEED) -> np.ndarray:
    """Fixed random projection of [affect, arousal] into embedding space.

    A real self-supervised encoder puts affect roughly linearly on top of a
    much larger speaker subspace. That is the property the head depends on and
    the only one reproduced here.
    """
    return np.random.default_rng(seed + 77).normal(0, 1.0, (2, EMBED_DIM))


def _voiceprint(speaker_id: str) -> np.ndarray:
    """A speaker's timbre: a stable, high-dimensional direction of its own.

    The first version encoded speaker identity as four latent dimensions, which
    left 160 speakers crowded into a 4-d subspace where no linear model could
    tell them apart. The consequence was measurable and awkward: a random
    train/test split scored the SAME as a grouped one, so the GroupKFold this
    module insists on was guarding a hazard the data did not contain.

    Real speaker embeddings are high-dimensional and near-orthogonal, which is
    why speaker leakage is the standard failure of speech-emotion evaluation.
    A per-creator random unit vector reproduces that, and the guard now has
    something to guard against - tests/test_audio.py measures the gap.
    """
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(f"voice:{speaker_id}".encode()).digest()[:8],
                       "big") % (2 ** 32))
    v = rng.normal(0, 1.0, EMBED_DIM)
    return v / np.linalg.norm(v)


def prosody_embeddings(posts: pd.DataFrame, speaker_ids,
                       noise: float = PROSODY_NOISE,
                       seed: int = SEED) -> np.ndarray:
    """128-d utterance embeddings, one per video post."""
    rng = np.random.default_rng(seed + 9109)
    n = len(posts)
    z = posts.z_spoken.to_numpy()
    arousal = 0.46 + 0.24 * np.abs(z) + rng.normal(0, 0.11, n)

    affect = np.column_stack([1.35 * z, 0.9 * arousal]) @ _projection(seed)

    cache: dict[str, np.ndarray] = {}
    speaker = np.stack([cache.setdefault(str(s), _voiceprint(str(s)))
                        for s in speaker_ids])

    emb = affect + SPEAKER_SCALE * speaker + rng.normal(0.0, PROSODY_SNR * noise,
                                                        (n, EMBED_DIM))
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    return emb.astype("float32")


def is_video(post_ids) -> np.ndarray:
    """Deterministic per post, so the same post is a Reel on every rebuild."""
    return np.array([_unit(str(p), "video") < VIDEO_SHARE for p in post_ids])


def recording_features(posts: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Sensor-level readings a real pipeline would compute from the waveform
    directly, without any model: how animated the delivery is, how much of the
    track is silence, and how much of it is backing music rather than speech.

    Simulated like everything else at this layer. They are display and gating
    signals - a clip that is 80% music has no voice for the emotion head to
    read, and the serving layer holds its label at neutral for that reason.
    """
    rng = np.random.default_rng(seed + 6613)
    n = len(posts)
    z = posts.z_spoken.to_numpy()
    arousal = np.clip(0.46 + 0.24 * np.abs(z) + rng.normal(0, 0.11, n), 0.0, 1.0)
    return pd.DataFrame({
        "audio_arousal": np.round(arousal, 3),
        "pause_ratio": np.round(np.clip(
            0.23 - 0.17 * (arousal - 0.5) + rng.normal(0, 0.035, n), 0.02, 0.60), 3),
        "pitch_variation": np.round(np.clip(
            rng.beta(2.4, 3.1, n) + 0.22 * (arousal - 0.5), 0.02, 1.0), 3),
        "music_ratio": np.round(np.clip(rng.beta(1.7, 4.2, n), 0.0, 1.0), 3),
    }, index=posts.index)
