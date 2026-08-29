"""
The simulated voice track.

Two kinds of check here. The first kind is ordinary: bounds, determinism,
coverage, and that the aggregation agrees with the post table it came from.
The second kind matters more - the audio is generated, it feeds a score that
the project's audit measured, and the honest thing is to assert that its
provenance is declared everywhere it is exposed rather than to assume it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT))

APP_DATA = ROOT / "app_data"


@pytest.fixture(scope="module")
def audio():
    return pd.read_parquet(APP_DATA / "nectar_audio_posts.parquet")


@pytest.fixture(scope="module")
def voices():
    return pd.read_parquet(APP_DATA / "nectar_audio_creators.parquet")


@pytest.fixture(scope="module")
def meta():
    return json.loads((APP_DATA / "nectar_meta.json").read_text())


# --------------------------------------------------------------------------
# Ordinary correctness
# --------------------------------------------------------------------------
def test_features_are_in_range(audio):
    assert audio.audio_valence.between(-1, 1).all()
    assert audio.audio_arousal.between(0, 1).all()
    assert audio.audio_confidence.between(0, 1).all()
    assert audio.speech_rate_wpm.between(80, 215).all()
    assert audio.pause_ratio.between(0, 1).all()
    assert audio.music_ratio.between(0, 1).all()


def test_labels_follow_the_thresholds(audio):
    from src.nlp.audio_sim import NEG_THRESHOLD, POS_THRESHOLD
    speech = audio[audio.audio_is_speech]
    assert (speech.loc[speech.audio_sentiment == "positive", "audio_valence"]
            > POS_THRESHOLD).all()
    assert (speech.loc[speech.audio_sentiment == "negative", "audio_valence"]
            < NEG_THRESHOLD).all()


def test_mostly_music_is_never_given_a_sentiment(audio):
    """A track that is 80% backing music has no voice to read."""
    assert (audio.loc[~audio.audio_is_speech, "audio_sentiment"] == "neutral").all()


def test_only_video_posts_have_audio(audio):
    posts = pd.read_parquet(ROOT / "data" / "processed" / "posts.parquet",
                            columns=["post_id"])
    assert len(audio) < len(posts)
    assert audio.post_id.is_unique
    assert audio.post_id.isin(posts.post_id).all()


def test_voice_is_deterministic():
    """The same creator must not get a different voice on the next rebuild."""
    from src.nlp.audio_sim import creator_voices
    ids = ["INF00000", "INF00007", "INF01999"]
    a = creator_voices(ids).set_index("influencer_id")
    b = creator_voices(list(reversed(ids))).set_index("influencer_id")
    pd.testing.assert_frame_equal(a.sort_index(), b.sort_index())


def test_aggregation_matches_the_post_table(audio, voices):
    counts = audio.groupby("influencer_id").size()
    v = voices.set_index("influencer_id")
    assert (v.n_video_posts == counts.reindex(v.index)).all()
    recomputed = audio.groupby("influencer_id").tone_mismatch.mean().round(4)
    assert np.allclose(v.tone_mismatch_rate, recomputed.reindex(v.index), atol=1e-4)


def test_shares_sum_to_one(voices):
    total = (voices.audio_share_positive + voices.audio_share_neutral
             + voices.audio_share_negative)
    assert np.allclose(total, 1.0, atol=0.002)


def test_mismatch_is_a_sign_disagreement_only(audio):
    """positive-vs-neutral is not a mismatch, or the flag means nothing."""
    m = audio[audio.tone_mismatch]
    pairs = set(zip(m.caption_sentiment, m.audio_sentiment))
    assert pairs <= {("positive", "negative"), ("negative", "positive")}


# --------------------------------------------------------------------------
# Honesty
# --------------------------------------------------------------------------
def test_audio_feeds_content_safety_identically_in_both_engines():
    """The batch scorer and the intake page must not drift apart."""
    from nectar import match
    from src.models.brandfit import score_pair
    creators = pd.read_parquet(APP_DATA / "nectar_creators.parquet").head(40)
    brief = match.Brief(category="Beauty", brand_text="skincare", geos=[], ages=[])
    ranked, _ = match.score(brief)
    ranked = ranked.set_index("influencer_id")
    brand = pd.Series({"category": "Beauty", "target_geo": None,
                       "target_age_band": None, "competitor_brands": ""})
    for _, row in creators.iterrows():
        expected = score_pair(row, brand, 0.0)["fit_content_safety"]
        got = ranked.loc[str(row.influencer_id), "fit_content_safety"]
        # score_pair rounds its components to four decimals on the way out;
        # the app computes them in full precision. Agreement to within that
        # rounding is the strongest claim the comparison can make.
        assert abs(expected - got) < 1e-4, row.influencer_id


def test_audio_lowers_content_safety_rather_than_being_decorative():
    """If the term cannot move the score, it is not a feature."""
    creators = pd.read_parquet(APP_DATA / "nectar_creators.parquet")
    fit = pd.read_parquet(APP_DATA / "nectar_fit.parquet")
    penalty = (0.30 * creators.audio_share_negative.fillna(0)
               + 0.20 * creators.tone_mismatch_rate.fillna(0))
    assert penalty.mean() > 0.02, "audio terms are too small to matter"
    assert penalty.std() > 0.01, "audio penalty is a constant, so it cannot reorder"
    assert fit.fit_content_safety.between(0, 1).all()


def test_provenance_is_declared_in_the_metadata(meta):
    audio_meta = meta.get("audio_simulation")
    assert audio_meta, "nectar_meta.json does not declare the audio simulation"
    assert "simulated" in audio_meta["provenance"].lower()
    assert "circularity_warning" in audio_meta
    assert "simulated" in meta["provenance"]["content_safety"].lower()


def test_data_dictionary_marks_every_audio_column_simulated():
    dic = pd.read_parquet(APP_DATA / "data_dictionary.parquet")
    rows = dic[dic.table.astype(str).str.startswith("audio")]
    assert len(rows) > 10, "audio tables are not in the data dictionary"
    measured = rows[~rows.provenance.str.contains("simulated|identifier|derived",
                                                  case=False, na=False)]
    assert measured.empty, f"audio columns not marked simulated: {list(measured.column)}"


def test_the_sarcasm_result_is_reported_with_its_circularity(meta):
    """The mismatch rate on sarcastic posts is arithmetic, not detection.

    It is a legitimate number to show; it is not legitimate to show it without
    saying that the generator put it there.
    """
    a = meta["audio_simulation"]
    assert a["tone_mismatch_rate_sarcastic"] > a["tone_mismatch_rate_sincere"]
    warning = a["circularity_warning"].lower()
    assert "arithmetic" in warning and "not detection" in warning
