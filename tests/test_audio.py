"""
The multimodal audio feature: ASR, prosody head, late fusion.

Three groups of checks.

  correctness  the simulators produce what they claim to, and the serving
               tables agree with the model output they came from
  methodology  the thing that would silently ruin this model - speaker
               identity leaking through a random split - is actually caught
  honesty      the provenance of every simulated component is declared where
               a reader could see it, and no artefact claims human labelling

The methodology group is the point. A speech-emotion head evaluated on a
random split of a corpus where each speaker appears many times measures
speaker recognition, not emotion, and reports a number two to three times
too high. test_random_split_inflates_the_score demonstrates that on this data
rather than asserting it.
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
AUDIO_ART = ROOT / "artifacts" / "audio"

pytestmark = pytest.mark.skipif(
    not (AUDIO_ART / "corpus.parquet").exists(),
    reason="audio model not trained; run python -m src.audio.train --stage all")


@pytest.fixture(scope="module")
def corpus():
    return pd.read_parquet(AUDIO_ART / "corpus.parquet")


@pytest.fixture(scope="module")
def results():
    return json.loads((AUDIO_ART / "audio_model_results.json").read_text())


@pytest.fixture(scope="module")
def posts():
    return pd.read_parquet(APP_DATA / "nectar_audio_posts.parquet")


@pytest.fixture(scope="module")
def voices():
    return pd.read_parquet(APP_DATA / "nectar_audio_creators.parquet")


# ==========================================================================
# correctness
# ==========================================================================
def test_asr_hits_its_target_word_error_rate(results):
    c = results["corpus"]
    assert abs(c["asr_wer_realised"] - c["asr_wer_target"]) < 0.03


def test_transcripts_are_not_the_captions(corpus):
    """If the transcript were the caption, the second modality would be a copy."""
    same = (corpus.transcript.str.lower().str.strip()
            == corpus.caption.str.lower().str.strip())
    assert same.mean() < 0.01


def test_asr_confidence_falls_when_word_error_rate_rises():
    from src.audio import simulate as S
    scripts = pd.Series(["so i have been using this for about four weeks now"] * 300)
    rate = np.full(300, 140.0)
    clean = S.transcribe(scripts, rate, wer=0.02)
    noisy = S.transcribe(scripts, rate, wer=0.40)
    assert noisy.asr_mean_confidence.mean() < clean.asr_mean_confidence.mean()
    assert noisy.asr_wer_true.mean() > clean.asr_wer_true.mean()


def test_prosody_embeddings_are_unit_norm_and_the_right_shape():
    from src.audio import simulate as S
    df = pd.DataFrame({"post_id": [f"P{i}" for i in range(50)],
                       "z_spoken": np.linspace(-1, 1, 50)})
    emb = S.prosody_embeddings(df, [f"INF{i % 7}" for i in range(50)])
    assert emb.shape == (50, S.EMBED_DIM)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-4)


def test_gold_label_is_not_the_latent_read_straight_off(corpus):
    """Annotator noise must put a real error floor under the task."""
    from src.audio.simulate import _label_from
    perfect = _label_from(corpus.z_spoken.to_numpy())
    assert (perfect == corpus.gold_label.to_numpy()).mean() < 0.95


def test_annotator_agreement_is_human_plausible(results):
    k = results["corpus"]["fleiss_kappa"]
    assert 0.30 < k < 0.80, f"kappa {k} is not a plausible human agreement level"


def test_undecidable_clips_are_dropped_not_coin_flipped(results, corpus):
    assert results["corpus"]["dropped_no_majority"] > 0
    assert corpus.adjudicated.all()


def test_served_labels_match_the_served_probabilities(posts):
    speech = posts[posts.audio_is_speech]
    lean_pos = speech[speech.audio_p_positive > speech.audio_p_negative + 0.25]
    assert (lean_pos.audio_sentiment != "negative").all()


def test_music_only_clips_get_no_sentiment(posts):
    assert (posts.loc[~posts.audio_is_speech, "audio_sentiment"] == "neutral").all()


def test_mismatch_is_a_sign_disagreement_only(posts):
    m = posts[posts.tone_mismatch]
    pairs = set(zip(m.caption_sentiment, m.audio_sentiment))
    assert pairs <= {("positive", "negative"), ("negative", "positive")}


def test_creator_aggregates_match_the_post_table(posts, voices):
    v = voices.set_index("influencer_id")
    assert (v.n_video_posts == posts.groupby("influencer_id").size()
            .reindex(v.index)).all()
    assert np.allclose(
        v.tone_mismatch_rate,
        posts.groupby("influencer_id").tone_mismatch.mean().reindex(v.index).round(4),
        atol=1e-4)
    total = (v.audio_share_positive + v.audio_share_neutral + v.audio_share_negative)
    assert np.allclose(total, 1.0, atol=0.002)


# ==========================================================================
# methodology
# ==========================================================================
def test_fusion_beats_both_branches_and_the_baseline(results):
    arms = {a["arm"]: a["macro_f1"] for a in results["arms"]}
    assert arms["late fusion"] > arms["text only (caption + ASR)"]
    assert arms["late fusion"] > arms["audio only (prosody head)"]
    assert arms["late fusion"] > arms["majority baseline"] + 0.20


def test_nobody_scores_suspiciously_well(results):
    """A three-class affect task with kappa 0.41 labels cannot be near-perfect.

    An earlier version of the generator tied the spoken affect to the caption
    almost deterministically and every arm scored above 0.90. That is the
    signature of a lookup, not a model, and it is worth a standing test.
    """
    for a in results["arms"]:
        assert a["macro_f1"] < 0.88, f"{a['arm']} at {a['macro_f1']} is too good"


def test_random_split_inflates_the_score(corpus):
    """Speaker identity leaks unless the split is grouped by creator.

    This is the reason every split in src/audio/models.py is a GroupKFold. The
    test demonstrates the leak rather than trusting the comment.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, KFold
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler

    from src.audio import simulate as S

    rng = np.random.default_rng(0)
    ids = corpus.influencer_id.unique()
    keep = set(rng.choice(ids, size=min(160, len(ids)), replace=False))
    d = corpus[corpus.influencer_id.isin(keep)].head(2400).reset_index(drop=True)
    emb = S.prosody_embeddings(d, d.influencer_id)
    y, g = d.gold_label.to_numpy(), d.influencer_id.to_numpy()

    def cv(splitter, **kw):
        pred = np.empty(len(y), dtype=object)
        for tr, te in splitter.split(emb, y, **kw):
            sc = StandardScaler().fit(emb[tr])
            clf = LogisticRegression(max_iter=800).fit(sc.transform(emb[tr]), y[tr])
            pred[te] = clf.predict(sc.transform(emb[te]))
        return f1_score(y, list(pred), average="macro")

    grouped = cv(GroupKFold(n_splits=3), groups=g)
    random = cv(KFold(n_splits=3, shuffle=True, random_state=0))
    assert random > grouped, (
        f"random {random:.4f} vs grouped {grouped:.4f} - the leak this design "
        "guards against did not appear, so the guard needs re-examining")


def test_the_sweeps_produce_curves_not_flat_lines(results):
    sw = results["sweeps"]
    noise = sw["prosody_noise"]
    assert noise[0]["audio_macro_f1"] - noise[-1]["audio_macro_f1"] > 0.15
    # Fusion must degrade more gracefully than the branch it is protecting.
    drop_audio = noise[0]["audio_macro_f1"] - noise[-1]["audio_macro_f1"]
    drop_fusion = noise[0]["fusion_macro_f1"] - noise[-1]["fusion_macro_f1"]
    assert drop_fusion < drop_audio

    curve = sw["learning_curve"]
    assert curve[-1]["fusion_macro_f1"] > curve[0]["fusion_macro_f1"] + 0.05


# ==========================================================================
# honesty
# ==========================================================================
def test_every_simulated_component_declares_itself(results):
    c = results["caveats"]
    for key in ("corpus", "asr", "prosody_encoder"):
        assert "SIMULATED" in c[key], key
    assert "REAL" in c["models"]
    assert "not a human-labelled corpus" in c["corpus"].lower()


def test_no_artefact_claims_human_labelling():
    """The one claim that would be fabrication rather than simulation."""
    import re
    phrase = re.compile(r"human[- ]labell?ed (video|clip)", re.I)
    # The phrase is allowed - and wanted - inside a denial. What must never
    # appear is an affirmative claim, so each occurrence has to sit just after
    # a negation.
    denied = re.compile(r"(no|not|never|nor)\s+(\w+\s+){0,3}$", re.I)
    for path in [(AUDIO_ART / "audio_model_results.json"),
                 (APP_DATA / "nectar_meta.json")]:
        text = path.read_text()
        for m in phrase.finditer(text):
            before = text[max(0, m.start() - 60):m.start()]
            assert denied.search(before), (
                f"{path.name} claims human-labelled video: "
                f"...{text[max(0, m.start() - 60):m.end() + 20]}...")


def test_the_caption_leak_is_recorded_not_buried(results):
    d = results["corpus_diagnostics"]
    assert d["most_leaking_tokens"][0]["lift_laplace"] > 50
    assert "not from catching sarcasm" in d["consequence"].lower()


def test_meta_carries_the_model_card_and_its_caveats():
    meta = json.loads((APP_DATA / "nectar_meta.json").read_text())
    card = meta["audio_model"]
    assert card["macro_f1"] > card["majority_baseline_macro_f1"]
    assert "GroupKFold by creator" in card["validation"]
    assert "SIMULATED" in card["caveats"]["asr"]
    assert "no waveform" in meta["provenance"]["content_safety"].lower()


def test_data_dictionary_marks_every_audio_column_simulated():
    dic = pd.read_parquet(APP_DATA / "data_dictionary.parquet")
    rows = dic[dic.table.astype(str).str.startswith("audio")]
    assert len(rows) > 10
    bad = rows[~rows.provenance.str.contains("simulated|identifier|derived",
                                             case=False, na=False)]
    assert bad.empty, f"unmarked audio columns: {list(bad.column)}"
