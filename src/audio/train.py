"""
Build the corpus, train the three arms, run the sweeps, write the artefacts.

    python -m src.audio.train --stage corpus
    python -m src.audio.train --stage train
    python -m src.audio.train --stage sweeps
    python -m src.audio.train --stage all

Staged because each stage has to finish inside a short shell window, and
because the corpus is expensive to regenerate but cheap to reuse.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from src.audio import models as M
from src.audio import simulate as S
from src.config import ARTIFACT_DIR, ROOT, SEED

AUDIO_DIR = ARTIFACT_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

CORPUS = AUDIO_DIR / "corpus.parquet"
EMBED = AUDIO_DIR / "prosody_embeddings.npy"
PROBA = AUDIO_DIR / "branch_probabilities.npz"
RESULTS = AUDIO_DIR / "audio_model_results.json"


# ==========================================================================
def build_corpus(wer: float = S.DEFAULT_WER) -> pd.DataFrame:
    posts = pd.read_parquet(
        ROOT / "data" / "processed" / "posts.parquet",
        columns=["post_id", "influencer_id", "caption", "gen_sentiment",
                 "gen_is_sarcastic", "gen_brand", "gen_product"])
    posts["influencer_id"] = posts.influencer_id.astype(str)
    posts = posts[S.is_video(posts.post_id)].reset_index(drop=True)

    posts = S.latent_affect(posts)
    posts = S.annotate(posts)

    voices = pd.DataFrame({"influencer_id": posts.influencer_id.unique()})
    voices["speech_rate"] = [118.0 + 46.0 * S._unit(i, "rate")
                             for i in voices.influencer_id]
    posts = posts.merge(voices, on="influencer_id", how="left")

    posts["spoken_script"] = S.spoken_script(posts)
    asr = S.transcribe(posts.spoken_script, posts.speech_rate.to_numpy(), wer=wer)
    posts = pd.concat([posts, asr], axis=1)

    kappa = S.fleiss_kappa(posts[[f"label_{a['name']}" for a in S.ANNOTATORS]])
    posts.attrs["fleiss_kappa"] = kappa

    keep = posts[posts.adjudicated].reset_index(drop=True)
    emb = S.prosody_embeddings(keep, keep.influencer_id)

    keep.to_parquet(CORPUS, index=False)
    np.save(EMBED, emb)
    meta = {
        "n_video_posts": int(len(posts)),
        "n_adjudicated": int(len(keep)),
        "dropped_no_majority": int((~posts.adjudicated).sum()),
        "fleiss_kappa": round(float(kappa), 4),
        "asr_wer_target": wer,
        "asr_wer_realised": round(float(keep.asr_wer_true.mean()), 4),
        "gold_distribution": keep.gold_label.value_counts(normalize=True)
                                 .round(4).to_dict(),
        "sarcastic_share": round(float(keep.is_sarcastic.mean()), 4),
        "n_creators": int(keep.influencer_id.nunique()),
    }
    print(json.dumps(meta, indent=2))
    return meta


# ==========================================================================
def caption_leak_diagnostic(top_k: int = 8) -> dict:
    """Can the CAPTION alone give away sarcasm in this corpus?

    Written because every arm scored 0.99 accuracy on the sarcastic subset,
    including the text-only arm, which should not have been able to see the
    sarcasm at all. It can: the synthetic caption generator writes sarcastic
    posts from a separate template vocabulary, so the two classes are lexically
    disjoint and a bag of words separates them almost perfectly.

    This is a defect in the CAPTION generator (src/data/generate_synthetic.py),
    not in the audio feature, and it means the fusion model's advantage here
    comes from ordinary delivery variation rather than from catching sarcasm.
    Recorded rather than quietly worked around.
    """
    import re
    from collections import Counter

    p = pd.read_parquet(ROOT / "data" / "processed" / "posts.parquet",
                        columns=["caption", "gen_is_sarcastic"])
    sar = p.gen_is_sarcastic.fillna(False).astype(bool)
    cs, cn = Counter(), Counter()
    for cap, is_sar in zip(p.caption, sar):
        t = set(re.findall(r"[a-z]{3,}", str(cap).lower()))
        (cs if is_sar else cn).update(t)
    ns, nn = int(sar.sum()), int((~sar).sum())
    rows = []
    for w, c in cs.most_common(600):
        ps, pn = c / ns, cn[w] / nn
        if ps > 0.05:
            # Laplace-smoothed, because several of these tokens appear in
            # exactly zero sincere captions and an unsmoothed ratio then
            # reports a lift in the hundreds of billions, which is noise
            # dressed as a finding.
            lift = ((c + 1) / (ns + 2)) / ((cn[w] + 1) / (nn + 2))
            rows.append({"token": w, "p_given_sarcastic": round(ps, 4),
                         "p_given_sincere": round(pn, 6),
                         "lift_laplace": round(lift, 1),
                         "sincere_occurrences": int(cn[w])})
    rows.sort(key=lambda r: -r["lift_laplace"])
    return {
        "finding": "Sarcastic and sincere captions are drawn from disjoint "
                   "template vocabularies, so a bag-of-words model separates "
                   "them almost perfectly without any audio at all.",
        "consequence": "The fusion model's gain in this corpus comes from "
                       "ordinary delivery variation, NOT from catching sarcasm. "
                       "Any sarcasm result measured on these synthetic captions "
                       "is a property of the generator. The NLP benchmark on "
                       "TweetEval and the Misra headlines is unaffected - that "
                       "corpus is real.",
        "n_sarcastic": ns, "n_sincere": nn,
        "most_leaking_tokens": rows[:top_k],
    }


# ==========================================================================
def train() -> dict:
    df = pd.read_parquet(CORPUS)
    emb = np.load(EMBED)
    y = df.gold_label.to_numpy()
    groups = df.influencer_id.to_numpy()
    sar = df.is_sarcastic.to_numpy(bool)

    t0 = time.time()
    text_p = M.text_branch(df, y, groups)
    audio_p = M.audio_branch(emb, y, groups)
    fused_p = M.fusion(text_p, audio_p, df, y, groups)
    np.savez_compressed(PROBA, text=text_p, audio=audio_p, fusion=fused_p)

    arms = [
        M.majority_baseline(y),
        M.evaluate("text only (caption + ASR)", text_p, y, sar),
        M.evaluate("audio only (prosody head)", audio_p, y, sar),
        M.evaluate("late fusion", fused_p, y, sar),
    ]
    for a in arms:
        print(f"    {a['arm']:<30} acc {a['accuracy']:.4f}  macroF1 {a['macro_f1']:.4f}"
              + (f"  [acc sarcastic {a['accuracy_sarcastic']:.4f} /"
                 f" sincere {a['accuracy_sincere']:.4f}]"
                 if a["accuracy_sarcastic"] is not None else ""))
    print(f"    fitted in {time.time() - t0:.1f}s")
    return {"arms": arms}


# ==========================================================================
# The sweeps are the honest experiments in this feature: their answers were not
# chosen when the generator was written. They run on a creator-stratified
# subsample at 3 folds rather than 5 - each sweep refits three models at every
# point, and the curve shape, not the third decimal, is what is being read.
SWEEP_CREATORS = 380
SWEEP_ROWS = 5200
SWEEP_SPLITS = 3


def _sweep_frame() -> pd.DataFrame:
    base = pd.read_parquet(CORPUS)
    rng = np.random.default_rng(SEED)
    creators = base.influencer_id.unique()
    take = set(rng.choice(creators, size=min(SWEEP_CREATORS, len(creators)),
                          replace=False))
    return base[base.influencer_id.isin(take)].head(SWEEP_ROWS).reset_index(drop=True)


def sweep_wer() -> list:
    """How much word error rate can the text branch absorb?"""
    M.N_SPLITS = SWEEP_SPLITS
    df0 = _sweep_frame()
    y, g = df0.gold_label.to_numpy(), df0.influencer_id.to_numpy()
    sar = df0.is_sarcastic.to_numpy(bool)
    emb = S.prosody_embeddings(df0, df0.influencer_id)
    ap = M.audio_branch(emb, y, g)
    out = []
    for wer in (0.0, 0.10, 0.20, 0.35, 0.50):
        asr = S.transcribe(df0.spoken_script, df0.speech_rate.to_numpy(), wer=wer)
        d = pd.concat([df0.drop(columns=[c for c in asr.columns if c in df0.columns]),
                       asr], axis=1)
        tp = M.text_branch(d, y, g, max_features=3000)
        fp = M.fusion(tp, ap, d, y, g)
        row = {"wer": wer,
               "text_macro_f1": M.evaluate("t", tp, y, sar)["macro_f1"],
               "fusion_macro_f1": M.evaluate("f", fp, y, sar)["macro_f1"],
               "audio_macro_f1": M.evaluate("a", ap, y, sar)["macro_f1"]}
        out.append(row)
        print(f"    WER {wer:.2f}   text {row['text_macro_f1']:.4f}   "
              f"fusion {row['fusion_macro_f1']:.4f}")
    return out


def sweep_prosody_noise() -> list:
    """How much recording noise does the speech-emotion head survive?"""
    M.N_SPLITS = SWEEP_SPLITS
    df0 = _sweep_frame()
    y, g = df0.gold_label.to_numpy(), df0.influencer_id.to_numpy()
    sar = df0.is_sarcastic.to_numpy(bool)
    tp = M.text_branch(df0, y, g, max_features=3000)
    out = []
    for noise in (0.5, 1.0, 1.75, 3.0, 5.0):
        emb = S.prosody_embeddings(df0, df0.influencer_id, noise=noise)
        ap = M.audio_branch(emb, y, g)
        fp = M.fusion(tp, ap, df0, y, g)
        row = {"noise_multiplier": noise,
               "audio_macro_f1": M.evaluate("a", ap, y, sar)["macro_f1"],
               "fusion_macro_f1": M.evaluate("f", fp, y, sar)["macro_f1"],
               "text_macro_f1": M.evaluate("t", tp, y, sar)["macro_f1"]}
        out.append(row)
        print(f"    noise x{noise:<5} audio {row['audio_macro_f1']:.4f}   "
              f"fusion {row['fusion_macro_f1']:.4f}")
    return out


def sweep_learning_curve() -> list:
    """How many labelled clips would a real version of this need?"""
    M.N_SPLITS = SWEEP_SPLITS
    base = pd.read_parquet(CORPUS)
    rng = np.random.default_rng(SEED + 5)
    creators = base.influencer_id.unique()
    out = []
    for n_cre in (25, 60, 140, 320):
        sub = set(rng.choice(creators, size=min(n_cre, len(creators)), replace=False))
        d = base[base.influencer_id.isin(sub)].reset_index(drop=True)
        if d.influencer_id.nunique() < SWEEP_SPLITS or len(d) < 300:
            continue
        y, g = d.gold_label.to_numpy(), d.influencer_id.to_numpy()
        sar = d.is_sarcastic.to_numpy(bool)
        e = S.prosody_embeddings(d, d.influencer_id)
        tp = M.text_branch(d, y, g, max_features=3000)
        ap = M.audio_branch(e, y, g)
        fp = M.fusion(tp, ap, d, y, g)
        row = {"n_creators": int(d.influencer_id.nunique()),
               "n_labelled_clips": int(len(d)),
               "fusion_macro_f1": M.evaluate("f", fp, y, sar)["macro_f1"]}
        out.append(row)
        print(f"    {row['n_labelled_clips']:>6} clips from "
              f"{row['n_creators']:>4} creators   fusion {row['fusion_macro_f1']:.4f}")
    return out


# ==========================================================================
CAVEATS = {
    "corpus": "SIMULATED. No video was recorded or watched. The gold label is "
              "the majority vote of three SIMULATED annotators over the "
              "generator's latent spoken affect. This is not a human-labelled "
              "corpus and must never be described as one.",
    "asr": "SIMULATED at the output interface. Whisper was not run. Word "
           "errors, timings and per-word confidence are generated at a chosen "
           "word error rate.",
    "prosody_encoder": "SIMULATED. wav2vec2 and HuBERT were not run. A fixed "
                       "random projection produces embeddings with the property "
                       "the head depends on - affect linearly decodable, "
                       "speaker identity a larger nuisance direction - and no "
                       "other property of a real encoder.",
    "models": "REAL. The text branch, the prosody head and the fusion layer are "
              "scikit-learn models fitted on this data with GroupKFold by "
              "creator and scored out of fold. The learning is genuine; the "
              "inputs are not.",
    "why_fusion_wins": "Sarcastic posts are constructed with a caption whose "
                       "sentiment is inverted while prosody keeps the true "
                       "affect. Fusion therefore SHOULD beat text-only on that "
                       "subset - that is the generator's design, not a finding. "
                       "The findings are the sweeps: how far word error rate "
                       "can rise before the text branch stops contributing, how "
                       "much recording noise the prosody head survives, and how "
                       "many labelled clips the fusion needs. None of those "
                       "curves were chosen.",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["corpus", "train", "sweep-wer", "sweep-noise",
                             "sweep-curve", "all"])
    args = ap.parse_args()

    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    results["caveats"] = CAVEATS

    if args.stage in ("corpus", "all"):
        print("  building the simulated multimodal corpus ...")
        results["corpus"] = build_corpus()
    if args.stage in ("train", "all"):
        print("  training text / prosody / fusion ...")
        results.update(train())
        results["corpus_diagnostics"] = caption_leak_diagnostic()
        d = results["corpus_diagnostics"]
        print(f"    caption leak: top token '{d['most_leaking_tokens'][0]['token']}' "
              f"appears in {d['most_leaking_tokens'][0]['p_given_sarcastic']:.1%} of "
              f"sarcastic captions and {d['most_leaking_tokens'][0]['sincere_occurrences']} "
              f"sincere ones (smoothed lift "
              f"{d['most_leaking_tokens'][0]['lift_laplace']:,.0f}x)")
    sw = results.setdefault("sweeps", {})
    if args.stage in ("sweep-wer", "all"):
        print("  sweep: ASR word error rate ...")
        sw["wer"] = sweep_wer()
    if args.stage in ("sweep-noise", "all"):
        print("  sweep: prosody recording noise ...")
        sw["prosody_noise"] = sweep_prosody_noise()
    if args.stage in ("sweep-curve", "all"):
        print("  sweep: labelled-clip learning curve ...")
        sw["learning_curve"] = sweep_learning_curve()

    RESULTS.write_text(json.dumps(results, indent=2))
    print(f"  wrote {RESULTS}")


if __name__ == "__main__":
    main()
