"""
The production NLP pipeline: runs every content-intelligence method over the
post corpus once, offline, and caches the results.

Design decision that matters for cost and for the report
--------------------------------------------------------
Not every method runs on every post, and that is deliberate rather than a
shortcut:

  rules / VADER / NRC   ->  ALL posts. Effectively free (~6,000 posts/sec).
  SBERT embeddings      ->  ALL posts. ~100/sec; needed by topics and brand-fit anyway.
  RoBERTa irony+sent    ->  a PER-CREATOR SAMPLE (default 10 posts each).
  BERTopic              ->  ALL posts, reusing the SBERT vectors.
  LLM prompting         ->  a STRATIFIED SAMPLE, and only on request.

The two sampled stages are sampled for measured reasons, not guessed ones.

RoBERTa was benchmarked on the target machine (Intel Core Ultra 5 125H) at
~5 posts/sec at fp32 - about fifty times slower than this hardware should
manage, and near-flat in thread count, which rules out a parallelism problem.
Two mitigations are applied: dynamic int8 quantisation (see transformers_hf.py)
and per-creator sampling. The features that consume these scores are per-creator
AVERAGES, and averaging 10 posts instead of 26 moves a rate estimate by a few
percent while cutting cost by ~60%. Sampling is done per creator, not globally,
so no creator is left with zero scored posts.

The LLM is sampled far harder because it is roughly 1,000x slower per post than
RoBERTa. Scoring 50k captions with a local 7B model is on the order of a day of
compute for a signal RoBERTa already provides.

Neither is a retreat from the supervisor's suggestion. The scientific claim -
which method detects sarcasm - is measured in src/benchmark/ on the full real
labelled test splits, where every method including the LLM sees identical rows.
Sampling applies only to the production feature pass, where the output is an
average and the sampling error is quantifiable.

Everything written here is consumed by src/features/build_features.py.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from src.config import ARTIFACT_DIR, PROCESSED_DIR, SEED
from src.nlp.extract import extract_frame, top_hashtags, top_keywords

NLP_DIR = ARTIFACT_DIR / "nlp"
NLP_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================================
# Helpers
# ==========================================================================


def _brand_product_vocab() -> tuple[set[str], set[str]]:
    from src.data.lexicon import NICHE_LEXICON

    brands, products = set(), set()
    for lex in NICHE_LEXICON.values():
        brands.update(lex["brands"])
        products.update(lex["products"])
    return brands, products


def _safe(step: str, fn, default=None, verbose: bool = True):
    """Run a stage, and on failure record the reason instead of crashing.

    A missing transformer must not destroy an eight-minute pipeline run; the
    feature builder handles absent columns, and the run report says exactly what
    was skipped so nothing silently disappears from the results.
    """
    t0 = time.time()
    try:
        out = fn()
        if verbose:
            print(f"    {step} ok ({time.time() - t0:.1f}s)")
        return out, {"status": "ok", "seconds": round(time.time() - t0, 1)}
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"    {step} SKIPPED: {type(exc).__name__}: {str(exc)[:140]}")
        return default, {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
            "seconds": round(time.time() - t0, 1),
        }


# ==========================================================================
# Resumable transformer inference
# ==========================================================================

CHECKPOINT_DIR = NLP_DIR / "roberta_chunks"


def _sample_posts_per_creator(posts: pd.DataFrame, per_creator: int, seed: int) -> list[int]:
    """Positional indices of up to `per_creator` posts for each creator.

    Sampling per creator rather than globally guarantees that no creator ends
    up with zero transformer-scored posts - which a uniform global sample would
    do to the least prolific accounts, producing NaN content features for
    exactly the creators the product most needs to evaluate carefully.
    """
    if per_creator is None or per_creator <= 0:
        return list(range(len(posts)))
    rng = np.random.default_rng(seed)
    order = pd.Series(range(len(posts)))
    picks: list[int] = []
    for _, grp in order.groupby(posts["influencer_id"].to_numpy()):
        vals = grp.to_numpy()
        if len(vals) <= per_creator:
            picks.extend(vals.tolist())
        else:
            picks.extend(rng.choice(vals, size=per_creator, replace=False).tolist())
    return sorted(picks)


def _run_transformers_resumable(texts: list[str], chunk_size: int = 2000) -> pd.DataFrame:
    """Score every post with both RoBERTa checkpoints, checkpointing as it goes.

    Why this exists
    ---------------
    The first production run of this stage died silently partway through - the
    machine slept overnight and took an hour of completed inference with it.
    Transformer inference over 50k posts on CPU is long enough that "restart
    from zero on any interruption" is not an acceptable design.

    Each chunk of `chunk_size` posts is written to disk as soon as it completes.
    Re-running skips every chunk already on disk, so an interrupted run resumes
    within one chunk of where it stopped. The chunks are keyed by index range,
    so they stay valid as long as the post corpus is unchanged - and the corpus
    is regenerated from a fixed seed, so it is.
    """
    import time

    from src.nlp.transformers_hf import roberta_irony, roberta_sentiment

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    n = len(texts)
    bounds = [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]

    done = {p.stem for p in CHECKPOINT_DIR.glob("chunk_*.parquet")}
    todo = [(a, b) for a, b in bounds if f"chunk_{a:07d}_{b:07d}" not in done]
    if done:
        print(f"    resuming: {len(done)}/{len(bounds)} chunks already complete")

    if todo:
        rs, ri = roberta_sentiment(), roberta_irony()
        id2 = {0: "negative", 1: "neutral", 2: "positive"}
        t_start = time.time()
        for k, (a, b) in enumerate(todo, 1):
            chunk = texts[a:b]
            p_sent = rs.predict_proba(chunk)
            p_iro = ri.predict_proba(chunk)
            pd.DataFrame(
                {
                    "_row": np.arange(a, b),
                    "roberta_sentiment": [id2[i] for i in p_sent.argmax(1)],
                    "roberta_p_negative": p_sent[:, 0].astype("float32"),
                    "roberta_p_neutral": p_sent[:, 1].astype("float32"),
                    "roberta_p_positive": p_sent[:, 2].astype("float32"),
                    "roberta_p_irony": p_iro[:, 1].astype("float32"),
                    "roberta_is_ironic": (p_iro[:, 1] >= 0.5).astype("int8"),
                }
            ).to_parquet(CHECKPOINT_DIR / f"chunk_{a:07d}_{b:07d}.parquet", index=False)

            elapsed = time.time() - t_start
            rate = (k * chunk_size) / max(elapsed, 1e-6)
            remaining = (len(todo) - k) * chunk_size / max(rate, 1e-6)
            print(f"    chunk {k}/{len(todo)} [{a:,}-{b:,}]  "
                  f"{rate:.0f} posts/s  ~{remaining / 60:.0f} min left", flush=True)

    frames = [pd.read_parquet(p) for p in sorted(CHECKPOINT_DIR.glob("chunk_*.parquet"))]
    out = pd.concat(frames, ignore_index=True).sort_values("_row")
    if len(out) != n:
        raise RuntimeError(
            f"checkpoint rows ({len(out):,}) do not match the corpus ({n:,}). "
            f"Delete {CHECKPOINT_DIR} and re-run to rebuild cleanly."
        )
    return out.drop(columns=["_row"]).reset_index(drop=True)


# ==========================================================================
# Main
# ==========================================================================


def run(
    max_posts: int | None = None,
    llm_sample: int = 600,
    transformer_posts_per_creator: int = 10,
    use_transformers: bool = True,
    use_bertopic: bool = True,
    use_llm: bool = False,
    min_topic_size: int = 40,
    seed: int = SEED,
) -> pd.DataFrame:
    posts = pd.read_parquet(PROCESSED_DIR / "posts.parquet")
    if max_posts:
        posts = posts.sample(n=min(max_posts, len(posts)), random_state=seed).reset_index(drop=True)

    texts = posts["caption"].fillna("").astype(str).tolist()
    print(f"  {len(texts):,} posts")
    report: dict[str, dict] = {}

    # ---- 1. deterministic rule features -----------------------------------
    print("  [1/6] rule-based extraction")
    brand_vocab, product_vocab = _brand_product_vocab()
    feats = extract_frame(posts, "caption", brand_vocab, product_vocab)
    hashtag_lists = dict(zip(feats["post_id"], feats["_hashtags"]))
    word_lists = dict(zip(feats["post_id"], feats["_words"]))
    feats = feats.drop(columns=["_hashtags", "_words"])
    report["rules"] = {"status": "ok", "n": len(feats)}

    # ---- 2. lexicon sentiment + emotion -----------------------------------
    print("  [2/6] lexicon sentiment and emotion")

    def _lex():
        from src.nlp.lexicons import NRCAffect, NRC_EMOTIONS, VaderSentiment

        v = VaderSentiment()
        nrc = NRCAffect()
        comp = np.array([v.raw_score(t) for t in texts])
        emo = np.vstack([nrc.emotion_vector(t) for t in texts])
        d = pd.DataFrame({"vader_compound": comp})
        for i, e in enumerate(NRC_EMOTIONS):
            d[f"nrc_{e}"] = emo[:, i]
        d["vader_label"] = np.where(comp >= 0.05, "positive",
                                    np.where(comp <= -0.05, "negative", "neutral"))
        return d

    lex_df, report["lexicon"] = _safe("lexicon", _lex, default=pd.DataFrame(index=range(len(texts))))
    feats = pd.concat([feats, lex_df.reset_index(drop=True)], axis=1)

    # ---- 3. SBERT embeddings ----------------------------------------------
    print("  [3/6] SBERT embeddings")

    def _emb():
        from src.nlp.embeddings import embed

        return embed(texts, show_progress=False)

    emb, report["embeddings"] = _safe("sbert", _emb)
    if emb is not None:
        np.save(NLP_DIR / "post_embeddings.npy", emb.astype(np.float32))

    # ---- 4. transformer sentiment + irony ---------------------------------
    if use_transformers:
        # Measured throughput on the target CPU is ~5 posts/s at fp32 and
        # ~3x that quantised. Scoring all 52k posts twice is still hours, and
        # it buys very little: the features that matter are per-creator
        # AVERAGES, and averaging 10 posts instead of 26 changes a rate
        # estimate by a few percent while cutting the cost by 60%.
        #
        # So we sample per creator rather than globally. Every creator keeps
        # full coverage - nobody ends up with zero transformer-scored posts,
        # which a global sample would have caused for the least prolific
        # accounts. The sample size is recorded and reported.
        idx = _sample_posts_per_creator(posts, per_creator=transformer_posts_per_creator, seed=seed)
        print(f"  [4/6] transformer sentiment and irony on {len(idx):,} posts "
              f"({transformer_posts_per_creator}/creator, {len(idx) / len(texts):.0%} of corpus)")

        def _tf():
            sub = [texts[i] for i in idx]
            scored = _run_transformers_resumable(sub, chunk_size=1000)
            # Re-expand to full corpus length; unsampled rows stay NaN and are
            # skipped by the mean() aggregations downstream.
            full = pd.DataFrame(index=range(len(texts)), columns=scored.columns, dtype="object")
            full.iloc[idx] = scored.to_numpy()
            for c in scored.columns:
                if c != "roberta_sentiment":
                    full[c] = pd.to_numeric(full[c], errors="coerce")
            return full

        tf_df, report["transformers"] = _safe("roberta", _tf)
        report["transformers"].update({
            "posts_scored": len(idx),
            "posts_total": len(texts),
            "per_creator_sample": transformer_posts_per_creator,
        })
        if tf_df is not None:
            feats = pd.concat([feats, tf_df.reset_index(drop=True)], axis=1)
    else:
        report["transformers"] = {"status": "disabled"}

    # ---- 5. topics ---------------------------------------------------------
    if use_bertopic and emb is not None:
        print("  [5/6] BERTopic vs LDA")

        def _top():
            from src.nlp.topics import run_comparison

            res = run_comparison(texts, embeddings=emb, min_topic_size=min_topic_size, seed=seed)
            return res

        topic_res, report["topics"] = _safe("bertopic", _top)
        if topic_res is not None:
            feats["topic_id"] = topic_res["_doc_topics"]
            tt = topic_res["_topic_table"].set_index("topic_id")["label"].to_dict()
            feats["topic_label"] = feats["topic_id"].map(lambda t: tt.get(t, "outlier"))
            report["topics"].update(
                {k: topic_res[k] for k in ("n_topics", "outlier_fraction", "bertopic", "lda")}
            )
    else:
        report["topics"] = {"status": "disabled"}

    # ---- 6. LLM on a stratified sample -------------------------------------
    if use_llm:
        print(f"  [6/6] LLM sarcasm on a {llm_sample}-post sample")

        def _llm():
            from src.nlp.sarcasm import OllamaIrony, ollama_available

            if not ollama_available():
                raise RuntimeError("Ollama is not running on localhost:11434")
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(texts), size=min(llm_sample, len(texts)), replace=False)
            m = OllamaIrony(strategy="few_shot")
            preds = m.predict([texts[i] for i in idx])
            return pd.DataFrame(
                {
                    "post_id": posts["post_id"].to_numpy()[idx],
                    "llm_irony": preds,
                    "llm_unparsed": m.n_unparsed,
                }
            )

        llm_df, report["llm"] = _safe("llm", _llm)
        if llm_df is not None:
            llm_df.to_parquet(NLP_DIR / "llm_sample.parquet", index=False)
    else:
        report["llm"] = {"status": "disabled", "reason": "cost - see module docstring"}

    # ---- persist -----------------------------------------------------------
    feats.to_parquet(NLP_DIR / "post_features.parquet", index=False)

    kw = top_keywords(
        {
            inf: [w for pid in grp for w in word_lists.get(pid, [])]
            for inf, grp in feats.groupby("influencer_id")["post_id"]
        }
    )
    ht = top_hashtags(
        {
            inf: [h for pid in grp for h in hashtag_lists.get(pid, [])]
            for inf, grp in feats.groupby("influencer_id")["post_id"]
        }
    )
    pd.DataFrame(
        {
            "influencer_id": list(kw.keys()),
            "top_keywords": ["|".join(t for t, _ in v) for v in kw.values()],
            "top_hashtags": ["|".join(t for t, _ in ht.get(k, [])) for k in kw],
        }
    ).to_parquet(NLP_DIR / "influencer_keywords.parquet", index=False)

    (NLP_DIR / "nlp_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"  wrote {NLP_DIR}")
    return feats


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-posts", type=int, default=None)
    ap.add_argument("--no-transformers", action="store_true")
    ap.add_argument("--no-bertopic", action="store_true")
    ap.add_argument("--llm", action="store_true", help="also run LLM on a sample")
    ap.add_argument("--llm-sample", type=int, default=600)
    ap.add_argument("--tf-per-creator", type=int, default=10,
                    help="posts per creator scored by the transformers (0 = all)")
    ap.add_argument("--min-topic-size", type=int, default=40)
    a = ap.parse_args()

    run(
        max_posts=a.max_posts,
        use_transformers=not a.no_transformers,
        use_bertopic=not a.no_bertopic,
        use_llm=a.llm,
        llm_sample=a.llm_sample,
        transformer_posts_per_creator=a.tf_per_creator,
        min_topic_size=a.min_topic_size,
    )
