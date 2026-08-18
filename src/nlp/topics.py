"""
Topic modelling: BERTopic (the method the supervisor recommended) with an LDA
baseline for comparison.

Why both
--------
BERTopic is not automatically better than LDA - it is better *when documents are
short*, which is precisely the influencer-caption case. LDA needs enough
word co-occurrence within a document to estimate a topic distribution, and a
20-word caption does not provide it; this is a well-documented weakness of
bag-of-words topic models on short text. BERTopic sidesteps it by clustering
sentence embeddings and only then extracting representative words via c-TF-IDF.

Reporting only BERTopic would assert its superiority. Running both and scoring
them with the same coherence metric demonstrates it, and quantifies the gap.

Coherence is measured with NPMI (c_npmi) and c_v over the same tokenised corpus
and the same top-n words per topic, so the two models are directly comparable.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from src.config import ARTIFACT_DIR, SEED

TOPIC_DIR = ARTIFACT_DIR / "topics"
TOPIC_DIR.mkdir(parents=True, exist_ok=True)

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z'\-]{2,}")


def tokenise_corpus(texts: list[str], stopwords: set[str] | None = None) -> list[list[str]]:
    from src.nlp.extract import STOPWORDS

    sw = stopwords if stopwords is not None else STOPWORDS
    return [[w for w in _TOKEN_RE.findall(t.lower()) if w not in sw] for t in texts]


# ==========================================================================
# BERTopic
# ==========================================================================


def fit_bertopic(
    texts: list[str],
    embeddings: np.ndarray | None = None,
    min_topic_size: int = 40,
    nr_topics: int | str | None = "auto",
    seed: int = SEED,
):
    """Fit BERTopic over pre-computed SBERT embeddings.

    Passing embeddings in (rather than letting BERTopic encode internally) means
    the same vectors are reused by brand-fit scoring and the classifier
    baselines - encoding 50k captions once instead of three times.

    UMAP is seeded for reproducibility. Note that UMAP is only deterministic
    when n_jobs=1; the speed loss is accepted so the reported topics are stable
    across runs.
    """
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP
    from hdbscan import HDBSCAN

    from src.nlp.extract import STOPWORDS

    umap_model = UMAP(
        n_neighbors=15, n_components=5, min_dist=0.0,
        metric="cosine", random_state=seed, n_jobs=1,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size, metric="euclidean",
        cluster_selection_method="eom", prediction_data=True,
    )
    vectorizer = CountVectorizer(
        stop_words=list(STOPWORDS), min_df=5, ngram_range=(1, 2), token_pattern=r"[a-zA-Z][a-zA-Z'\-]{2,}"
    )

    model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
        nr_topics=nr_topics,
        calculate_probabilities=False,
        verbose=True,
    )
    topics, _ = model.fit_transform(texts, embeddings=embeddings)
    return model, list(topics)


# ==========================================================================
# LDA baseline
# ==========================================================================


def fit_lda(tokens: list[list[str]], n_topics: int, seed: int = SEED):
    from gensim import corpora
    from gensim.models import LdaMulticore

    dictionary = corpora.Dictionary(tokens)
    dictionary.filter_extremes(no_below=5, no_above=0.4)
    corpus = [dictionary.doc2bow(t) for t in tokens]

    lda = LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=n_topics,
        random_state=seed,
        passes=8,
        chunksize=2000,
        workers=2,
        eval_every=None,
    )
    return lda, dictionary, corpus


# ==========================================================================
# Coherence
# ==========================================================================


def coherence(
    topic_words: list[list[str]],
    tokens: list[list[str]],
    measure: str = "c_npmi",
) -> float:
    """Coherence over the same tokenised corpus for any model's topic words."""
    from gensim import corpora
    from gensim.models import CoherenceModel

    dictionary = corpora.Dictionary(tokens)
    # Keep only words the dictionary knows, and drop degenerate topics.
    cleaned = [[w for w in t if w in dictionary.token2id] for t in topic_words]
    cleaned = [t for t in cleaned if len(t) >= 2]
    if not cleaned:
        return float("nan")
    cm = CoherenceModel(
        topics=cleaned, texts=tokens, dictionary=dictionary,
        coherence=measure, processes=1,
    )
    return float(cm.get_coherence())


def topic_diversity(topic_words: list[list[str]], top_n: int = 10) -> float:
    """Fraction of distinct words across all topics. 1.0 = no overlap.

    Coherence alone is gameable: a model that produces the same generic topic
    ten times scores well. Diversity is the counterweight, and the pair is the
    standard way to report topic quality.
    """
    words = [w for t in topic_words for w in t[:top_n]]
    return len(set(words)) / len(words) if words else float("nan")


# ==========================================================================
# Orchestration
# ==========================================================================


def run_comparison(
    texts: list[str],
    embeddings: np.ndarray | None = None,
    min_topic_size: int = 40,
    top_n_words: int = 10,
    seed: int = SEED,
) -> dict:
    """Fit BERTopic and a matched LDA, score both, save everything."""
    tokens = tokenise_corpus(texts)

    print("  fitting BERTopic ...")
    bt_model, bt_topics = fit_bertopic(
        texts, embeddings=embeddings, min_topic_size=min_topic_size, seed=seed
    )
    info = bt_model.get_topic_info()
    real_topics = [t for t in info["Topic"].tolist() if t != -1]
    n_topics = len(real_topics)
    bt_words = [
        [w for w, _ in bt_model.get_topic(t)[:top_n_words]] for t in real_topics
    ]
    outlier_frac = float(np.mean([t == -1 for t in bt_topics]))
    print(f"    {n_topics} topics, {outlier_frac:.1%} outliers")

    # Match LDA's topic count to BERTopic's so the comparison is like-for-like.
    print(f"  fitting LDA baseline with n_topics={n_topics} ...")
    lda, dictionary, corpus = fit_lda(tokens, n_topics=max(n_topics, 2), seed=seed)
    lda_words = [
        [w for w, _ in lda.show_topic(i, topn=top_n_words)] for i in range(lda.num_topics)
    ]

    print("  scoring coherence ...")
    results = {
        "n_topics": n_topics,
        "outlier_fraction": round(outlier_frac, 4),
        "top_n_words": top_n_words,
        "min_topic_size": min_topic_size,
        "n_documents": len(texts),
        "bertopic": {
            "npmi": round(coherence(bt_words, tokens, "c_npmi"), 4),
            "c_v": round(coherence(bt_words, tokens, "c_v"), 4),
            "diversity": round(topic_diversity(bt_words, top_n_words), 4),
        },
        "lda": {
            "npmi": round(coherence(lda_words, tokens, "c_npmi"), 4),
            "c_v": round(coherence(lda_words, tokens, "c_v"), 4),
            "diversity": round(topic_diversity(lda_words, top_n_words), 4),
        },
        "citations": {
            "bertopic": "Grootendorst, M. (2022). BERTopic. arXiv:2203.05794",
            "lda": "Blei, D., Ng, A., Jordan, M. (2003). Latent Dirichlet Allocation. JMLR 3.",
            "npmi": "Bouma, G. (2009). Normalized (Pointwise) Mutual Information in Collocation Extraction.",
            "c_v": "Röder, M., Both, A., Hinneburg, A. (2015). Exploring the Space of Topic Coherence Measures. WSDM.",
        },
    }

    topic_table = pd.DataFrame(
        {
            "topic_id": real_topics,
            "size": [int(info.loc[info["Topic"] == t, "Count"].iloc[0]) for t in real_topics],
            "top_words": ["|".join(w) for w in bt_words],
            "label": [
                ", ".join(w[:3]) for w in bt_words
            ],
        }
    )
    topic_table.to_parquet(TOPIC_DIR / "bertopic_topics.parquet", index=False)
    pd.DataFrame({"lda_topic_id": range(len(lda_words)),
                  "top_words": ["|".join(w) for w in lda_words]}).to_parquet(
        TOPIC_DIR / "lda_topics.parquet", index=False
    )
    (TOPIC_DIR / "coherence.json").write_text(json.dumps(results, indent=2))

    results["_bertopic_model"] = bt_model
    results["_doc_topics"] = bt_topics
    results["_topic_table"] = topic_table
    return results
