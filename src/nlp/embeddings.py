"""
SBERT sentence embeddings, plus the supervised classifiers built on top of them.

Role in the project
-------------------
Embeddings do three jobs here:

1. They give every method-comparison a *learned* baseline. An SBERT encoder with
   a logistic-regression head is the cheapest way to ask "how much of this task
   is solvable by generic semantics + a small amount of supervision?" It sits
   between the word lists and the fine-tuned transformers.
2. They power Brand-Fit scoring: a brand's category and keywords are encoded
   into the same space as an influencer's aggregated content, and fit becomes a
   cosine similarity. This replaces string matching on niche labels.
3. They are the document representation BERTopic clusters over.

Embeddings are computed ONCE and cached to disk. The deployed dashboard never
loads a transformer - it reads the cached vectors. That keeps the hosted app
inside a 1 GB memory budget.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from src.config import ARTIFACT_DIR, SBERT_MODEL, SEED
from src.nlp.base import MethodMeta, TextMethod

_MODEL_CACHE: dict[str, object] = {}


def get_encoder(model_name: str = SBERT_MODEL):
    """Load (and memoise) a SentenceTransformer encoder."""
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        _MODEL_CACHE[model_name] = SentenceTransformer(model_name, device="cpu")
    return _MODEL_CACHE[model_name]


def _cache_key(texts: list[str], model_name: str) -> str:
    h = hashlib.sha256()
    h.update(model_name.encode())
    h.update(str(len(texts)).encode())
    for t in texts[:: max(1, len(texts) // 500)]:      # sample for speed
        h.update(t.encode("utf-8", "ignore"))
    return h.hexdigest()[:16]


def embed(
    texts: list[str],
    model_name: str = SBERT_MODEL,
    batch_size: int = 64,
    cache: bool = True,
    show_progress: bool = True,
) -> np.ndarray:
    """Encode texts to L2-normalised vectors, with an on-disk cache."""
    cache_dir = ARTIFACT_DIR / "embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path: Path | None = None
    if cache:
        path = cache_dir / f"emb_{_cache_key(texts, model_name)}.npy"
        if path.exists():
            arr = np.load(path)
            if arr.shape[0] == len(texts):
                return arr

    enc = get_encoder(model_name)
    arr = enc.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    if path is not None:
        np.save(path, arr)
    return arr


# ==========================================================================
# Supervised head on frozen embeddings
# ==========================================================================


class SbertClassifier(TextMethod):
    """Frozen SBERT encoder + logistic regression.

    Deliberately *not* fine-tuned. The point of this baseline is to isolate how
    much signal lives in generic sentence semantics, separately from how much
    comes from task-specific fine-tuning (which the RoBERTa methods provide).
    """

    def __init__(
        self,
        task_name: str,
        model_name: str = SBERT_MODEL,
        C: float = 4.0,
        max_iter: int = 2000,
    ):
        self.model_name = model_name
        self.C = C
        self.max_iter = max_iter
        self.clf = None
        self._classes: list[str] = []
        self.meta = MethodMeta(
            name=f"SBERT + LogisticRegression ({task_name})",
            family="classical-ml",
            supervised=True,
            citation="Reimers, N. & Gurevych, I. (2019). Sentence-BERT. EMNLP 2019.",
            notes=f"Frozen {model_name} embeddings, logistic-regression head. Not fine-tuned.",
            params={"C": C, "encoder": model_name},
        )

    def fit(self, texts: list[str], labels: list[str]) -> "SbertClassifier":
        from sklearn.linear_model import LogisticRegression

        X = embed(list(texts), self.model_name, show_progress=False)
        self.clf = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )
        self.clf.fit(X, labels)
        self._classes = list(self.clf.classes_)
        return self

    def predict(self, texts: list[str]) -> list[str]:
        if self.clf is None:
            raise RuntimeError("SbertClassifier.fit() must be called first.")
        X = embed(list(texts), self.model_name, show_progress=False)
        return list(self.clf.predict(X))

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        X = embed(list(texts), self.model_name, show_progress=False)
        return self.clf.predict_proba(X)


# ==========================================================================
# Similarity helpers used by Brand-Fit scoring
# ==========================================================================


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between two sets of L2-normalised vectors."""
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a @ b.T
