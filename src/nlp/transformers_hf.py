"""
Fine-tuned transformer classifiers from the CardiffNLP TweetEval family.

These are zero-shot from our perspective: the checkpoints were fine-tuned by
their authors on the same SemEval tasks that TweetEval packages, so we run them
as-is rather than training our own.

IMPORTANT EVALUATION CAVEAT, stated plainly in the report
---------------------------------------------------------
`cardiffnlp/twitter-roberta-base-*` checkpoints were trained on the TweetEval
*training* splits. Scoring them on the TweetEval *test* split is legitimate and
is exactly what the original paper reports. Scoring them on train/validation
rows would be leakage. The benchmark runner therefore evaluates every method on
the held-out test split only, and the supervised methods we train ourselves
(SBERT+LR) are fitted on train and never see test.
"""
from __future__ import annotations

import os

import numpy as np

from src.config import ROBERTA_EMOTION, ROBERTA_IRONY, ROBERTA_SENTIMENT
from src.nlp.base import MethodMeta, TextMethod, batched

_PIPE_CACHE: dict[str, object] = {}

# Dynamic int8 quantisation of the Linear layers.
#
# Measured on the target machine (Intel Core Ultra 5 125H, 14 threads), the
# fp32 model ran at ~5 posts/second on 51-token inputs - roughly fifty times
# slower than this hardware should manage, and thread count barely moved it,
# which rules out a parallelism problem. Dynamic int8 quantisation replaces the
# fp32 GEMMs with int8 ones at load time; it costs a fraction of a point of
# accuracy on classification and typically gives 2-4x on CPU.
#
# Set INFLUENCER_NO_QUANTIZE=1 to disable and compare.
QUANTIZE = os.environ.get("INFLUENCER_NO_QUANTIZE", "") != "1"


def _load(model_id: str):
    if model_id not in _PIPE_CACHE:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_id)
        mdl.eval()
        if QUANTIZE:
            try:
                mdl = torch.quantization.quantize_dynamic(
                    mdl, {torch.nn.Linear}, dtype=torch.qint8
                )
            except Exception as exc:  # noqa: BLE001
                print(f"    [quantisation unavailable, using fp32: {exc}]")
        _PIPE_CACHE[model_id] = (tok, mdl)
    return _PIPE_CACHE[model_id]


class HFSequenceClassifier(TextMethod):
    """Generic wrapper around a HuggingFace sequence-classification checkpoint."""

    def __init__(
        self,
        model_id: str,
        label_map: dict[str, str] | None,
        display_name: str,
        citation: str,
        notes: str = "",
        batch_size: int = 32,
        max_length: int = 128,
    ):
        self.model_id = model_id
        self.label_map = label_map or {}
        self.batch_size = batch_size
        self.max_length = max_length
        self.meta = MethodMeta(
            name=display_name,
            family="transformer",
            supervised=True,
            citation=citation,
            notes=notes or f"Pre-trained checkpoint {model_id}, run without further tuning.",
            params={"model_id": model_id},
        )
        self._classes = sorted(set(self.label_map.values())) if self.label_map else []

    # ------------------------------------------------------------------
    def _raw_logits(self, texts: list[str]) -> np.ndarray:
        import torch

        tok, mdl = _load(self.model_id)
        outs = []
        with torch.no_grad():
            for chunk in batched(list(texts), self.batch_size):
                enc = tok(
                    chunk,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                outs.append(mdl(**enc).logits.numpy())
        return np.vstack(outs)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        logits = self._raw_logits(texts)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, texts: list[str]) -> list[str]:
        _, mdl = _load(self.model_id)
        idx = self._raw_logits(texts).argmax(axis=1)
        id2label = mdl.config.id2label
        raw = [id2label[int(i)] for i in idx]
        if self.label_map:
            return [self.label_map.get(r, self.label_map.get(r.lower(), r)) for r in raw]
        return raw


# ==========================================================================
# Concrete task instances
# ==========================================================================

_CARDIFF_CITE = (
    "Barbieri, F., Camacho-Collados, J., Espinosa-Anke, L., Neves, L. (2020). "
    "TweetEval. Findings of EMNLP 2020."
)


def roberta_sentiment() -> HFSequenceClassifier:
    return HFSequenceClassifier(
        model_id=ROBERTA_SENTIMENT,
        label_map={
            "negative": "negative", "neutral": "neutral", "positive": "positive",
            "LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive",
        },
        display_name="RoBERTa (twitter-sentiment)",
        citation=_CARDIFF_CITE,
        notes="RoBERTa-base fine-tuned on ~124M tweets then on TweetEval sentiment.",
    )


def roberta_irony() -> HFSequenceClassifier:
    return HFSequenceClassifier(
        model_id=ROBERTA_IRONY,
        label_map={
            "non_irony": "non_irony", "irony": "irony",
            "LABEL_0": "non_irony", "LABEL_1": "irony",
        },
        display_name="RoBERTa (twitter-irony)",
        citation=_CARDIFF_CITE,
        notes="RoBERTa-base fine-tuned on TweetEval irony (SemEval-2018 Task 3).",
    )


def roberta_emotion() -> HFSequenceClassifier:
    return HFSequenceClassifier(
        model_id=ROBERTA_EMOTION,
        label_map={
            "anger": "anger", "joy": "joy", "optimism": "optimism", "sadness": "sadness",
            "LABEL_0": "anger", "LABEL_1": "joy", "LABEL_2": "optimism", "LABEL_3": "sadness",
        },
        display_name="RoBERTa (twitter-emotion)",
        citation=_CARDIFF_CITE,
        notes="RoBERTa-base fine-tuned on TweetEval emotion (SemEval-2018 Task 1).",
    )
