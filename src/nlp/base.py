"""
Common interface for every text-analysis method in the project.

Every sentiment / emotion / sarcasm method - whether it is a 1990s word list, a
transformer, or an LLM prompt - implements the same three-method interface so
that `src/benchmark/` can evaluate them all through one loop and produce a
like-for-like comparison table. That comparability is the entire point: the
project's central empirical claim is "method X beats method Y on real labelled
data", and that claim is only credible if every method saw exactly the same
inputs and was scored exactly the same way.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class MethodMeta:
    """Everything the report needs to describe a method honestly."""

    name: str
    family: str          # lexicon | classical-ml | transformer | llm
    supervised: bool
    citation: str
    notes: str = ""
    params: dict = field(default_factory=dict)


class TextMethod(ABC):
    """Base class for a text-classification method."""

    meta: MethodMeta

    # -- lifecycle ---------------------------------------------------------
    def fit(self, texts: list[str], labels: list[str]) -> "TextMethod":
        """Unsupervised / zero-shot methods ignore this."""
        return self

    @abstractmethod
    def predict(self, texts: list[str]) -> list[str]:
        """Return one predicted label per input text."""

    def predict_proba(self, texts: list[str]) -> np.ndarray | None:
        """Optional. Return (n_texts, n_classes) probabilities."""
        return None

    # -- convenience -------------------------------------------------------
    @property
    def classes_(self) -> list[str]:
        return getattr(self, "_classes", [])

    def timed_predict(self, texts: list[str]) -> tuple[list[str], float]:
        """Predict and report wall-clock seconds - throughput is a real
        deployment constraint and belongs in the comparison table."""
        t0 = time.perf_counter()
        preds = self.predict(texts)
        return preds, time.perf_counter() - t0

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{self.__class__.__name__} {self.meta.name}>"


def batched(seq: list, size: int):
    """Yield successive chunks - used by every transformer/LLM method."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]
