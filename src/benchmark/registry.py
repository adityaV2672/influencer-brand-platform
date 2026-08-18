"""
Registry of every method to be benchmarked, per task.

Methods are declared lazily (as factory callables) so that importing this module
does not pull a transformer into memory. The benchmark runner instantiates only
what it is asked to run, and skips - loudly - anything whose dependency is
missing, rather than silently dropping a row from the results table.
"""
from __future__ import annotations

from collections.abc import Callable

from src.nlp.base import TextMethod

# task -> {method_key: factory}
Factory = Callable[[], TextMethod]


# --------------------------------------------------------------------------
# Sentiment (3-class: negative / neutral / positive)
# --------------------------------------------------------------------------
def _bing():
    from src.nlp.lexicons import BingSentiment

    return BingSentiment()


def _vader():
    from src.nlp.lexicons import VaderSentiment

    return VaderSentiment()


def _nrc_sent():
    from src.nlp.lexicons import NRCSentiment

    return NRCSentiment()


def _sbert_sent():
    from src.nlp.embeddings import SbertClassifier

    return SbertClassifier(task_name="sentiment")


def _roberta_sent():
    from src.nlp.transformers_hf import roberta_sentiment

    return roberta_sentiment()


def _llm_sent():
    from src.nlp.sarcasm import OllamaSentiment

    return OllamaSentiment()


# --------------------------------------------------------------------------
# Emotion (4-class: anger / joy / optimism / sadness)
# --------------------------------------------------------------------------
def _nrc_emo():
    from src.nlp.lexicons import NRCEmotion

    return NRCEmotion()


def _sbert_emo():
    from src.nlp.embeddings import SbertClassifier

    return SbertClassifier(task_name="emotion")


def _roberta_emo():
    from src.nlp.transformers_hf import roberta_emotion

    return roberta_emotion()


# --------------------------------------------------------------------------
# Irony / sarcasm (binary)
# --------------------------------------------------------------------------
def _bing_irony():
    from src.nlp.lexicons import BingSentiment
    from src.nlp.sarcasm import LexiconIronyBaseline

    return LexiconIronyBaseline(BingSentiment(), threshold=0.5, name="Bing -> irony heuristic")


def _vader_irony():
    from src.nlp.lexicons import VaderSentiment
    from src.nlp.sarcasm import LexiconIronyBaseline

    return LexiconIronyBaseline(VaderSentiment(), threshold=0.5, name="VADER -> irony heuristic")


def _nrc_irony():
    from src.nlp.lexicons import NRCSentiment
    from src.nlp.sarcasm import LexiconIronyBaseline

    return LexiconIronyBaseline(NRCSentiment(), threshold=0.5, name="NRC -> irony heuristic")


def _sbert_irony():
    from src.nlp.embeddings import SbertClassifier

    return SbertClassifier(task_name="irony")


def _roberta_irony():
    from src.nlp.transformers_hf import roberta_irony

    return roberta_irony()


def _llm_irony_zero():
    from src.nlp.sarcasm import OllamaIrony

    return OllamaIrony(strategy="zero_shot")


def _llm_irony_few():
    from src.nlp.sarcasm import OllamaIrony

    return OllamaIrony(strategy="few_shot")


def _llm_irony_cot():
    from src.nlp.sarcasm import OllamaIrony

    return OllamaIrony(strategy="chain_of_thought")


# --------------------------------------------------------------------------
REGISTRY: dict[str, dict[str, Factory]] = {
    "sentiment": {
        "bing": _bing,
        "vader": _vader,
        "nrc": _nrc_sent,
        "sbert_lr": _sbert_sent,
        "roberta": _roberta_sent,
        "llm": _llm_sent,
    },
    "emotion": {
        "nrc": _nrc_emo,
        "sbert_lr": _sbert_emo,
        "roberta": _roberta_emo,
    },
    "irony": {
        "bing": _bing_irony,
        "vader": _vader_irony,
        "nrc": _nrc_irony,
        "sbert_lr": _sbert_irony,
        "roberta": _roberta_irony,
        "llm_zero_shot": _llm_irony_zero,
        "llm_few_shot": _llm_irony_few,
        "llm_cot": _llm_irony_cot,
    },
}

# Methods that are slow enough that the runner subsamples the test set for them.
SLOW_METHODS = {"llm", "llm_zero_shot", "llm_few_shot", "llm_cot"}

# Methods that need a train split fitted first.
NEEDS_FIT = {"sbert_lr"}

# Corpora each task is evaluated on. Multiple entries = cross-domain evaluation.
TASK_CORPORA = {
    "sentiment": ["tweeteval_sentiment"],
    "emotion": ["tweeteval_emotion"],
    "irony": ["tweeteval_irony", "sarcasm_headlines"],
}

# Label spaces differ between the two irony corpora; normalise to one vocabulary.
LABEL_ALIASES = {
    "sarcastic": "irony",
    "non_sarcastic": "non_irony",
}
