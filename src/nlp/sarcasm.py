"""
Sarcasm / irony detection - the method family the supervisor specifically asked
for ("LLM Prompting (this traces the sarcasm)").

Why sarcasm is the interesting case
-----------------------------------
Sarcasm is where lexicon sentiment analysis structurally fails, and it fails in
a way that is easy to demonstrate and impossible to argue with. "Oh great,
another subscription fee. Brilliant." contains {great, brilliant} and zero
negative words, so Bing scores it maximally positive. VADER scores it positive
too - its negation rules cover "not good", not ironic praise. The failure is not
a tuning problem; the information needed to resolve it is not in the words.

That gives the project a clean experimental spine:

    Bing        - no context at all              (expected: fails)
    VADER       - local syntactic context        (expected: fails)
    SBERT + LR  - learned sentence semantics     (expected: partial)
    RoBERTa     - fine-tuned on labelled irony   (expected: strong, in-domain)
    LLM prompt  - world knowledge + reasoning    (expected: strong, transfers)

and a genuinely interesting question: RoBERTa-irony is fine-tuned *on this
exact task*, so it should win in-domain. Does it still win when moved to a
different domain (news headlines)? That is the cross-domain test, and it is
where LLM prompting tends to earn its cost.

Three prompting strategies are implemented so the LLM result is not a single
lucky prompt:
    zero-shot         - plain instruction
    few-shot          - four hand-written labelled examples
    chain-of-thought  - reason first, then answer
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request

import numpy as np

from src.config import OLLAMA_HOST, OLLAMA_MODEL
from src.nlp.base import MethodMeta, TextMethod


# ==========================================================================
# Naive lexicon baselines (expected to fail - that is the finding)
# ==========================================================================


class LexiconIronyBaseline(TextMethod):
    """Predict irony from a sentiment lexicon's polarity.

    The heuristic under test: ironic text tends to be *superficially* very
    positive. So "strongly positive surface sentiment" is used as an irony
    signal. This is the strongest reasonable thing a word list can do, and
    documenting that it lands near chance is the point.
    """

    _classes = ["non_irony", "irony"]

    def __init__(self, scorer: TextMethod, threshold: float = 0.5, name: str | None = None):
        self.scorer = scorer
        self.threshold = threshold
        self.meta = MethodMeta(
            name=name or f"{scorer.meta.name} -> irony heuristic",
            family="lexicon",
            supervised=False,
            citation=scorer.meta.citation,
            notes="Heuristic: strongly positive surface polarity is treated as ironic. "
                  "Included to quantify the failure mode, not as a serious detector.",
            params={"threshold": threshold},
        )

    def predict(self, texts: list[str]) -> list[str]:
        raw = getattr(self.scorer, "raw_score", None)
        if raw is None:
            raise TypeError("scorer must expose raw_score()")
        return ["irony" if raw(t) >= self.threshold else "non_irony" for t in texts]


# ==========================================================================
# LLM prompting via Ollama
# ==========================================================================

ZERO_SHOT = """You are an expert linguist annotating social media text for irony and sarcasm.

Sarcasm/irony means the writer's intended meaning is the opposite of, or sharply at odds with, the literal words - typically mock praise, feigned enthusiasm, or exaggerated understatement used to criticise.

Text: {text}

Answer with exactly one word, either IRONIC or LITERAL. No explanation."""


FEW_SHOT = """You are an expert linguist annotating social media text for irony and sarcasm.

Sarcasm/irony means the writer's intended meaning is the opposite of, or sharply at odds with, the literal words.

Examples:

Text: Oh fantastic, the update deleted all my settings. Exactly what I wanted today.
Answer: IRONIC

Text: The battery lasted about six hours in my testing, which is reasonable for this price.
Answer: LITERAL

Text: Love how the "waterproof" case survived precisely one rainstorm. Great engineering.
Answer: IRONIC

Text: I've been using this serum for three weeks and my skin does look calmer.
Answer: LITERAL

Text: {text}
Answer:"""


CHAIN_OF_THOUGHT = """You are an expert linguist annotating social media text for irony and sarcasm.

Text: {text}

Think step by step:
1. What is the literal surface sentiment of the words?
2. Is there any cue that the writer means the opposite - mock praise, an absurd or unwanted outcome described positively, exaggeration, scare quotes, or a mismatch between the praise and the thing being praised?
3. Would a typical reader take this at face value?

Then give your final answer on a new line in exactly this format:
ANSWER: IRONIC
or
ANSWER: LITERAL"""


PROMPTS = {
    "zero_shot": ZERO_SHOT,
    "few_shot": FEW_SHOT,
    "chain_of_thought": CHAIN_OF_THOUGHT,
}


def ollama_available(host: str = OLLAMA_HOST, timeout: int = 3) -> bool:
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def ollama_models(host: str = OLLAMA_HOST) -> list[str]:
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=5) as r:
            data = json.loads(r.read().decode())
        return [m["name"] for m in data.get("models", [])]
    except Exception:  # noqa: BLE001
        return []


class OllamaIrony(TextMethod):
    """Irony detection by prompting a locally-hosted instruction-tuned LLM.

    Runs entirely on the user's machine via Ollama - no API key, no per-token
    cost, no data leaving the device. Slower than the transformer methods, which
    the benchmark reports as throughput rather than hiding.
    """

    _classes = ["non_irony", "irony"]

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        strategy: str = "few_shot",
        host: str = OLLAMA_HOST,
        temperature: float = 0.0,
        num_predict: int = 200,
        max_retries: int = 2,
    ):
        if strategy not in PROMPTS:
            raise ValueError(f"strategy must be one of {list(PROMPTS)}")
        self.model = model
        self.strategy = strategy
        self.host = host
        self.temperature = temperature
        self.num_predict = num_predict if strategy == "chain_of_thought" else 8
        self.max_retries = max_retries
        self.n_unparsed = 0
        self.meta = MethodMeta(
            name=f"LLM prompting ({model}, {strategy})",
            family="llm",
            supervised=False,
            citation="Local instruction-tuned LLM served via Ollama (https://ollama.com).",
            notes="Zero-shot/few-shot prompting. Deterministic decoding (temperature=0). "
                  "Runs locally; no data leaves the machine.",
            params={"model": model, "strategy": strategy, "temperature": temperature},
        )

    # ------------------------------------------------------------------
    def _generate(self, prompt: str) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.num_predict,
                    "top_p": 1.0,
                    "seed": 42,
                },
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    return json.loads(r.read().decode()).get("response", "")
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Ollama generation failed after retries: {last_exc}")

    @staticmethod
    def _parse(response: str) -> str | None:
        """Extract the verdict. Returns None if the model did not comply."""
        text = response.strip().upper()
        m = re.search(r"ANSWER\s*:\s*(IRONIC|LITERAL)", text)
        if m:
            return "irony" if m.group(1) == "IRONIC" else "non_irony"
        # Fall back to the last standalone occurrence.
        hits = re.findall(r"\b(IRONIC|LITERAL|SARCASTIC|SINCERE)\b", text)
        if hits:
            last = hits[-1]
            return "irony" if last in {"IRONIC", "SARCASTIC"} else "non_irony"
        return None

    def predict(self, texts: list[str]) -> list[str]:
        tmpl = PROMPTS[self.strategy]
        out: list[str] = []
        self.n_unparsed = 0
        for t in texts:
            resp = self._generate(tmpl.format(text=t.strip()[:1200]))
            verdict = self._parse(resp)
            if verdict is None:
                self.n_unparsed += 1
                verdict = "non_irony"      # majority class, recorded not hidden
            out.append(verdict)
        return out


class OllamaSentiment(TextMethod):
    """Three-class sentiment by prompting the same local LLM.

    Included so the sentiment table has an LLM row directly comparable to the
    lexicon and transformer rows.
    """

    _classes = ["negative", "neutral", "positive"]

    PROMPT = """Classify the sentiment the writer intends to express.

If the text is sarcastic or ironic, classify the INTENDED sentiment, not the literal words. Mock praise is negative.

Text: {text}

Answer with exactly one word: POSITIVE, NEGATIVE, or NEUTRAL. No explanation."""

    def __init__(self, model: str = OLLAMA_MODEL, host: str = OLLAMA_HOST):
        self.model = model
        self.host = host
        self.n_unparsed = 0
        self._irony = OllamaIrony(model=model, host=host)   # reuse transport
        self.meta = MethodMeta(
            name=f"LLM prompting ({model}, sentiment)",
            family="llm",
            supervised=False,
            citation="Local instruction-tuned LLM served via Ollama.",
            notes="Explicitly instructed to resolve sarcasm to intended sentiment.",
            params={"model": model},
        )

    def predict(self, texts: list[str]) -> list[str]:
        out = []
        self.n_unparsed = 0
        for t in texts:
            resp = self._irony._generate(self.PROMPT.format(text=t.strip()[:1200]))
            up = resp.strip().upper()
            hits = re.findall(r"\b(POSITIVE|NEGATIVE|NEUTRAL)\b", up)
            if hits:
                out.append(hits[-1].lower())
            else:
                self.n_unparsed += 1
                out.append("neutral")
        return out
