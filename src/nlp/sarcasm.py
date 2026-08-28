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
    """Predict irony from a sentiment lexicon's polarity, tuned honestly.

    The heuristic under test: ironic text tends to be *superficially* very
    positive, so strong positive surface polarity is used as an irony signal.

    Why this class fits a threshold
    -------------------------------
    An earlier version hard-coded `threshold=0.5` and fixed the direction to
    "more positive means more ironic", then reported that lexicons land below a
    majority-class baseline. That was not a fair comparison: the learned methods
    it was measured against fit their own decision boundary, and this one was
    handed an arbitrary one.

    What a fair tuning actually shows
    ---------------------------------
    Fitting the threshold AND the sign on the training split moves Bing on
    TweetEval irony from 0.496 to 0.508 accuracy and from 0.416 to 0.447 macro
    F1. The majority baseline is 0.603 accuracy and 0.376 macro F1. So the
    tuned lexicon:

      * is still WORSE than predicting the majority class on accuracy, and
      * is BETTER than it on macro F1, which is the class-balanced metric.

    Both halves matter. The lexicon is picking up something real - it beats a
    do-nothing baseline once you stop letting the majority class carry the
    score - and it is still far below any method that reads the sentence.

    A note on how easy it is to fool yourself here: an intermediate version of
    this audit reported that flipping the direction lifted Bing to 0.625
    accuracy. It does, on the test set. On the training set the original
    direction wins by a wide margin (macro F1 0.566 against 0.434), so choosing
    the flip would have meant selecting a hyper-parameter on the test data - the
    exact sin this class was rewritten to stop committing. The fit below sees
    training data only.

    One more thing worth knowing: a lexicon polarity score over a short tweet
    takes only about sixteen distinct values, so there are roughly seven usable
    thresholds. There is very little to tune, which is itself part of why the
    method cannot work.

    So the class now fits both the threshold and the sign on training data. The
    conclusion that survives is narrower and better evidenced: a tuned lexicon
    beats a do-nothing baseline but remains far below any method that reads the
    sentence, and the folk explanation for why lexicons might work on irony is
    itself wrong.
    """

    _classes = ["non_irony", "irony"]

    def __init__(self, scorer: TextMethod, threshold: float = 0.5,
                 name: str | None = None, tune: bool = True):
        self.scorer = scorer
        self.threshold = threshold
        self.direction = 1          # +1: high score -> irony;  -1: low score -> irony
        self.tune = tune
        self.meta = MethodMeta(
            name=name or f"{scorer.meta.name} -> irony heuristic",
            family="lexicon",
            supervised=False,
            citation=scorer.meta.citation,
            notes="Heuristic: surface polarity as an irony signal. Threshold and "
                  "direction are fitted on the training split so the comparison "
                  "against learned methods is like-for-like.",
            params={"threshold": threshold, "tuned": tune},
        )

    def _raw(self, texts: list[str]) -> list[float]:
        raw = getattr(self.scorer, "raw_score", None)
        if raw is None:
            raise TypeError("scorer must expose raw_score()")
        return [float(raw(t)) for t in texts]

    def fit(self, texts: list[str], labels: list[str]) -> "LexiconIronyBaseline":
        """Choose the threshold and the sign that maximise macro F1 on TRAIN."""
        if not self.tune:
            return self
        from sklearn.metrics import f1_score

        import numpy as np

        scores = np.asarray(self._raw(texts))
        y = np.asarray([1 if l == "irony" else 0 for l in labels])
        if len(np.unique(y)) < 2 or len(np.unique(scores)) < 2:
            return self
        cands = np.unique(np.quantile(scores, np.linspace(0.02, 0.98, 97)))
        best = (-1.0, self.threshold, 1)
        for t in cands:
            for d in (1, -1):
                pred = (scores >= t) if d == 1 else (scores <= t)
                f1 = f1_score(y, pred.astype(int), average="macro", zero_division=0)
                if f1 > best[0]:
                    best = (f1, float(t), d)
        _, self.threshold, self.direction = best
        self.meta.params = {"threshold": round(self.threshold, 4),
                            "direction": "high=irony" if self.direction == 1 else "low=irony",
                            "tuned_on": "train split", "train_macro_f1": round(best[0], 4)}
        return self

    def predict(self, texts: list[str]) -> list[str]:
        scores = self._raw(texts)
        if self.direction == 1:
            return ["irony" if s >= self.threshold else "non_irony" for s in scores]
        return ["irony" if s <= self.threshold else "non_irony" for s in scores]


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
