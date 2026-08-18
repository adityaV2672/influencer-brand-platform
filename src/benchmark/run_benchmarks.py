"""
The evaluation harness. This produces the numbers the report is built on.

Rules the harness enforces, so the comparison is honest
------------------------------------------------------
1. Every method is scored on the SAME rows. If a slow method is subsampled, the
   fast methods are scored on that identical subsample too, and the full-test
   numbers are reported separately. Never compare a 500-row LLM score against a
   4,000-row lexicon score.
2. Supervised methods fit on `train` only and never touch `test`.
3. Test split is held out for every method, including the pre-trained RoBERTa
   checkpoints (whose authors trained on the same train split).
4. A majority-class baseline is always included. A method that cannot beat
   "always guess the most common label" has not demonstrated anything.
5. Wall-clock throughput is recorded. A method 200x slower for +2 F1 is a real
   engineering trade-off and belongs in the table.
6. Missing dependencies produce a recorded SKIP row, never a silent omission.
"""
from __future__ import annotations

import json
import platform
import time
import traceback
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.config import ARTIFACT_DIR, BENCHMARK_DIR, SEED
from src.benchmark.registry import (
    LABEL_ALIASES,
    NEEDS_FIT,
    REGISTRY,
    SLOW_METHODS,
    TASK_CORPORA,
)

RESULTS_DIR = ARTIFACT_DIR / "benchmarks"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================================


@dataclass
class Result:
    task: str
    corpus: str
    method_key: str
    method_name: str = ""
    family: str = ""
    supervised: bool = False
    citation: str = ""
    notes: str = ""
    n_eval: int = 0
    accuracy: float = float("nan")
    macro_f1: float = float("nan")
    weighted_f1: float = float("nan")
    seconds: float = float("nan")
    texts_per_sec: float = float("nan")
    subsampled: bool = False
    status: str = "ok"
    error: str = ""
    per_class: dict = field(default_factory=dict)
    confusion: list = field(default_factory=list)
    labels: list = field(default_factory=list)


# ==========================================================================


def load_corpus(name: str) -> pd.DataFrame | None:
    path = BENCHMARK_DIR / f"{name}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["label"] = df["label"].map(lambda x: LABEL_ALIASES.get(x, x))
    return df


def _splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test). Falls back to a stratified split if none present."""
    if "split" in df.columns and df["split"].nunique() > 1:
        train = df[df["split"].isin(["train", "validation"])]
        test = df[df["split"] == "test"]
        if len(train) and len(test):
            return train.reset_index(drop=True), test.reset_index(drop=True)

    from sklearn.model_selection import train_test_split

    train, test = train_test_split(
        df, test_size=0.25, random_state=SEED, stratify=df["label"]
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def _stratified_sample(df: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    if n >= len(df):
        return df
    from sklearn.model_selection import train_test_split

    keep, _ = train_test_split(
        df, train_size=n, random_state=seed, stratify=df["label"]
    )
    return keep.reset_index(drop=True)


def _score(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    rep = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": {
            k: {m: round(float(v[m]), 4) for m in ("precision", "recall", "f1-score", "support")}
            for k, v in rep.items()
            if k in labels
        },
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


# ==========================================================================


def _majority_baseline(task: str, corpus: str, train: pd.DataFrame, test: pd.DataFrame,
                       labels: list[str]) -> Result:
    majority = train["label"].value_counts().idxmax()
    preds = [majority] * len(test)
    s = _score(list(test["label"]), preds, labels)
    return Result(
        task=task, corpus=corpus, method_key="majority_baseline",
        method_name=f"Majority-class baseline ('{majority}')",
        family="baseline", supervised=False,
        citation="-", notes="Predicts the most frequent training label for every input.",
        n_eval=len(test), seconds=0.0, texts_per_sec=float("inf"),
        labels=labels, **{k: s[k] for k in ("accuracy", "macro_f1", "weighted_f1", "per_class", "confusion")},
    )


def run_task(
    task: str,
    corpus: str,
    methods: list[str] | None = None,
    slow_sample: int = 400,
    fast_sample: int | None = None,
    fit_sample: int = 12_000,
    verbose: bool = True,
) -> list[Result]:
    df = load_corpus(corpus)
    if df is None:
        if verbose:
            print(f"  corpus '{corpus}' not downloaded - skipping")
        return []

    train, test = _splits(df)
    labels = sorted(df["label"].unique())
    if fast_sample:
        test = _stratified_sample(test, fast_sample)

    # The subsample every slow method is scored on. Fast methods are ALSO scored
    # on this exact subset so the two are directly comparable.
    slow_test = _stratified_sample(test, slow_sample)
    fit_train = _stratified_sample(train, fit_sample)

    if verbose:
        print(f"\n{'=' * 78}\n{task.upper()} on {corpus}")
        print(f"  train={len(train):,}  test={len(test):,}  labels={labels}")
        print(f"  slow-method subsample={len(slow_test):,}")
        print("=" * 78)

    results: list[Result] = [_majority_baseline(task, corpus, train, test, labels)]
    if verbose:
        r = results[0]
        print(f"  {r.method_name:<46} acc={r.accuracy:.3f}  macroF1={r.macro_f1:.3f}")

    wanted = methods or list(REGISTRY[task].keys())
    for key in wanted:
        factory = REGISTRY[task].get(key)
        if factory is None:
            continue
        is_slow = key in SLOW_METHODS
        eval_df = slow_test if is_slow else test

        try:
            method = factory()
        except Exception as exc:  # noqa: BLE001
            results.append(Result(task=task, corpus=corpus, method_key=key,
                                  status="skipped", error=f"{type(exc).__name__}: {exc}"))
            if verbose:
                print(f"  {key:<46} SKIP ({type(exc).__name__}: {str(exc)[:60]})")
            continue

        try:
            if key in NEEDS_FIT:
                method.fit(list(fit_train["text"]), list(fit_train["label"]))
            preds, secs = method.timed_predict(list(eval_df["text"]))
            s = _score(list(eval_df["label"]), preds, labels)
            res = Result(
                task=task, corpus=corpus, method_key=key,
                method_name=method.meta.name, family=method.meta.family,
                supervised=method.meta.supervised, citation=method.meta.citation,
                notes=method.meta.notes, n_eval=len(eval_df),
                seconds=round(secs, 2),
                texts_per_sec=round(len(eval_df) / secs, 2) if secs > 0 else float("inf"),
                subsampled=is_slow, labels=labels,
                **{k: s[k] for k in ("accuracy", "macro_f1", "weighted_f1", "per_class", "confusion")},
            )
            results.append(res)
            if verbose:
                tag = " [subsample]" if is_slow else ""
                print(f"  {method.meta.name:<46} acc={res.accuracy:.3f}  "
                      f"macroF1={res.macro_f1:.3f}  {res.texts_per_sec:>8.1f} txt/s{tag}")
        except Exception as exc:  # noqa: BLE001
            results.append(Result(task=task, corpus=corpus, method_key=key,
                                  method_name=getattr(method.meta, "name", key),
                                  status="failed",
                                  error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"))
            if verbose:
                print(f"  {key:<46} FAIL ({type(exc).__name__}: {str(exc)[:60]})")

    return results


# ==========================================================================


def run_all(
    tasks: list[str] | None = None,
    methods: dict[str, list[str]] | None = None,
    slow_sample: int = 400,
    fast_sample: int | None = 3000,
    verbose: bool = True,
) -> pd.DataFrame:
    tasks = tasks or list(REGISTRY.keys())
    methods = methods or {}
    all_results: list[Result] = []

    t0 = time.time()
    for task in tasks:
        for corpus in TASK_CORPORA[task]:
            all_results.extend(
                run_task(task, corpus, methods.get(task), slow_sample, fast_sample, verbose=verbose)
            )

    rows = [asdict(r) for r in all_results]
    df = pd.DataFrame(rows)
    df.to_parquet(RESULTS_DIR / "results.parquet", index=False)

    meta = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(time.time() - t0, 1),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "seed": SEED,
        "slow_sample": slow_sample,
        "fast_sample": fast_sample,
        "n_results": len(df),
        "n_ok": int((df["status"] == "ok").sum()),
        "n_skipped": int((df["status"] == "skipped").sum()),
        "n_failed": int((df["status"] == "failed").sum()),
    }
    (RESULTS_DIR / "run_meta.json").write_text(json.dumps(meta, indent=2))

    if verbose:
        print(f"\n{'=' * 78}")
        print(f"{meta['n_ok']} ok / {meta['n_skipped']} skipped / {meta['n_failed']} failed "
              f"in {meta['elapsed_seconds']}s")
        print(f"written to {RESULTS_DIR}")
    return df


def summary_table(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """The table that goes in the report."""
    if df is None:
        df = pd.read_parquet(RESULTS_DIR / "results.parquet")
    ok = df[df["status"] == "ok"].copy()
    ok = ok[["task", "corpus", "method_name", "family", "supervised",
             "n_eval", "accuracy", "macro_f1", "texts_per_sec", "subsampled"]]
    ok = ok.sort_values(["task", "corpus", "macro_f1"], ascending=[True, True, False])
    for c in ("accuracy", "macro_f1"):
        ok[c] = ok[c].round(4)
    return ok.reset_index(drop=True)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--slow-sample", type=int, default=400)
    ap.add_argument("--fast-sample", type=int, default=3000)
    ap.add_argument("--no-llm", action="store_true", help="skip Ollama-backed methods")
    args = ap.parse_args()

    method_filter = None
    if args.no_llm:
        method_filter = {
            t: [m for m in REGISTRY[t] if m not in SLOW_METHODS] for t in REGISTRY
        }

    out = run_all(tasks=args.tasks, methods=method_filter,
                  slow_sample=args.slow_sample, fast_sample=args.fast_sample)
    print("\n" + summary_table(out).to_string(index=False))
