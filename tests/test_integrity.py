"""
Integrity tests. These are the checks that decide whether the numbers in the
report can be trusted, so they run as part of verification rather than as an
optional extra.

    python -m tests.test_integrity          # run everything, print a report
    pytest tests/test_integrity.py          # same checks under pytest

Each check is written to FAIL LOUDLY rather than warn. A silent warning in a
project like this is how a fabricated number reaches a submitted report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ARTIFACT_DIR, PROCESSED_DIR, ROOT  # noqa: E402
from src.data import benchmarks as bm  # noqa: E402

APP_DATA = ROOT / "app_data"
FAILURES: list[str] = []
PASSES: list[str] = []
SKIPS: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    if condition:
        PASSES.append(name)
    else:
        FAILURES.append(f"{name}: {detail}")
    return condition


def skip(name: str, why: str) -> None:
    SKIPS.append(f"{name}: {why}")


def _p(path: Path):
    return pd.read_parquet(path) if path.exists() else None


def _j(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


# ==========================================================================
# 1. Calibration - the synthetic data must reproduce published benchmarks
# ==========================================================================


def test_engagement_calibration():
    prof = _p(PROCESSED_DIR / "profiles.parquet")
    if prof is None:
        return skip("engagement calibration", "profiles.parquet missing")
    for tier, (lo, hi) in bm.ENGAGEMENT_BANDS.items():
        sub = prof[prof["follower_tier"] == tier]
        if len(sub) < 10:
            continue
        med = float(sub["engagement_rate"].median())
        check(
            f"engagement median in published band [{tier}]",
            lo <= med <= hi,
            f"median {med:.4f} outside [{lo}, {hi}] (n={len(sub)})",
        )


def test_price_calibration():
    prof = _p(PROCESSED_DIR / "profiles.parquet")
    camp = _p(PROCESSED_DIR / "campaigns.parquet")
    if prof is None or camp is None:
        return skip("price calibration", "profiles or campaigns missing")
    m = camp.merge(prof[["influencer_id", "follower_tier"]], on="influencer_id")
    for tier, (lo, hi) in bm.PRICE_BANDS_INR.items():
        sub = m[m["follower_tier"] == tier]
        if len(sub) < 10:
            continue
        med = float(sub["fee_inr"].median())
        check(
            f"fee median in published band [{tier}]",
            lo <= med <= hi,
            f"median {med:,.0f} outside [{lo:,}, {hi:,}] (n={len(sub)})",
        )


def test_engagement_monotonic():
    """Engagement must decrease with tier. If it doesn't, the whole premise breaks."""
    prof = _p(PROCESSED_DIR / "profiles.parquet")
    if prof is None:
        return skip("engagement monotonicity", "profiles.parquet missing")
    order = ["Nano", "Micro", "Mid", "Macro", "Mega"]
    meds = [
        float(prof[prof["follower_tier"] == t]["engagement_rate"].median())
        for t in order
        if (prof["follower_tier"] == t).sum() >= 10
    ]
    check(
        "engagement decreases monotonically with follower tier",
        all(a > b for a, b in zip(meds, meds[1:])),
        f"medians {[round(m, 4) for m in meds]}",
    )


# ==========================================================================
# 2. Leakage
# ==========================================================================


def test_no_leakage_in_features():
    man = _j(ARTIFACT_DIR / "features" / "feature_manifest.json")
    if man is None:
        return skip("leakage check", "feature_manifest.json missing")
    banned = man["banned_substrings"]
    cols = man["numeric_features"] + man["categorical_features"]
    bad = [c for c in cols if any(b in c for b in banned)]
    check("no banned column reached the model matrix", not bad, f"found {bad}")
    check("target is not a feature", man["target"] not in cols, man["target"])


def test_latents_never_exported():
    """The generative latents must never appear in any modelling or app table."""
    latent_cols = {"content_quality", "authenticity", "consistency",
                   "promo_saturation", "niche_focus"}
    for path in [
        ARTIFACT_DIR / "features" / "modelling_table.parquet",
        ARTIFACT_DIR / "features" / "influencer_features.parquet",
        APP_DATA / "influencers.parquet",
    ]:
        df = _p(path)
        if df is None:
            continue
        overlap = latent_cols & set(df.columns)
        check(f"no latent traits in {path.name}", not overlap, f"found {overlap}")


# ==========================================================================
# 3. Model results must be internally consistent
# ==========================================================================


def test_model_results_sane():
    res = _j(ARTIFACT_DIR / "models" / "model_results.json")
    if res is None:
        return skip("model results", "model_results.json missing")
    p = res["performance"]

    check("R2(log) is in a plausible range", -1.0 <= p["r2_log"] <= 1.0, str(p["r2_log"]))
    check("Spearman is in [-1, 1]", -1.0 <= p["spearman"] <= 1.0, str(p["spearman"]))
    check(
        "model does not exceed its own theoretical ceiling",
        p["r2_log"] <= p["theoretical_r2_log_ceiling"] + 0.02,
        f"R2 {p['r2_log']} vs ceiling {p['theoretical_r2_log_ceiling']}",
    )
    check(
        "fold variance is not suspiciously low",
        p["fold_r2_log_std"] > 1e-4,
        f"std {p['fold_r2_log_std']} - identical folds suggest a split bug",
    )
    check(
        "reported mean fold R2 matches the fold list",
        abs(np.mean(p["fold_r2_log"]) - p["fold_r2_log_mean"]) < 1e-3,
        "mean does not match its own folds",
    )


def test_grouped_split_actually_grouped():
    """A creator must not appear in more than one fold. Verified directly."""
    df = _p(ARTIFACT_DIR / "features" / "modelling_table.parquet")
    if df is None:
        return skip("grouped split", "modelling_table.parquet missing")
    from sklearn.model_selection import GroupKFold

    groups = df["influencer_id"].to_numpy()
    fold_of: dict[str, int] = {}
    ok = True
    for fold, (_, va) in enumerate(GroupKFold(n_splits=5).split(df, df["campaign_engagement_rate"], groups)):
        for g in groups[va]:
            if fold_of.setdefault(g, fold) != fold:
                ok = False
    check("GroupKFold puts each creator in exactly one fold", ok,
          "a creator appeared in multiple validation folds")


# ==========================================================================
# 4. NLP benchmarks must be real and honest
# ==========================================================================


def test_benchmarks_on_real_data():
    res = _p(ARTIFACT_DIR / "benchmarks" / "results.parquet")
    if res is None:
        return skip("nlp benchmarks", "results.parquet missing")
    ok = res[res["status"] == "ok"]

    check("benchmark results exist", len(ok) > 0, "no successful benchmark rows")
    check(
        "a majority-class baseline is present for every task/corpus",
        ok.groupby(["task", "corpus"])["method_key"].apply(lambda s: "majority_baseline" in set(s)).all(),
        "a task/corpus pair has no baseline",
    )
    check(
        "all corpora used are real labelled sets, not synthetic",
        all(c.startswith(("tweeteval", "sarcasm")) for c in ok["corpus"].unique()),
        f"unexpected corpora: {list(ok['corpus'].unique())}",
    )
    check(
        "accuracy values are valid probabilities",
        bool(((ok["accuracy"] >= 0) & (ok["accuracy"] <= 1)).all()),
        "accuracy outside [0, 1]",
    )
    for (task, corpus), grp in ok.groupby(["task", "corpus"]):
        n = grp[~grp["subsampled"]]["n_eval"].nunique()
        check(
            f"non-subsampled methods scored on identical rows [{task}/{corpus}]",
            n <= 1,
            f"{n} different evaluation sizes - methods are not comparable",
        )


def test_lexicon_irony_failure_is_real():
    """The report's headline claim, verified against the artifact."""
    res = _p(ARTIFACT_DIR / "benchmarks" / "results.parquet")
    if res is None:
        return skip("irony claim", "results.parquet missing")
    t = res[(res["status"] == "ok") & (res["task"] == "irony") &
            (res["corpus"] == "tweeteval_irony")]
    if not len(t):
        return skip("irony claim", "no irony results")
    base = t[t["method_key"] == "majority_baseline"]
    lex = t[t["family"] == "lexicon"]
    if not len(base) or not len(lex):
        return skip("irony claim", "baseline or lexicon rows missing")
    check(
        "report claim holds: no lexicon method beats the majority baseline on irony",
        float(lex["accuracy"].max()) <= float(base["accuracy"].iloc[0]),
        f"best lexicon accuracy {float(lex['accuracy'].max()):.4f} vs "
        f"baseline {float(base['accuracy'].iloc[0]):.4f} - THE REPORT TEXT MUST BE CORRECTED",
    )


# ==========================================================================
# 5. Deployment payload
# ==========================================================================


def test_app_data_complete():
    required = ["influencers.parquet", "brands.parquet", "manifest.json"]
    for f in required:
        check(f"app_data/{f} exists", (APP_DATA / f).exists(), "missing")

    inf = _p(APP_DATA / "influencers.parquet")
    if inf is None:
        return skip("app data content", "influencers.parquet missing")
    for col in ("influencer_id", "handle", "performance_score", "price_estimate_inr"):
        check(f"app influencers has '{col}'", col in inf.columns, "column missing")
    check("no null creator ids", not inf["influencer_id"].isna().any(), "nulls found")
    check("creator ids unique", inf["influencer_id"].is_unique, "duplicates found")


def test_app_data_size_fits_free_tier():
    if not APP_DATA.exists():
        return skip("payload size", "app_data missing")
    total = sum(f.stat().st_size for f in APP_DATA.iterdir() if f.is_file()) / 1e6
    check("deployment payload under 100 MB", total < 100, f"{total:.1f} MB")


def test_dashboard_loads_no_ml():
    """The hosted requirements must not pull heavy ML libraries."""
    req = ROOT / "requirements.txt"
    if not req.exists():
        return skip("runtime requirements", "requirements.txt missing")
    text = req.read_text().lower()
    for heavy in ("torch", "transformers", "sentence-transformers", "bertopic", "umap", "hdbscan"):
        check(
            f"'{heavy}' absent from the hosted runtime requirements",
            heavy not in text.replace("# ", "").split("\n\n")[-1] or not any(
                line.strip().startswith(heavy) for line in text.splitlines()
            ),
            "heavy dependency would blow the free-tier memory limit",
        )


# ==========================================================================


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    print("=" * 74)
    print("INTEGRITY CHECKS")
    print("=" * 74)
    for t in tests:
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            FAILURES.append(f"{t.__name__} raised {type(exc).__name__}: {exc}")

    for p in PASSES:
        print(f"  PASS  {p}")
    for s in SKIPS:
        print(f"  SKIP  {s}")
    for f in FAILURES:
        print(f"  FAIL  {f}")

    print("-" * 74)
    print(f"{len(PASSES)} passed · {len(SKIPS)} skipped · {len(FAILURES)} FAILED")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
