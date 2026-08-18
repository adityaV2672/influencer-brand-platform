"""
Model training and evaluation.

Three models are trained:

  1. Performance  - predicts sponsored-campaign engagement rate (the target the
                    proposal's Phase 2 describes).
  2. Price        - predicts the negotiated fee in INR.
  3. Brand-fit    - not learned; a transparent composite of semantic similarity
                    and hard eligibility gates. See src/models/brandfit.py for
                    why that is the right call rather than a cop-out.

Evaluation discipline
---------------------
* GroupKFold on influencer_id. A creator appears in up to three campaigns; a
  random split would put the same creator in train and test and inflate every
  metric. This single decision changes the reported R^2 substantially and is the
  most common way academic projects of this shape overstate their results.
* Metrics in log space, because engagement rate is log-normally distributed and
  a plain R^2 on the raw scale is dominated by a handful of large values.
* Ranking metrics (Spearman, NDCG@10) alongside regression metrics, because the
  product ranks creators - being right about the *order* matters more than the
  absolute number.
* Two baselines that must be beaten:
      - the published benchmark curve alone (a lookup table, no ML)
      - the Phase-1 weighted composite index from the original proposal
  If the learned model cannot beat a hand-weighted index, the ML layer is not
  earning its complexity and the report should say so.
* The theoretical R^2 ceiling is computed from the known noise variance, so the
  model's performance is reported as a fraction of what is achievable rather
  than as an unanchored number.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from src.config import ARTIFACT_DIR, PILLAR_WEIGHTS, SEED
from src.features.build_features import (
    FEATURE_DIR,
    assert_no_leakage,
    feature_columns,
)

warnings.filterwarnings("ignore", category=UserWarning)
try:
    from lightgbm import LGBMDeprecationWarning
    warnings.filterwarnings("ignore", category=LGBMDeprecationWarning)
except Exception:
    warnings.filterwarnings("ignore", message=".*eval_set.*")
    warnings.filterwarnings("ignore", message=".*deprecated.*")

MODEL_DIR = ARTIFACT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5


# ==========================================================================
# Serving helpers
# ==========================================================================


def apply_categories(X: pd.DataFrame, categories: dict[str, list]) -> pd.DataFrame:
    """Re-apply the training category sets to a prediction frame.

    LightGBM encodes a pandas categorical by its integer code, not by its label.
    If a prediction frame has a different set or order of categories, every
    categorical feature is silently mis-mapped - no exception, just wrong
    numbers. Anything unseen in training becomes NaN, which LightGBM handles as
    a missing value.
    """
    X = X.copy()
    for col, cats in (categories or {}).items():
        if col in X.columns:
            X[col] = pd.Categorical(X[col].astype("object"), categories=cats)
    return X


# ==========================================================================
# Metrics
# ==========================================================================


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    """NDCG using the true value as the relevance grade."""
    order = np.argsort(-y_score)[:k]
    gains = y_true[order]
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float((gains * discounts).sum())
    ideal = np.sort(y_true)[::-1][:k]
    idcg = float((ideal * discounts[: len(ideal)]).sum())
    return dcg / idcg if idcg > 0 else float("nan")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> dict:
    from scipy.stats import spearmanr

    lt, lp = np.log(np.clip(y_true, 1e-9, None)), np.log(np.clip(y_pred, 1e-9, None))
    return {
        f"{prefix}r2_log": round(float(r2_score(lt, lp)), 4),
        f"{prefix}rmse_log": round(float(np.sqrt(mean_squared_error(lt, lp))), 4),
        f"{prefix}mae_log": round(float(mean_absolute_error(lt, lp)), 4),
        f"{prefix}r2_raw": round(float(r2_score(y_true, y_pred)), 4),
        f"{prefix}mae_raw": round(float(mean_absolute_error(y_true, y_pred)), 6),
        f"{prefix}mape": round(float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-9, None)))), 4),
        f"{prefix}spearman": round(float(spearmanr(y_true, y_pred).statistic), 4),
        f"{prefix}ndcg@10": round(ndcg_at_k(y_true, y_pred, 10), 4),
        f"{prefix}ndcg@50": round(ndcg_at_k(y_true, y_pred, 50), 4),
    }


def r2_ceiling(y: np.ndarray, noise_sigma: float) -> float:
    """Max achievable R^2 in log space given known multiplicative noise."""
    var_total = float(np.var(np.log(np.clip(y, 1e-9, None))))
    return round(max(0.0, 1.0 - (noise_sigma ** 2) / var_total), 4) if var_total > 0 else float("nan")


# ==========================================================================
# Baselines
# ==========================================================================


def benchmark_baseline(df: pd.DataFrame) -> np.ndarray:
    """Published rate-card curve only. No learning at all."""
    from src.data import benchmarks as bm
    from src.data.generate_synthetic import SPONSORED_DISCOUNT

    return np.array([
        bm.expected_er(f, c) * SPONSORED_DISCOUNT
        for f, c in zip(df["followers"], df["brand_category"])
    ])


def composite_index(df: pd.DataFrame, weights: dict | None = None) -> np.ndarray:
    """The Phase-1 transparent weighted index from the original proposal.

    Each pillar is min-max normalised to 0-1 across the population, then
    combined with fixed domain-judgement weights. This is what the project
    would ship without a learned model, and it is the bar the learned model has
    to clear to justify itself.
    """
    w = weights or PILLAR_WEIGHTS

    def norm(col: str) -> np.ndarray:
        if col not in df.columns:
            return np.zeros(len(df))
        v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        v = np.nan_to_num(v, nan=np.nanmedian(v) if np.isfinite(np.nanmedian(v)) else 0.0)
        lo, hi = np.percentile(v, 1), np.percentile(v, 99)
        return np.clip((v - lo) / (hi - lo + 1e-12), 0, 1)

    reach = 0.5 * norm("log_followers") + 0.3 * norm("avg_views") + 0.2 * norm("follower_growth_rate")
    engagement = 0.6 * norm("engagement_rate") + 0.25 * norm("comments_to_likes") + 0.15 * norm("views_to_followers")
    content = (
        0.4 * (1 - norm("content_promo_rate"))
        + 0.3 * norm("content_vader_mean")
        + 0.3 * (1 - norm("content_outlier_rate"))
    )
    network = 0.5 * norm("pagerank") + 0.3 * norm("eigenvector_centrality") + 0.2 * norm("degree_centrality")

    score = (
        w["reach"] * reach + w["engagement"] * engagement
        + w["content"] * content + w["network"] * network
    )
    return score


# ==========================================================================
# Performance model
# ==========================================================================


def train_performance(df: pd.DataFrame, target: str = "campaign_engagement_rate") -> dict:
    import lightgbm as lgb

    num, cat = feature_columns(df)
    assert_no_leakage(num + cat)

    X = df[num + cat].copy()
    for c in cat:
        X[c] = X[c].astype("category")
    y = df[target].to_numpy(dtype=float)
    y_log = np.log(np.clip(y, 1e-9, None))
    groups = df["influencer_id"].to_numpy()

    print(f"    {len(df):,} rows, {len(num)} numeric + {len(cat)} categorical features")
    print(f"    GroupKFold({N_SPLITS}) on influencer_id "
          f"({pd.Series(groups).nunique():,} unique creators)")

    params = dict(
        objective="regression", metric="l2", learning_rate=0.045,
        num_leaves=31, min_child_samples=25, feature_fraction=0.75,
        bagging_fraction=0.85, bagging_freq=1, lambda_l2=1.5,
        n_estimators=1600, verbosity=-1, random_state=SEED, n_jobs=-1,
    )

    oof = np.zeros(len(df))
    fold_scores, models, best_iters = [], [], []
    gkf = GroupKFold(n_splits=N_SPLITS)
    for fold, (tr, va) in enumerate(gkf.split(X, y_log, groups)):
        m = lgb.LGBMRegressor(**params)
        m.fit(
            X.iloc[tr], y_log[tr],
            eval_set=[(X.iloc[va], y_log[va])],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(0)],
        )
        oof[va] = m.predict(X.iloc[va])
        models.append(m)
        best_iters.append(m.best_iteration_ or params["n_estimators"])
        fold_scores.append(round(float(r2_score(y_log[va], oof[va])), 4))
        print(f"      fold {fold + 1}: R2(log)={fold_scores[-1]:.4f}  best_iter={best_iters[-1]}")

    pred = np.exp(oof)
    results = {
        "model": "LightGBM (GroupKFold OOF)",
        "target": target,
        "n_rows": len(df),
        "n_features": len(num) + len(cat),
        "fold_r2_log": fold_scores,
        "fold_r2_log_mean": round(float(np.mean(fold_scores)), 4),
        "fold_r2_log_std": round(float(np.std(fold_scores)), 4),
        "mean_best_iteration": int(np.mean(best_iters)),
        **regression_metrics(y, pred),
    }

    # --- baselines, scored on exactly the same rows ------------------------
    bench = benchmark_baseline(df)
    results["baseline_benchmark_curve"] = regression_metrics(y, bench)

    comp = composite_index(df)
    # The index is unitless; map it onto the target scale by isotonic fit so the
    # comparison is about *ranking quality*, not about calibration.
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip").fit(comp, y)
    results["baseline_composite_index"] = regression_metrics(y, np.clip(iso.predict(comp), 1e-9, None))
    results["baseline_composite_index"]["note"] = (
        "Isotonically calibrated to the target scale, and fitted on the full set - "
        "this FLATTERS the baseline, so beating it is a conservative claim."
    )

    # --- ceiling -----------------------------------------------------------
    from src.data.generate_synthetic import CAMPAIGN_NOISE_SIGMA

    ceiling = r2_ceiling(y, CAMPAIGN_NOISE_SIGMA)
    results["theoretical_r2_log_ceiling"] = ceiling
    results["fraction_of_ceiling"] = (
        round(results["r2_log"] / ceiling, 4) if ceiling and ceiling > 0 else float("nan")
    )
    results["ceiling_note"] = (
        f"Target carries lognormal(0, {CAMPAIGN_NOISE_SIGMA}) irreducible noise by "
        "construction; no model can exceed this R^2 in log space."
    )

    # --- feature importance + SHAP ----------------------------------------
    imp = pd.DataFrame(
        {"feature": X.columns,
         "gain": np.mean([m.booster_.feature_importance("gain") for m in models], axis=0)}
    ).sort_values("gain", ascending=False)
    imp["gain_pct"] = (imp["gain"] / imp["gain"].sum() * 100).round(2)
    imp.to_parquet(MODEL_DIR / "performance_importance.parquet", index=False)
    results["top_features"] = imp.head(20)[["feature", "gain_pct"]].to_dict("records")

    try:
        import shap

        sample = X.sample(n=min(600, len(X)), random_state=SEED)
        sv = shap.TreeExplainer(models[0]).shap_values(sample)
        shap_imp = pd.DataFrame(
            {"feature": X.columns, "mean_abs_shap": np.abs(sv).mean(axis=0)}
        ).sort_values("mean_abs_shap", ascending=False)
        shap_imp.to_parquet(MODEL_DIR / "performance_shap.parquet", index=False)
        results["top_features_shap"] = shap_imp.head(15).to_dict("records")
    except Exception as exc:  # noqa: BLE001
        results["shap_error"] = f"{type(exc).__name__}: {exc}"

    # --- pillar ablation ---------------------------------------------------
    results["ablation"] = _ablation(X, y_log, groups, num, cat, params)

    # --- refit on everything for serving -----------------------------------
    final = lgb.LGBMRegressor(**{**params, "n_estimators": int(np.mean(best_iters))})
    final.fit(X, y_log)
    import joblib

    joblib.dump(
        {
            "model": final,
            "numeric": num,
            "categorical": cat,
            "log_target": True,
            # Exact training categories, in order. See apply_categories().
            "categories": {c: list(X[c].cat.categories) for c in cat},
        },
        MODEL_DIR / "performance_model.joblib",
    )
    np.save(MODEL_DIR / "performance_oof.npy", pred)
    return results


def _ablation(X, y_log, groups, num, cat, params) -> dict:
    """Drop each pillar and measure the damage. This is what tells you whether
    the SNA and NLP pillars are pulling their weight or are decoration."""
    import lightgbm as lgb

    pillars = {
        "reach": ["followers", "log_followers", "avg_views", "avg_reach", "follower_growth_rate",
                  "posting_frequency_month", "following", "follower_following_ratio", "reach_efficiency"],
        "engagement": ["engagement_rate", "avg_likes", "avg_comments", "comments_to_likes",
                       "views_to_followers", "engagement_per_post", "er_vs_benchmark"],
        "content": [c for c in X.columns if c.startswith("content_")],
        "network": ["degree_centrality", "degree_weighted", "pagerank", "eigenvector_centrality",
                    "betweenness_centrality", "closeness_centrality", "clustering_coefficient",
                    "k_core", "community", "community_size", "network_tier"]
                   + [c for c in X.columns if c.endswith("_pct")],
        "brandfit": ["match_primary_niche", "match_secondary_niche", "match_geo",
                     "match_age", "log_budget", "brand_category"],
    }

    def _cv(cols) -> float:
        Xs = X[cols]
        gkf = GroupKFold(n_splits=N_SPLITS)
        scores = []
        for tr, va in gkf.split(Xs, y_log, groups):
            m = lgb.LGBMRegressor(**{**params, "n_estimators": 700})
            m.fit(Xs.iloc[tr], y_log[tr],
                  eval_set=[(Xs.iloc[va], y_log[va])],
                  callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
            scores.append(r2_score(y_log[va], m.predict(Xs.iloc[va])))
        return round(float(np.mean(scores)), 4)

    full = _cv(list(X.columns))
    out = {"all_features_r2_log": full, "drops": {}, "only": {}}
    for name, cols in pillars.items():
        present = [c for c in cols if c in X.columns]
        if not present:
            continue
        remaining = [c for c in X.columns if c not in present]
        if remaining:
            without = _cv(remaining)
            out["drops"][name] = {
                "r2_without": without,
                "delta": round(full - without, 4),
                "n_features": len(present),
            }
        out["only"][name] = _cv(present)
        print(f"      ablation {name:<11} drop={out['drops'].get(name, {}).get('delta', float('nan')):+.4f}  "
              f"alone={out['only'][name]:.4f}")
    return out


# ==========================================================================
# Price model
# ==========================================================================


def train_price(df: pd.DataFrame) -> dict:
    import lightgbm as lgb
    import joblib

    num, cat = feature_columns(df)
    # The fee is agreed before the campaign runs, so campaign outcome features
    # must not be used. feature_columns already excludes them.
    X = df[num + cat].copy()
    for c in cat:
        X[c] = X[c].astype("category")
    y = df["fee_inr"].to_numpy(dtype=float)
    y_log = np.log(np.clip(y, 1.0, None))
    groups = df["influencer_id"].to_numpy()

    params = dict(
        objective="regression", learning_rate=0.05, num_leaves=31,
        min_child_samples=25, feature_fraction=0.8, bagging_fraction=0.85,
        bagging_freq=1, lambda_l2=1.0, n_estimators=1200,
        verbosity=-1, random_state=SEED, n_jobs=-1,
    )

    oof = np.zeros(len(df))
    iters = []
    for tr, va in GroupKFold(n_splits=N_SPLITS).split(X, y_log, groups):
        m = lgb.LGBMRegressor(**params)
        m.fit(X.iloc[tr], y_log[tr], eval_set=[(X.iloc[va], y_log[va])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        oof[va] = m.predict(X.iloc[va])
        iters.append(m.best_iteration_ or params["n_estimators"])

    pred = np.exp(oof)
    res = {"model": "LightGBM price regressor", "target": "fee_inr", **regression_metrics(y, pred)}

    # Rule-based Phase-1 baseline: the published rate card.
    from src.data import benchmarks as bm

    rule = np.array([bm.expected_fee(f, c) for f, c in zip(df["followers"], df["brand_category"])])
    res["baseline_rate_card"] = regression_metrics(y, rule)

    # Prediction interval from the OOF residual spread in log space - this is
    # what the dashboard shows as a price *band* rather than a false point
    # estimate.
    resid = np.log(np.clip(y, 1, None)) - oof
    lo_q, hi_q = np.percentile(resid, [10, 90])
    res["band_multipliers"] = {"low": round(float(np.exp(lo_q)), 3), "high": round(float(np.exp(hi_q)), 3)}
    res["band_coverage_p10_p90"] = round(
        float(np.mean((y >= pred * np.exp(lo_q)) & (y <= pred * np.exp(hi_q)))), 4
    )

    final = lgb.LGBMRegressor(**{**params, "n_estimators": int(np.mean(iters))})
    final.fit(X, y_log)
    joblib.dump(
        {"model": final, "numeric": num, "categorical": cat,
         "categories": {c: list(X[c].cat.categories) for c in cat},
         "band": {"low": float(np.exp(lo_q)), "high": float(np.exp(hi_q))}},
        MODEL_DIR / "price_model.joblib",
    )
    return res


# ==========================================================================


def run() -> dict:
    df = pd.read_parquet(FEATURE_DIR / "modelling_table.parquet")

    print("  training performance model ...")
    perf = train_performance(df)
    print("  training price model ...")
    price = train_price(df)

    out = {"performance": perf, "price": price}
    (MODEL_DIR / "model_results.json").write_text(json.dumps(out, indent=2, default=str))

    print("\n  ---- performance ----")
    print(f"    R2(log)        {perf['r2_log']:.4f}   (ceiling {perf['theoretical_r2_log_ceiling']}, "
          f"{perf['fraction_of_ceiling']:.1%} of achievable)")
    print(f"    Spearman       {perf['spearman']:.4f}")
    print(f"    NDCG@10        {perf['ndcg@10']:.4f}")
    print(f"    vs benchmark curve   R2(log)={perf['baseline_benchmark_curve']['r2_log']:.4f}")
    print(f"    vs composite index   R2(log)={perf['baseline_composite_index']['r2_log']:.4f}")
    print("  ---- price ----")
    print(f"    R2(log)={price['r2_log']:.4f}   MAPE={price['mape']:.3f}   "
          f"band coverage={price['band_coverage_p10_p90']:.1%}")
    return out


if __name__ == "__main__":
    run()
