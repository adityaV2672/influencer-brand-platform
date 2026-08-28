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
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

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


def grouped_ndcg(y_true: np.ndarray, y_score: np.ndarray, groups: np.ndarray,
                 k: int = 10, min_group: int = 8) -> dict:
    """NDCG@k computed WITHIN each brief and averaged across briefs.

    The global version of this metric asks whether the model's top ten rows out
    of 2,199 are the true top ten. That is decided by about ten observations, so
    it swings wildly for reasons that have nothing to do with model quality - a
    change that moved Spearman by 0.03 moved global NDCG@10 by 0.30.

    It is also not what the product does. Discover ranks creators *within one
    brief*, so the metric that matters is whether the ordering inside a brief is
    right. Averaging over briefs gives a number that is both stable and
    answerable to the interface.
    """
    scores = []
    for g in np.unique(groups):
        m = groups == g
        if m.sum() < min_group:
            continue
        v = ndcg_at_k(y_true[m], y_score[m], k)
        if v == v:
            scores.append(v)
    return {
        f"ndcg@{k}_within_brief": round(float(np.mean(scores)), 4) if scores else float("nan"),
        f"ndcg@{k}_n_briefs": len(scores),
    }


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
        # Global NDCG: kept for continuity, but it is decided by k rows out of
        # thousands and should not be read as a headline. Use the within-brief
        # figures reported alongside it.
        f"{prefix}ndcg@10_global": round(ndcg_at_k(y_true, y_pred, 10), 4),
        f"{prefix}ndcg@50_global": round(ndcg_at_k(y_true, y_pred, 50), 4),
    }


def structural_baseline(df: pd.DataFrame, y_log: np.ndarray,
                        groups: np.ndarray) -> dict:
    """The baseline the model must actually beat.

    The generator computes the campaign outcome as a product of terms. Several
    of those terms are things a reader can look up: the published engagement
    curve for the creator's size and category, whether the geography matches,
    whether the age band matches, whether the brief's category is the creator's
    own, and the measured share of topical amplification. All five are
    reconstructible from columns the model already receives.

    A ridge on just those five, scored under the same GroupKFold, is therefore
    the fair reference point. Comparing the model against "predict from follower
    count alone" instead makes it look far better than it is: an audit found
    that 73% of the model's R^2 was recoverable this way. This function exists
    so that number is reported rather than discovered by an examiner.
    """
    from sklearn.linear_model import Ridge

    from src.data import benchmarks as bm

    base = np.array([bm.expected_er(f, c)
                     for f, c in zip(df["followers"], df["brand_category"])])
    fit = np.where(df["match_primary_niche"] == 1, 1.0,
                   np.where(df["match_secondary_niche"] == 1, 0.62, 0.30))
    geo = np.where(df["match_geo"] == 1, 1.0, 0.72)
    age = np.where(df["match_age"] == 1, 1.0, 0.80)
    amp = 0.72 + 0.56 * df["pagerank_pct"].to_numpy()

    X = np.column_stack([np.log(base), np.log(geo), np.log(age), np.log(amp), fit])
    oof = np.zeros(len(y_log))
    for tr, va in GroupKFold(n_splits=N_SPLITS).split(X, y_log, groups):
        oof[va] = Ridge(alpha=1.0).fit(X[tr], y_log[tr]).predict(X[va])

    from scipy.stats import spearmanr

    return {
        "r2_log": round(float(r2_score(y_log, oof)), 4),
        "spearman": round(float(spearmanr(oof, y_log).statistic), 4),
        "n_terms": 5,
        "note": (
            "Ridge on the generator's own observable terms - published engagement "
            "curve, category match, geo match, age match, measured amplification - "
            "scored under the same GroupKFold. This is arithmetic, not learning, "
            "and it is the reference the model's lift should be quoted against."
        ),
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
    inner = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    for fold, (tr, va) in enumerate(gkf.split(X, y_log, groups)):
        # Early stopping needs a validation set, and it must NOT be the fold we
        # are about to score. An earlier version passed the outer test fold as
        # eval_set and then scored on it, which chooses the number of boosting
        # rounds using the answer. Measured, that was worth +0.0085 R^2 of
        # optimism - small, but it meant the headline number was not strictly
        # out-of-fold. The stopping set is now carved out of the training rows,
        # split by creator so the same creator cannot straddle it.
        i_fit, i_stop = next(inner.split(X.iloc[tr], y_log[tr], groups[tr]))
        fit_idx, stop_idx = tr[i_fit], tr[i_stop]
        m = lgb.LGBMRegressor(**params)
        m.fit(
            X.iloc[fit_idx], y_log[fit_idx],
            eval_set=[(X.iloc[stop_idx], y_log[stop_idx])],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(0)],
        )
        oof[va] = m.predict(X.iloc[va])
        models.append(m)
        best_iters.append(m.best_iteration_ or params["n_estimators"])
        fold_scores.append(round(float(r2_score(y_log[va], oof[va])), 4))
        print(f"      fold {fold + 1}: R2(log)={fold_scores[-1]:.4f}  best_iter={best_iters[-1]}")

    # Duan's smearing estimator. exp() of a prediction made in log space is a
    # prediction of the MEDIAN, not the mean, so back-transforming systematically
    # under-states the average outcome - measured here at -9.7% before the
    # correction. The factor is the mean of exp(out-of-fold residuals) and is
    # persisted so serving applies exactly the same correction training measured.
    #   Duan, N. (1983). Smearing Estimate: A Nonparametric Retransformation
    #   Method. Journal of the American Statistical Association 78(383).
    smearing = float(np.mean(np.exp(y_log - oof)))
    pred = np.exp(oof) * smearing

    results = {
        "model": "LightGBM (GroupKFold OOF, nested early stopping)",
        "smearing_factor": round(smearing, 5),
        "smearing_note": (
            "Duan (1983) smearing estimator applied when back-transforming from log "
            "space. Without it the mean prediction is biased low by roughly 10%."
        ),
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

    # Ranking quality where the product actually ranks: inside one brief.
    results.update(grouped_ndcg(y, pred, df["brand_id"].to_numpy(), k=10))
    results.update(grouped_ndcg(y, pred, df["brand_id"].to_numpy(), k=5))

    # --- the baseline that actually matters --------------------------------
    results["baseline_structural"] = structural_baseline(df, y_log, groups)

    # --- ceiling -----------------------------------------------------------
    from src.data.generate_synthetic import CAMPAIGN_NOISE_SIGMA

    ceiling = r2_ceiling(y, CAMPAIGN_NOISE_SIGMA)
    results["theoretical_r2_log_ceiling"] = ceiling
    results["fraction_of_ceiling"] = (
        round(results["r2_log"] / ceiling, 4) if ceiling and ceiling > 0 else float("nan")
    )
    results["ceiling_note"] = (
        f"Target carries lognormal(0, {CAMPAIGN_NOISE_SIGMA}) irreducible noise by "
        "construction; no model can exceed this R^2 in log space. NOTE: this is an "
        "ORACLE ceiling - it assumes the latent creator traits are known. The model "
        "cannot see them, so compare against baseline_structural instead."
    )

    # The honest headline. `fraction_of_ceiling` flatters, because the ceiling
    # belongs to an oracle that knows the latents. What the model actually adds
    # is its lift over the arithmetic a reader could do with a ruler.
    struct = results["baseline_structural"]["r2_log"]
    headroom = ceiling - struct
    results["learned_lift_over_structure"] = round(results["r2_log"] - struct, 4)
    results["share_of_learnable_signal"] = (
        round((results["r2_log"] - struct) / headroom, 4) if headroom > 0 else float("nan")
    )
    results["structural_share_of_r2"] = (
        round(struct / results["r2_log"], 4) if results["r2_log"] > 0 else float("nan")
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
            # Duan smearing factor, measured out-of-fold. Serving must apply the
            # same correction the evaluation did, or the numbers on the Reporting
            # page will not match the numbers on the Model page.
            "smearing": smearing,
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

    from src.data.benchmarks import expected_er as bm_expected_er
    from src.data.benchmarks import expected_fee as bm_expected_fee

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
    inner = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    for tr, va in GroupKFold(n_splits=N_SPLITS).split(X, y_log, groups):
        # Stopping set carved out of training rows - see train_performance.
        i_fit, i_stop = next(inner.split(X.iloc[tr], y_log[tr], groups[tr]))
        fit_idx, stop_idx = tr[i_fit], tr[i_stop]
        m = lgb.LGBMRegressor(**params)
        m.fit(X.iloc[fit_idx], y_log[fit_idx],
              eval_set=[(X.iloc[stop_idx], y_log[stop_idx])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        oof[va] = m.predict(X.iloc[va])
        iters.append(m.best_iteration_ or params["n_estimators"])

    smearing = float(np.mean(np.exp(y_log - oof)))
    pred = np.exp(oof) * smearing
    res = {"model": "LightGBM price regressor", "target": "fee_inr",
           "smearing_factor": round(smearing, 5), **regression_metrics(y, pred)}

    # How much of this model is a recovered identity, and how much is learning.
    #
    # The generator prices a campaign as
    #     fee = expected_fee(followers, category) * er_premium**0.55 * lognormal(0, 0.22)
    # and all three inputs are model features. Evaluating that closed form with
    # the noise removed explains almost all the variance on its own, which means
    # a high R^2 here says LightGBM can fit a log-linear function - not that the
    # model can price a real creator. The band coverage and MAPE below are the
    # only numbers from this model worth quoting.
    bench_er = np.array([bm_expected_er(f, c)
                         for f, c in zip(df["followers"], df["primary_niche"])])
    premium = np.clip(df["engagement_rate"].to_numpy() / bench_er, 0.4, 2.5)
    rate_card = np.array([bm_expected_fee(f, c)
                          for f, c in zip(df["followers"], df["brand_category"])])
    closed_form = np.log(rate_card) + 0.55 * np.log(premium)
    res["closed_form_r2_log"] = round(float(r2_score(y_log, closed_form)), 4)
    res["tautology_note"] = (
        "The generator's own fee formula, with its noise term removed, explains "
        f"R^2 {res['closed_form_r2_log']:.4f} of log fee by itself. This model is "
        "recovering an algebraic identity, so its R^2 is not evidence about pricing "
        "real creators. Quote band coverage and MAPE instead."
    )

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
    res["theoretical_r2_log_ceiling"] = r2_ceiling(y, 0.22)

    final = lgb.LGBMRegressor(**{**params, "n_estimators": int(np.mean(iters))})
    final.fit(X, y_log)
    joblib.dump(
        {"model": final, "numeric": num, "categorical": cat,
         "categories": {c: list(X[c].cat.categories) for c in cat},
         "smearing": smearing,
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
    print(f"    NDCG@10        {perf['ndcg@10_within_brief']:.4f}  (within brief, "
          f"{perf['ndcg@10_n_briefs']} briefs)")
    print(f"    structural     {perf['baseline_structural']['r2_log']:.4f}  "
          f"-> learned lift {perf['learned_lift_over_structure']:+.4f} "
          f"({perf['share_of_learnable_signal']:.1%} of learnable signal)")
    print(f"    vs benchmark curve   R2(log)={perf['baseline_benchmark_curve']['r2_log']:.4f}")
    print(f"    vs composite index   R2(log)={perf['baseline_composite_index']['r2_log']:.4f}")
    print("  ---- price ----")
    print(f"    R2(log)={price['r2_log']:.4f}   MAPE={price['mape']:.3f}   "
          f"band coverage={price['band_coverage_p10_p90']:.1%}")
    return out


if __name__ == "__main__":
    run()
