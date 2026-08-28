"""
Regression tests for the nine defects an ML audit of this project found.

Each test names the defect it prevents coming back. They are separate from
test_nectar.py because these are claims about *honesty of measurement* rather
than about internal consistency of the interface.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app_data"


@pytest.fixture(scope="module")
def perf():
    return json.loads((APP / "model_results.json").read_text())["performance"]


@pytest.fixture(scope="module")
def price():
    return json.loads((APP / "model_results.json").read_text())["price"]


# --------------------------------------------------------------------------
# 1-2. Circularity, and the ceiling that flattered
# --------------------------------------------------------------------------

def test_structural_baseline_is_reported(perf):
    """73% of the old R2 was recoverable by arithmetic on the model's own inputs.

    Quoting the model against "predict from follower count" hid that. The
    structural baseline has to be computed and published, or the comparison is
    dishonest by omission.
    """
    b = perf.get("baseline_structural")
    assert b and "r2_log" in b
    assert 0 < b["r2_log"] < perf["r2_log"], "the model must beat the arithmetic"


def test_learnable_signal_share_is_published(perf):
    """The honest headline: lift over structure as a share of real headroom,
    not the model's score as a share of an oracle's ceiling."""
    for k in ("learned_lift_over_structure", "share_of_learnable_signal",
              "structural_share_of_r2"):
        assert k in perf, f"{k} missing - the ceiling anchor is flattering again"
    assert 0 < perf["share_of_learnable_signal"] < 1


def test_ceiling_is_labelled_as_an_oracle_ceiling(perf):
    assert "ORACLE" in perf.get("ceiling_note", "").upper()


def test_amplification_is_not_fully_observable():
    """Campaign outcomes must not be an exact function of a model feature.

    The generator used to set amplification = 0.72 + 0.56 * pagerank_pct, and
    pagerank_pct is a feature - so "the network pillar earns its place" was a
    straight line through a column the model was handed.
    """
    from src.config import AMPLIFICATION_OBSERVED_SHARE
    assert AMPLIFICATION_OBSERVED_SHARE < 1.0


# --------------------------------------------------------------------------
# 3. The price model is a recovered identity
# --------------------------------------------------------------------------

def test_price_model_declares_its_tautology(price):
    assert "closed_form_r2_log" in price
    # The generator's own formula should explain nearly everything on its own -
    # that is the point being made, and if it stops being true the note is wrong.
    assert price["closed_form_r2_log"] > 0.9
    assert "tautology_note" in price
    assert price["band_coverage_p10_p90"] > 0.7, "the band is the number worth quoting"


# --------------------------------------------------------------------------
# 4. Label alignment
# --------------------------------------------------------------------------

def test_every_benchmark_row_carries_an_alignment_check():
    b = pd.read_parquet(APP / "benchmark_results.parquet")
    ok = b[b.status == "ok"]
    assert len(ok) > 10
    missing = [r.method_name for r in ok.itertuples()
               if not isinstance(r.alignment, dict) or "alignment_gap" not in r.alignment]
    assert not missing, f"no permutation check on: {missing}"


def test_alignment_check_catches_the_known_fault():
    """A classifier that gains half an accuracy point from being renamed is
    mis-wired, not weak. This is the row that started it."""
    b = pd.read_parquet(APP / "benchmark_results.parquet")
    flagged = [r.method_name for r in b[b.status == "ok"].itertuples()
               if (r.alignment or {}).get("label_alignment_suspect")]
    assert any("emotion" in n.lower() or "RoBERTa" in n for n in flagged), (
        "the emotion label permutation is no longer detected")


def test_alignment_check_does_not_cry_wolf():
    b = pd.read_parquet(APP / "benchmark_results.parquet")
    ok = b[b.status == "ok"]
    flagged = sum(1 for r in ok.itertuples()
                  if (r.alignment or {}).get("label_alignment_suspect"))
    assert flagged <= 2, f"{flagged} rows flagged - the threshold is too tight"


# --------------------------------------------------------------------------
# 5. Early stopping
# --------------------------------------------------------------------------

def test_early_stopping_does_not_use_the_scored_fold(perf):
    """Selecting boosting rounds on the fold you then score is worth about
    +0.0085 R2 of optimism. Measured, and removed."""
    assert "nested" in perf.get("model", "").lower()


# --------------------------------------------------------------------------
# 6. The irony heuristic must be tuned, not hand-set
# --------------------------------------------------------------------------

def test_lexicon_irony_threshold_is_fitted():
    from src.benchmark.registry import NEEDS_FIT
    from src.nlp.sarcasm import LexiconIronyBaseline

    assert {"bing", "vader", "nrc"} <= NEEDS_FIT
    assert callable(getattr(LexiconIronyBaseline, "fit", None))


def test_tuned_lexicon_still_loses_to_learned_methods():
    """The claim that survives a fair tuning is narrower than the original one.

    A tuned lexicon beats the majority baseline on macro F1 and loses to it on
    accuracy, and is far below anything that reads the sentence. If a future
    change makes a lexicon competitive with SBERT on irony, that is a bug.
    """
    b = pd.read_parquet(APP / "benchmark_results.parquet")
    irony = b[(b.task == "irony") & (b.status == "ok")]
    for corpus, g in irony.groupby("corpus"):
        lex = g[g.family == "lexicon"]
        learned = g[g.family.isin(["classical-ml", "transformer"])]
        if lex.empty or learned.empty:
            continue
        assert lex.macro_f1.max() < learned.macro_f1.max(), corpus


# --------------------------------------------------------------------------
# 7-9. Presentation defects
# --------------------------------------------------------------------------

def test_displayed_fit_spreads_across_the_range():
    f = pd.read_parquet(APP / "nectar_fit.parquet")
    q1, q3 = f.campaign_fit.quantile([0.25, 0.75])
    assert q3 - q1 > 35, f"displayed fit is bunched again ({q1:.0f}-{q3:.0f})"


def test_weight_sensitivity_is_measured():
    """The composite weights are asserted, not learned. That is only defensible
    if the shortlist does not swing on them - which has to be measured."""
    s = pd.read_parquet(APP / "nectar_weight_sensitivity.parquet")
    assert len(s) >= 15
    assert s.mean_overlap.min() > 0.5, (
        "the shortlist now depends on unvalidated weights; they need justifying")


def test_ranking_metric_is_computed_within_a_brief(perf):
    """Global NDCG@10 over 2,199 rows is decided by ten of them. Discover ranks
    inside one brief, so the metric has to as well."""
    assert "ndcg@10_within_brief" in perf
    assert perf["ndcg@10_n_briefs"] > 20


def test_predictions_are_smearing_corrected(perf):
    """exp() of a log-space prediction estimates the median, and biased every
    campaign forecast about 10% low."""
    assert perf.get("smearing_factor", 1.0) > 1.0
    import joblib
    bundle = joblib.load(ROOT / "artifacts" / "models" / "performance_model.joblib")
    assert "smearing" in bundle, "serving would not apply the correction"


def test_reported_calibration_is_centred():
    """After the correction, predicted-vs-actual should straddle zero rather
    than sitting uniformly positive."""
    cal = pd.read_parquet(APP / "nectar_calibration.parquet")
    delta = (cal.actual_er / cal.predicted_er - 1) * 100
    assert abs(delta.mean()) < 5, f"mean gap {delta.mean():.1f}% - still biased"
    assert (delta < 0).any() and (delta > 0).any(), "gap is one-signed"
