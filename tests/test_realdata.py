"""
The one dataset in this project that nobody here generated.

The checks that matter are provenance checks. A synthetic dataset can be
regenerated if it is wrong; a real one that has been silently swapped, drifted
or truncated produces a result that looks fine and means nothing. So the file
is pinned by digest, its schema is checked against the publication, and a test
fails if any of that stops holding.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.realdata import news_popularity as NP  # noqa: E402

RESULTS = ROOT / "artifacts" / "realdata" / "news_popularity_results.json"

pytestmark = pytest.mark.skipif(
    not NP.CSV.exists(),
    reason="real dataset not fetched; run python -m src.realdata.train_news")


@pytest.fixture(scope="module")
def results():
    if not RESULTS.exists():
        pytest.skip("model not trained yet")
    return json.loads(RESULTS.read_text())


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------
def test_the_cached_file_is_the_file_we_pinned():
    assert hashlib.sha256(NP.CSV.read_bytes()).hexdigest() == NP.SHA256


def test_shape_and_schema_match_the_publication():
    df = NP.load()                     # load() raises if either drifts
    assert df.shape == NP.EXPECTED_SHAPE
    assert NP.TARGET in df.columns


def test_non_predictive_columns_never_reach_the_model():
    """`timedelta` encodes the collection window, not anything a publisher
    could know before pressing publish. Leaving it in is a leak."""
    df = NP.load()
    X, _ = NP.features(df)
    for c in NP.NON_PREDICTIVE:
        assert c not in X.columns
    assert NP.TARGET not in X.columns


def test_provenance_record_is_written_and_honest():
    prov = json.loads(NP.PROVENANCE.read_text())
    assert prov["sha256"] == NP.SHA256
    assert len(prov["mirrors_checked"]) >= 3
    assert "REAL" in prov["provenance"]
    assert "archive.ics.uci.edu" in prov["canonical_host"]
    # The reason for not using the canonical host must be recorded, not hidden.
    assert prov["reason_not_canonical"]


def test_a_wrong_digest_is_refused(tmp_path, monkeypatch):
    """The guard has to actually fire, or it is decoration."""
    monkeypatch.setattr(NP, "SHA256", "0" * 64)
    monkeypatch.setattr(NP, "CSV", tmp_path / "x.csv")
    with pytest.raises(RuntimeError):
        NP.fetch(force=True)


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------
def test_the_model_beats_its_baselines(results):
    reg = results["regression"]
    assert reg["random_kfold"]["r2_log"] > reg["baseline_median"]["r2_log"]
    assert reg["random_kfold"]["r2_log"] > reg["baseline_ridge_all_features"]["r2_log"]
    clf = results["classification"]
    assert clf["lightgbm"]["accuracy"] > clf["majority_baseline"]["accuracy"]
    assert clf["lightgbm"]["roc_auc"] > 0.5


def test_the_result_is_in_the_range_published_work_reports(results):
    """Wildly beating the literature on a well-studied dataset is a bug report,
    not a breakthrough."""
    acc = results["classification"]["lightgbm"]["accuracy"]
    assert 0.60 < acc < 0.75, f"{acc} is outside the published range for this task"


def test_the_chronological_split_is_reported_beside_the_random_one(results):
    """A random split over a time-ordered corpus flatters. Both are reported so
    the gap is visible rather than argued about."""
    reg = results["regression"]
    assert "chronological" in reg
    assert reg["chronological"]["n_test"] > 1000


def test_smearing_is_applied_on_the_back_transform(results):
    """Same correction the project's own model uses; without it the mean
    prediction is biased low."""
    assert results["regression"]["random_kfold"]["smearing_factor"] > 1.0


def test_the_comparison_does_not_flatter_the_synthetic_model(results):
    """The headline of this whole exercise. If this ever inverts, something has
    gone wrong with the synthetic pipeline, not with reality."""
    c = results["comparison"]
    synth = c["synthetic_performance_model"]["r2_log"]
    real = c["real_news_popularity"]["r2_log"]
    assert synth > real, "the simulated benchmark is no longer the easier task"
    assert "inverts the generator" in c["reading"]


def test_published_comparison_cites_something_checkable(results):
    pub = results["classification"]["published_comparison"]
    assert pub["url"].startswith("https://")
    assert "NOT verified" in pub["note"]
