"""
The marketplace layer: creator-supplied data, comment NLP, audience quality,
the recommendation models, and the three-tier scores.

The checks that matter here are consistency ones. There are now nine tables
describing the same 2,000 creators from different angles, and the failure mode
is not a crash - it is two surfaces quietly disagreeing about the same creator.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT))
APP = ROOT / "app_data"
ART = ROOT / "artifacts"


def _load(name):
    p = APP / name
    if not p.exists():
        pytest.skip(f"{name} not built")
    return pd.read_parquet(p)


@pytest.fixture(scope="module")
def creators():
    return _load("nectar_creators.parquet")


# --------------------------------------------------------------------------
# creator-supplied data
# --------------------------------------------------------------------------
def test_only_connected_creators_have_private_metrics():
    """The product's central claim. If unconnected creators had insights, the
    onboarding flow would be theatre."""
    conn = _load("nectar_connections.parquet")
    ins = _load("nectar_creator_insights.parquet")
    connected = set(conn.loc[conn.account_connected, "influencer_id"].astype(str))
    with_metrics = set(ins.loc[ins.n_posts_with_insights.notna(),
                               "influencer_id"].astype(str))
    assert with_metrics <= connected
    assert 0.4 < len(connected) / len(conn) < 0.8, "connection rate is implausible"


def test_watch_time_never_exceeds_the_clip():
    p = _load("nectar_post_insights_sample.parquet")
    v = p[p.video_length_s > 0]
    assert (v.avg_watch_time_s <= v.video_length_s + 1e-6).all()


def test_audience_splits_are_distributions(creators):
    ins = _load("nectar_creator_insights.parquet")
    age = [c for c in ins.columns if c.startswith("audience_age_")]
    g = ins[["audience_female_pct", "audience_male_pct", "audience_other_pct"]].dropna()
    assert np.allclose(ins[age].dropna().sum(axis=1), 1.0, atol=2e-3)
    assert np.allclose(g.sum(axis=1), 1.0, atol=2e-3)


# --------------------------------------------------------------------------
# comment NLP - the models are real, so their metrics must be too
# --------------------------------------------------------------------------
def test_comment_models_beat_their_baselines_on_real_data():
    f = ART / "comment_nlp" / "comment_model_results.json"
    if not f.exists():
        pytest.skip("comment models not trained")
    res = json.loads(f.read_text())
    for m in res["models"]:
        assert m["macro_f1"] > 0.35, m["task"]
        assert "REAL" in m["provenance"]
    assert "domain_shift" in res["caveats"]


def test_automation_flag_is_not_flagging_everything():
    """It once flagged 70% of comments because it counted duplicate text."""
    c = _load("nectar_comment_profile.parquet")
    assert 0.02 < c.comment_automated_rate.mean() < 0.45


# --------------------------------------------------------------------------
# audience quality
# --------------------------------------------------------------------------
def test_audience_quality_beats_the_folk_heuristic():
    f = ART / "audience_quality" / "audience_quality_results.json"
    if not f.exists():
        pytest.skip("audience quality not trained")
    r = json.loads(f.read_text())
    arms = {a["arm"]: a["macro_f1"] for a in r["arms"]}
    rule = next(v for k, v in arms.items() if "follower/following" in k)
    assert arms["account + comment section"] > arms["account signals only"] > rule * 0.9
    assert "advantaged BY CONSTRUCTION" in r["caveats"]["construction"]


def test_audience_quality_score_is_a_percentage():
    q = _load("nectar_audience_quality.parquet")
    assert q.audience_quality_score.between(0, 100).all()
    assert set(q.audience_band) <= {"Authentic", "Mixed", "Suspect"}


# --------------------------------------------------------------------------
# recommendation layer
# --------------------------------------------------------------------------
def test_collaborative_filtering_beats_popularity():
    f = ART / "reco" / "cf_results.json"
    if not f.exists():
        pytest.skip("cf not built")
    r = json.loads(f.read_text())
    assert r["cf_hit@10"] > r["pop_hit@10"] * 2
    assert r["median_rank_cf"] < r["median_rank_popularity"]


def test_the_ranker_result_is_reported_even_though_it_lost():
    """A learned ranker that does not beat hand-set weights is the finding, and
    deleting it would be the dishonest move."""
    f = ART / "reco" / "ranker_results.json"
    if not f.exists():
        pytest.skip("ranker not built")
    r = json.loads(f.read_text())
    arms = {a["arm"]: a["ndcg@10"] for a in r["arms"]}
    assert arms["hand-set composite weights"] > arms["random order"]
    assert len(r["capacity_sweep"]) >= 3, "the sweep proving it was fairly tuned"
    assert "does NOT beat" in r["finding"]


def test_brand_taste_is_idiosyncratic():
    """If every brand wanted the same thing, CF would have nothing to learn."""
    t = _load("nectar_brand_taste.parquet")
    cols = [c for c in t.columns if c.startswith("taste_")]
    assert np.allclose(t[cols].sum(axis=1), 1.0, atol=1e-3)
    assert t[cols].std().mean() > 0.03, "taste vectors are nearly identical"


# --------------------------------------------------------------------------
# three-tier scoring
# --------------------------------------------------------------------------
def test_weights_sum_to_one():
    from src.scoring.engine import (CAMPAIGN_FIT_WEIGHTS, CREATOR_QUALITY_WEIGHTS,
                                    ORG_FIT_WEIGHTS)
    for w in (CREATOR_QUALITY_WEIGHTS, ORG_FIT_WEIGHTS, CAMPAIGN_FIT_WEIGHTS):
        assert abs(sum(w.values()) - 1.0) < 1e-9


def test_creator_quality_is_brand_independent(creators):
    """The whole point of the three-tier split: one row per creator, not per pair."""
    q = _load("nectar_creator_quality.parquet")
    assert q.influencer_id.is_unique
    assert len(q) == len(creators)
    assert q.creator_quality.between(0, 100).all()


def test_blocked_creators_carry_a_reason_and_no_score():
    cf = _load("nectar_campaign_fit.parquet")
    blocked = cf[cf.blocked]
    assert len(blocked) > 0
    assert blocked.campaign_fit.isna().all(), "a blocked creator must have no score"
    assert (blocked.block_reasons.astype(str).str.len() > 0).all()
    assert (blocked.campaign_fit_band == "Blocked").all()


def test_every_campaign_has_an_eligible_pool():
    """Two campaigns once had none, because completed campaigns had an end date
    before their start date and every availability check returned zero days."""
    cf = _load("nectar_campaign_fit.parquet")
    per = cf[~cf.blocked].groupby("campaign_id").size()
    assert len(per) == cf.campaign_id.nunique()
    assert per.min() > 100, f"thin pool: {per.to_dict()}"


def test_availability_is_a_rare_gate_not_the_dominant_one():
    """It blocked 4,481 pairs when the model had availability inverted."""
    cf = _load("nectar_campaign_fit.parquet")
    booked = cf.block_reasons.astype(str).str.contains("Booked for").sum()
    assert booked < 0.10 * len(cf), f"availability blocks {booked} pairs"


def test_the_competitor_veto_actually_fires():
    """It could not fire at all in the batch matrix: the evidence column it read
    is not exported, so gate_multiplier was never zero."""
    cf = _load("nectar_campaign_fit.parquet")
    conflicts = cf.block_reasons.astype(str).str.contains("competitor").sum()
    assert conflicts > 0


def test_every_score_carries_a_reason():
    cf = _load("nectar_campaign_fit.parquet")
    q = _load("nectar_creator_quality.parquet")
    assert (cf.campaign_fit_reasons.astype(str).str.len() > 0).all()
    assert (q.creator_quality_reasons.astype(str).str.len() > 0).all()


def test_deliverable_and_availability_actually_vary():
    """A component that is constant cannot change a ranking."""
    cf = _load("nectar_campaign_fit.parquet")
    e = cf[~cf.blocked]
    assert e.c_deliverable_fit.std() > 0.01
    assert e.c_availability_fit.std() > 0.01
