"""
Integrity checks for the Nectar product layer.

These are not unit tests of helper functions. Each one asserts a claim the
interface makes to a user, so that a claim cannot silently become false:
if the Campaigns table says a campaign spent 7.5 lakh, the request pipeline
has to contain 7.5 lakh of accepted fees.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_DATA = ROOT / "app_data"


def load(name: str) -> pd.DataFrame:
    p = APP_DATA / name
    assert p.exists(), f"{name} is missing — run python -m src.features.export_nectar"
    return pd.read_parquet(p)


@pytest.fixture(scope="module")
def d():
    return {
        "creators": load("nectar_creators.parquet"),
        "campaigns": load("nectar_campaigns.parquet"),
        "fit": load("nectar_fit.parquet"),
        "requests": load("nectar_requests.parquet"),
        "funnel": load("nectar_funnel.parquet"),
        "summary": load("nectar_campaign_summary.parquet"),
        "perf": load("nectar_creator_performance.parquet"),
        "earnings": load("nectar_earnings.parquet"),
        "meta": json.loads((APP_DATA / "nectar_meta.json").read_text()),
    }


# --------------------------------------------------------------------------
# The numbers on one page must equal the numbers on another
# --------------------------------------------------------------------------

def test_campaign_spend_equals_accepted_fees(d):
    """Campaigns, Overview and Reporting all quote spend. They must agree."""
    accepted = d["requests"][d["requests"].stage_index >= 4]
    by_campaign = accepted.groupby("campaign_id").fee_inr.sum()
    for c in d["campaigns"].itertuples():
        stated = float(c.spent_inr)
        actual = float(by_campaign.get(c.campaign_id, 0.0))
        assert abs(stated - actual) < 1.0, (
            f"{c.name}: Campaigns says {stated:,.0f} but the pipeline holds "
            f"{actual:,.0f} of accepted fees")


def test_creators_count_matches_pipeline(d):
    signed = d["requests"][d["requests"].stage_index >= 4].groupby(
        "campaign_id").influencer_id.nunique()
    for c in d["campaigns"].itertuples():
        assert int(c.creators_count) == int(signed.get(c.campaign_id, 0))


def test_no_campaign_overspends_its_budget(d):
    """A brand cannot commit more than it has. This failed by 25-90% before the
    budget check was added to the request builder."""
    for c in d["campaigns"].itertuples():
        assert c.spent_inr <= c.budget_inr + 1, (
            f"{c.name} committed {c.spent_inr:,.0f} against a budget of "
            f"{c.budget_inr:,.0f}")


def test_funnel_is_monotonic(d):
    """A request at stage k has by definition passed every earlier stage, so the
    cumulative counts can only fall."""
    for cid, g in d["funnel"].groupby("campaign_id"):
        counts = g.sort_values("stage_index")["count"].to_numpy()
        assert (np.diff(counts) <= 0).all(), f"{cid} funnel goes up: {counts}"


def test_progress_matches_spend(d):
    c = d["campaigns"]
    expected = (c.spent_inr / c.budget_inr).round(3)
    assert np.allclose(c.progress, expected, atol=0.002)


# --------------------------------------------------------------------------
# The model, not decoration
# --------------------------------------------------------------------------

def test_fit_components_recompose(d):
    """fit_composite must be the brand-fit composite, not an independent number."""
    f = d["fit"]
    assert np.allclose(f.fit_composite, (f.brand_fit_ungated * 100).round(1))


def test_displayed_fit_is_a_percentile_and_uses_the_range(d):
    """The card shows a percentile because the raw composite does not spread.

    Before this change the displayed 0-100 score had an inter-quartile range of
    five points, so two cards reading 86 and 84 were indistinguishable in fact.
    """
    f = d["fit"]
    assert f.campaign_fit.between(0, 100).all()
    for cid, g in f.groupby("campaign_id"):
        q1, q3 = g.campaign_fit.quantile([0.25, 0.75])
        assert q3 - q1 > 35, f"{cid}: displayed fit still bunched ({q1:.0f}-{q3:.0f})"
    # and it must still be a monotone transform of the composite it came from
    for cid, g in f.groupby("campaign_id"):
        assert g[["fit_composite", "campaign_fit"]].corr(method="spearman").iloc[0, 1] > 0.999


def test_org_fit_uses_its_stated_weights(d):
    f = d["fit"]
    expected = ((0.45 * f.fit_content_safety + 0.35 * f.fit_consistency
                 + 0.20 * f.fit_audience_match) * 100).round(1)
    assert np.allclose(f.org_composite, expected)


def test_brief_fee_is_the_price_model(d):
    """Fees quoted to creators are the price model's per-deliverable rates
    multiplied by the brief, not a random draw."""
    f = d["fit"]
    c = d["creators"].set_index("influencer_id")
    sample = f.sample(n=min(200, len(f)), random_state=0)
    for r in sample.itertuples():
        cr = c.loc[r.influencer_id]
        want = sum(cr[f"rate_{k.lower()}"] * dd["qty"]
                   for dd in r.deliverables
                   for k in [dd["type"]])
        assert abs(r.brief_fee_inr - round(want, -2)) <= 100


def test_requests_only_go_to_eligible_creators(d):
    f = d["fit"].set_index(["campaign_id", "influencer_id"])
    for r in d["requests"].itertuples():
        key = (r.campaign_id, r.influencer_id)
        assert key in f.index
        assert bool(f.loc[key, "eligible"]), (
            f"{r.creator_name} was approached for {r.campaign_name} despite failing "
            f"an eligibility gate")


def test_blocked_creators_are_never_approached(d):
    f = d["fit"]
    blocked = set(zip(f[f.blocked].campaign_id, f[f.blocked].influencer_id))
    approached = set(zip(d["requests"].campaign_id, d["requests"].influencer_id))
    assert not (blocked & approached), "a brand-safety block was overridden"


def test_ranking_objectives_actually_differ(d):
    """Three ranking modes that produce the same order are one ranking mode."""
    f = d["fit"]
    cid = f.campaign_id.iloc[0]
    g = f[f.campaign_id == cid]
    top_best = set(g.nsmallest(20, "rank_best").influencer_id)
    top_reach = set(g.nsmallest(20, "rank_reach").influencer_id)
    top_eng = set(g.nsmallest(20, "rank_engagement").influencer_id)
    assert len(top_best & top_reach) < 18
    assert len(top_best & top_eng) < 18
    assert len(top_reach & top_eng) < 18


def test_reach_ranking_favours_larger_creators(d):
    f = d["fit"]
    g = f[f.campaign_id == f.campaign_id.iloc[0]]
    assert (g.nsmallest(20, "rank_reach").followers.median()
            > g.nsmallest(20, "rank_engagement").followers.median())


# --------------------------------------------------------------------------
# Nothing that should be hidden is exposed
# --------------------------------------------------------------------------

BANNED = ("latent", "content_quality", "authenticity_score", "promo_saturation",
          "niche_focus", "true_", "_true", "campaign_engagement")


def test_no_latent_traits_reach_the_product_layer(d):
    for name, df in d.items():
        if not isinstance(df, pd.DataFrame):
            continue
        for col in df.columns:
            assert not any(b in col.lower() for b in BANNED), (
                f"{name}.{col} looks like ground truth the product should not carry")


def test_presentation_fields_are_deterministic():
    """A creator must be the same person on every rebuild."""
    import sys
    sys.path.insert(0, str(ROOT))
    from src.nectar import names as N
    a = [N.display_name(f"INF{i:05d}", 0.5) for i in range(50)]
    b = [N.display_name(f"INF{i:05d}", 0.5) for i in range(50)]
    assert a == b
    assert len(set(a)) > 25, "names collapse onto too few values"


def test_handles_are_unique(d):
    c = d["creators"]
    assert c.nectar_handle.nunique() == len(c)


def test_audience_distributions_sum_to_100(d):
    c = d["creators"].sample(n=200, random_state=0)
    for r in c.itertuples():
        assert sum(x["pct"] for x in r.audience_age) == 100
        assert sum(x["pct"] for x in r.audience_gender) == 100
        assert sum(x["pct"] for x in r.audience_locations) == 100


def test_semantic_extension_is_declared(d):
    """Rows whose semantic similarity is a bound rather than an SBERT score must
    say so, or the number is a quiet fabrication."""
    f = d["fit"]
    assert "semantic_imputed" in f.columns
    assert f.semantic_imputed.any()
    stats = d["meta"].get("semantic_extension", {})
    assert stats.get("n_bounded", 0) > 0
    method = stats.get("method", "").lower()
    assert "floor" in method or "bound" in method


def test_provenance_is_recorded(d):
    prov = d["meta"].get("provenance", {})
    for key in ("campaign_fit", "org_fit", "fees", "predicted_vs_actual",
                "simulated_here"):
        assert key in prov and len(prov[key]) > 20


# --------------------------------------------------------------------------
# Reporting honesty
# --------------------------------------------------------------------------

def test_predicted_vs_actual_comes_from_out_of_fold_predictions(d):
    """The calibration table must match what the model actually produced."""
    import numpy as np
    oof = np.load(ROOT / "artifacts" / "models" / "performance_oof.npy")
    mt = pd.read_parquet(ROOT / "artifacts" / "features" / "modelling_table.parquet")
    mt = mt.assign(pred=oof)
    want = mt.groupby("brand_category").agg(
        predicted_er=("pred", "mean"),
        actual_er=("campaign_engagement_rate", "mean")).reset_index()
    got = load("nectar_calibration.parquet").rename(
        columns={"brand_category": "brand_category"})
    merged = want.merge(got, on="brand_category", suffixes=("_want", "_got"))
    assert len(merged) == len(want)
    assert np.allclose(merged.predicted_er_want, merged.predicted_er_got)
    assert np.allclose(merged.actual_er_want, merged.actual_er_got)


def test_cpe_is_cost_over_engagements(d):
    p = d["perf"]
    live = p[p.engagements > 0]
    assert np.allclose(live.cpe, (live.cost / live.engagements).round(2), atol=0.02)
